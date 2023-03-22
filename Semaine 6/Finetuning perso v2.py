import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import time
import gc
from torch.utils.tensorboard import SummaryWriter
import os

ver = "6.0"
writer = SummaryWriter(log_dir="runs/V"+ver)

model = AutoModelForCausalLM.from_pretrained("../Semaine 4/finetuned_model_v5.3")
tokenizer = AutoTokenizer.from_pretrained("../Semaine 4/finetuned_model_v5.3")
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Running on : "+ str(device))


nbBatches = 3051842 #nombre de batches indiqué dans data_stats.json

src = "./Batches"
stories_train = 600
check_interval = 50
batch_size = 2
chunk = 1024   #Nombre de tokens par bloc
loss = nn.CrossEntropyLoss().to(device) #loss function
lr = 1e-4   #learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)   #optimizer : adapte les paramètres du modèle après chaque batch
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.95) #scheduler : réduit le learning rate au fil du temps
epochs = 1
savePath = r"./finetuned_model_v"+ver

stats_list = []
stats_list.append(pd.Series(epochs, name="epochs"))
stats_list.append(pd.Series(batch_size, name="batch_size"))
stats_list.append(pd.Series(chunk, name="chunk"))

tokenizer.pad_token = tokenizer.eos_token

class StoryDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.i_min = -1
        tmp = ""
        
        for f in os.listdir(data_dir):
            if not "Batch" in f:
                break
            tmp = f[7:] #On récupère le numéro de Batch
            if self.i_min == -1:
                self.i_min = int(tmp)
        self.i_max = int(tmp)
            
            
    
    def __len__(self):
        return self.i_max - self.i_min + 1
    
    def __getitem__(self, i):
        index = i + self.i_min
        return torch.load("Batch "+str(index)+".pt"), torch.load("Mask "+str(index)+".pt"), torch.load("Labels "+str(index)+".pt")

def train(loader: DataLoader, e):  #Entraîne le modèle suivant le loader et renvoie le loss moyen et le temps d'exécution
    print("Starting training...")
    t0 = time.time()
    total_loss = 0
    model.train()
    l = len(loader)
    
    nb_loss = 0
    
    for step, batch in enumerate(loader):
        model.zero_grad()
        inputs = batch[0].to(device)
        mask = batch[1].to(device)
        labels = batch[2].type(torch.LongTensor).to(device)
        outputs = model(inputs, attention_mask = mask, labels = labels)
        loss = outputs[0]
        batch_loss = loss.item()
        nlr = scheduler.get_last_lr()[0]
        
        if labels[0][0] != -100:
            total_loss += batch_loss
            nb_loss +=1
            writer.add_scalar("train_loss", batch_loss, step*batch_size + l*e*batch_size)
        
        writer.add_scalar("learning_rate", nlr, step*batch_size + l*e*batch_size)
        
        if step % check_interval == 0 and step != 0:
            print("Batch : "+ str(step+1) +" of "+ str(l)+", loss : "+ str(batch_loss)+ ", learning rate : "+ str(nlr) +", time : "+str(time.time() - t0))
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        """
    
    loss=total_loss/nb_loss
    t=time.time() - t0
    print("")
    print("=== Results ===")
    print("Average loss : "+str(loss)+", elapsed time : "+str(t)+" -------------------")
    print("")
    return loss, t

def evalu(loader: DataLoader):   #Evalue le modèle suivant le loader et renvoie le loss moyen et le temps d'exécution
    print("------------------- Evaluation -------------------")
    print("")
    model.eval()
    t0 = time.time()
    total_loss = 0
    l = len(loader)
    for step, batch in enumerate(loader):
        inputs = batch[0].to(device)
        mask = batch[1].to(device)
        
        with torch.no_grad():
            outputs = model(inputs, attention_mask = mask, labels = inputs)
            loss = outputs[0]
            batch_loss = loss.item()
            total_loss += batch_loss
            
            #writer.add_scalar("test_loss", batch_loss, step*batch_size)
        
        if step % check_interval == 0 and step != 0:
            print("Batch : "+ str(step+1) +" of "+ str(l)+", loss : "+ str(batch_loss)+", time : "+str(time.time() - t0))
        
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        """
        
    loss=total_loss/len(loader)
    t=time.time() - t0
    print("")
    print("=== Results ===")
    print("Average loss : "+str(loss)+", elapsed time : "+str(t))
    print("")
    return loss, t
    
stories_data = [StoryDataset(p) for p in os.listdir(src)]
trainData = stories_data[:stories_train]
testData = stories_data[stories_train:]

trainLoaders = [DataLoader(d, batch_size) for d in trainData]
testLoaders = [DataLoader(d, batch_size, shuffle = True) for d in testData]


def start():
    print("======================================================")
    print("====================== Starting ======================")
    print("======================================================")
    print("")
    
    tTime = 0
    
    for i in range(epochs):
        print("------------------- epoch "+str(i+1)+"/"+str(epochs))
        print("")
        
        for trainLoader in trainLoaders:   
            loss, t = train(trainLoader, i)
            tTime += t
        
    stats_list.append(pd.Series(loss, name="train_loss"))
    stats_list.append(pd.Series(tTime, name="train_time"))
    stats_list.append(pd.Series(len(trainData), name="train_batches"))
    stats_list.append(pd.Series(lr, name="init_learning_rate"))
    stats_list.append(pd.Series(scheduler.get_last_lr(), name="last_learning_rate"))
    
    for testLoader in testLoaders:
        loss, t = evalu(testLoader)
    
    stats_list.append(pd.Series(loss, name="test_loss"))
    stats_list.append(pd.Series(t, name="test_time"))
    stats_list.append(pd.Series(len(testData), name="test_batches"))
    
    
    model.save_pretrained(savePath)
    tokenizer.save_pretrained(savePath)
    pd.DataFrame(stats_list).to_csv(savePath+"/stats.csv")
    print("======================================================")
    print("Training complete ! Model saved at location : "+savePath)
