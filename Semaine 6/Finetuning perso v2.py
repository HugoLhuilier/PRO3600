import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import time
import gc
from torch.utils.tensorboard import SummaryWriter
import os
import math

ver = "6.1"
writer = SummaryWriter(log_dir="runs/V"+ver)

model = AutoModelForCausalLM.from_pretrained("../Semaine 4/finetuned_model_v5.3")
tokenizer = AutoTokenizer.from_pretrained("../Semaine 4/finetuned_model_v5.3")
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Running on : "+ str(device))


nbBatches = 2506122 #nombre de batches indiqué dans data_stats.json

src = "./Batches3"
stories_train = 650
check_interval = 50
story_checkpoint = 100
batch_size = 2
chunk = 1024   #Nombre de tokens par bloc
loss = nn.CrossEntropyLoss().to(device) #loss function
lr = 1e-4   #learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)   #optimizer : adapte les paramètres du modèle après chaque batch
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95) #scheduler : réduit le learning rate au fil du temps
epochs = 1
savePath = r"./finetuned_model_v"+ver

stats_list = []
stats_list.append(pd.Series(epochs, name="epochs"))
stats_list.append(pd.Series(batch_size, name="batch_size"))
stats_list.append(pd.Series(chunk, name="chunk"))
nullTensor = torch.tensor([[-100] * chunk] * batch_size)

tokenizer.pad_token = tokenizer.eos_token

class StoryDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = src+"/"+data_dir+"/"
        self.i_min = math.inf
        self.i_max = -1
        
        tmp = 0
        
        for f in os.listdir(self.data_dir):           
            if not "Batch" in f:
                continue

            tmp = int(f[6:].split(".")[0])   #On récupère le numéro de Batch
            if self.i_min > tmp:
                self.i_min = tmp
        if self.i_max < tmp:
            self.i_max = tmp
            
            
    
    def __len__(self):
        return self.i_max - self.i_min + 1
    
    def __getitem__(self, i):
        index = i + self.i_min
        b = torch.load(self.data_dir+"Batch "+str(index)+".pt")
        pSize = chunk - len(b)
        return nn.functional.pad(b, (0,pSize)), nn.functional.pad(torch.load(self.data_dir+"Mask "+str(index)+".pt"), (0,pSize)), nn.functional.pad(torch.load(self.data_dir+"Labels "+str(index)+".pt"), (0,pSize), "constant", -100)

def train(loader: DataLoader):  #Entraîne le modèle suivant le loader et renvoie le loss moyen et le temps d'exécution
    #print("Starting training...")
    t0 = time.time()
    total_loss = 0
    model.train()
    
    nb_loss = 0
    l = len(loader)
    t0Log = time.time()
    nbSkip = 0
    
    for step, batch in enumerate(loader):
        labels = batch[2].type(torch.LongTensor)
        
        if torch.equal(nullTensor, labels):
            nbSkip += 1
            continue
        
        model.zero_grad()
        inputs = batch[0].type(torch.LongTensor).to(device)
        mask = batch[1].to(device)
        labels = labels.to(device)
        
        outputs = model(inputs, attention_mask = mask, labels = labels)
        loss = outputs[0]
        batch_loss = loss.item()
        total_loss += batch_loss 
        nb_loss += 1
    
        loss.backward()
        optimizer.step()
        
        if time.time() - t0Log > 1:
            print(str(step)+"/"+str(l),"batches,",nbSkip,"skipped")
            t0Log = time.time()
            
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        """
    
    loss=total_loss/nb_loss
    t=time.time() - t0
    print("")
    print("=== Story from", loader.dataset.data_dir, "===")
    print("Average loss : "+str(loss)+",", nbSkip, "skipped, elapsed time : "+str(t)+" -------------------")
    print("")
    return loss, t

def checkpoint(storiesSaved, nb):
    p = savePath+"/checkpoint "+ str(nb) + " stories"
    
    if not os.path.exists(p):
        os.makedirs(p)
        
    pd.DataFrame(storiesSaved).to_csv(p+"/listStories.csv")
    model.save_pretrained(p) 
    print("CHECKPOINT : ", p)
    

def evalu(loader: DataLoader):   #Evalue le modèle suivant le loader et renvoie le loss moyen et le temps d'exécution
    print("------------------- Evaluation -------------------")
    print("")
    model.eval()
    t0 = time.time()
    total_loss = 0
    for step, batch in enumerate(loader):
        inputs = batch[0].type(torch.LongTensor).to(device)
        mask = batch[1].to(device)
        
        with torch.no_grad():
            outputs = model(inputs, attention_mask = mask, labels = inputs)
            loss = outputs[0]
            batch_loss = loss.item()
            total_loss += batch_loss
            
            #writer.add_scalar("test_loss", batch_loss, step*batch_size)
            
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        """
        
    loss=total_loss/len(loader)
    t=time.time() - t0
    print("")
    print("=== Story from", loader.dataset.data_dir, "===")
    print("Average loss : "+str(loss)+", elapsed time : "+str(t))
    print("")
    return loss, t
    
stories_data = [StoryDataset(p) for p in os.listdir(src)]
trainData = stories_data[:stories_train]
testData = stories_data[stories_train:]

trainLoaders = [DataLoader(d, batch_size, shuffle = True) for d in trainData]
testLoaders = [DataLoader(d, batch_size, shuffle = True) for d in testData]


def start():
    print("======================================================")
    print("====================== Starting ======================")
    print("======================================================")
    print("")
    
    tTime = 0
    
    for i in range(epochs):
        eLoss = 0
        
        
        print("------------------- epoch "+str(i+1)+"/"+str(epochs))
        print("")
        
        stepStory = 0
        sSaved = []
        
        for trainLoader in trainLoaders:   
            loss, t = train(trainLoader)
            tTime += t
            eLoss += loss
            
            sSaved.append(trainLoader.dataset.data_dir)
            stepStory += 1
            print("(Story", stepStory, ")")
            if (stepStory % story_checkpoint == 0):
                checkpoint(sSaved, stepStory)
        
        avLoss = eLoss/len(trainLoaders)
        print("Epoch", i, "completed. Average loss :", avLoss, "; learning rate :", scheduler.get_last_lr())
        if not os.path.exists(savePath+"/epoch"):
            os.makedirs(savePath+"/epoch")
        model.save_pretrained(savePath+"/epoch", i)
        print("Model saved at location", savePath+"/epoch", "-------------------")
        print("")
        writer.add_scalar("train loss by epoch", avLoss)
        writer.add_scalar("learning rate by epoch", scheduler.get_last_lr()[0])
        scheduler.step()
    
    if not os.path.exists(savePath+"/final"):
        os.makedirs(savePath+"/final") 
    model.save_pretrained(savePath+"/final")
    tokenizer.save_pretrained(savePath)
    print("======================================================")
    print("Training complete ! Model saved at location : "+savePath+"/final")
    
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
       
    pd.DataFrame(stats_list).to_csv(savePath+"/stats.csv")
