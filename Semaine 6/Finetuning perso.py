import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import time
import gc
from torch.utils.tensorboard import SummaryWriter

ver = "6.0"
writer = SummaryWriter(log_dir="runs/V"+ver)

model = AutoModelForCausalLM.from_pretrained("../Semaine 4/finetuned_model_v5.3")
tokenizer = AutoTokenizer.from_pretrained("../Semaine 4/finetuned_model_v5.3")
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Running on : "+ str(device))


nbBatches = 2437362 #nombre de batches indiqué dans data_stats.json

src = "./batched_data.csv"
prop_train = 0.75
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

class ScriptDataset(Dataset):
    def __init__(self, data_dir, i_min=0, i_max=0):
        self.data_dir = data_dir
        self.data = pd.read_csv(data_dir)
        self.i_min = i_min
        self.i_max = i_max if i_max > 0 else (self.data.size - 1)
    
    def __len__(self):
        return self.i_max - self.i_min + 1
    
    def __getitem__(self, i):
        index = i + self.i_min
        encoded = tokenizer(self.data.iloc[index]["text"], truncation = True, max_length = chunk, padding='max_length')
        return torch.tensor(encoded["input_ids"]), torch.tensor(encoded["attention_mask"]),  ((torch.tensor(encoded["input_ids"])) if (self.data.iloc[index]["apply_loss"] == 1) else (-100 * torch.ones(len(encoded["input_ids"]))))

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
    
trainData = ScriptDataset(src, 0, int(prop_train * nbBatches))
testData = ScriptDataset(src, int(prop_train * nbBatches) + 1, nbBatches - 1)

trainLoader = DataLoader(trainData, batch_size, """shuffle=True""")
testLoader = DataLoader(testData, batch_size, shuffle=True)    


def start():
    print("======================================================")
    print("====================== Starting ======================")
    print("======================================================")
    print("")
    
    tTime = 0
    
    for i in range(epochs):
        print("------------------- epoch "+str(i+1)+"/"+str(epochs))
        print("")
        
        loss, t = train(trainLoader, i)
        tTime += t
        
    stats_list.append(pd.Series(loss, name="train_loss"))
    stats_list.append(pd.Series(tTime, name="train_time"))
    stats_list.append(pd.Series(len(trainData), name="train_batches"))
    stats_list.append(pd.Series(lr, name="init_learning_rate"))
    stats_list.append(pd.Series(scheduler.get_last_lr(), name="last_learning_rate"))
    
    loss, t = evalu(testLoader)
    
    stats_list.append(pd.Series(loss, name="test_loss"))
    stats_list.append(pd.Series(t, name="test_time"))
    stats_list.append(pd.Series(len(testData), name="test_batches"))
    
    
    model.save_pretrained(savePath)
    tokenizer.save_pretrained(savePath)
    pd.DataFrame(stats_list).to_csv(savePath+"/stats.csv")
    print("======================================================")
    print("Training complete ! Model saved at location : "+savePath)
