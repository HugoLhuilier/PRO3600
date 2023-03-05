import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import time
import gc
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Running on : "+ str(device))


srcTest = "../Semaine 3/FixedDataset S3 with summary test.csv"
srcTrain = "../Semaine 3/FixedDataset S3 with summary train.csv"
check_interval = 50
batch_size = 1
chunk = 700   #Nombre de tokens par bloc
loss = nn.CrossEntropyLoss().to(device) #loss function
lr = 1e-4   #learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)   #optimizer : adapte les paramètres du modèle après chaque batch
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.95) #scheduler : réduit le learning rate au fil du temps
epochs = 1
savePath = r"./Premier test script perso"

stats_list = []
stats_list.append(pd.Series(epochs, name="epochs"))
stats_list.append(pd.Series(batch_size, name="batch_size"))
stats_list.append(pd.Series(chunk, name="chunk"))

class ScriptDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = pd.read_csv(data_dir)
    
    def __len__(self):
        return self.data.size
    
    def __getitem__(self, i):
        encoded = tokenizer(self.data.iloc[i][0], truncation = True, max_length = chunk)
        return torch.tensor(encoded["input_ids"]), torch.tensor(encoded["attention_mask"])

def train(loader: DataLoader):  #Entraîne le modèle suivant le loader et renvoie le loss moyen et le temps d'exécution
    print("Starting training...")
    t0 = time.time()
    total_loss = 0
    model.train()
    for step, batch in enumerate(loader):
        model.zero_grad()
        inputs = batch[0].to(device)
        mask = batch[1].to(device)
        outputs = model(inputs, attention_mask = mask, labels = inputs)
        loss = outputs[0]
        batch_loss = loss.item()
        total_loss += batch_loss
        nlr = scheduler.get_last_lr()[0]
        
        writer.add_scalar("train_loss", batch_loss, step)
        writer.add_scalar("learning_rate", nlr, step)
        
        if step % check_interval == 0 and step != 0:
            print("Batch : "+ str(step+1) +" of "+ str(len(loader))+", loss : "+ str(batch_loss)+ ", learning rate : "+ str(nlr) +", time : "+str(time.time() - t0))
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        """
    
    loss=total_loss/len(loader)
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
    for step, batch in enumerate(loader):
        inputs = batch[0].to(device)
        mask = batch[1].to(device)
        
        with torch.no_grad():
            outputs = model(inputs, attention_mask = mask, labels = inputs)
            loss = outputs[0]
            batch_loss = loss.item()
            total_loss += batch_loss
            
            writer.add_scalar("test_loss", batch_loss, step)
        
        if step % check_interval == 0 and step != 0:
            print("Batch : "+ str(step+1) +" of "+ str(len(loader))+", loss : "+ str(batch_loss)+", time : "+str(time.time() - t0))
        
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
    
trainData = ScriptDataset(srcTrain)
testData = ScriptDataset(srcTest)

trainLoader = DataLoader(trainData, batch_size, shuffle=True)
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
        
        loss, t = train(trainLoader)
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
