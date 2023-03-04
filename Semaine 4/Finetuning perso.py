import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import time

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Running on : "+ str(device))


srcTest = "../Semaine 3/FixedDataset S3 with summary test.csv"
srcTrain = "../Semaine 3/FixedDataset S3 with summary train.csv"
check_interval = 1
batch_size = 8
chunk = 1024    #Nombre de tokens par bloc
loss = nn.CrossEntropyLoss() #loss function
lr = 0.01   #learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)   #optimizer : adapte les paramètres du modèle après chaque batch
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95) #scheduler : réduit le learning rate au fil du temps
epochs = 1
savePath = r"C:\Users\user\Documents\Ecole\TSP\1A\PRO3600\PRO3600\Semaine 4\Premier test script perso"

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
        
        if step % check_interval == 0:
            print("Batch : "+ str(step+1) +" of "+ str(len(loader))+", loss : "+ str(batch_loss)+", time : "+str(time.time() - t0))
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    return total_loss/len(loader), time.time() - t0

def eval(loader: DataLoader):   #Evalue le modèle suivant le loader et renvoie le loss moyen et le temps d'exécution
    print("Starting evaluation...")
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
        
        if step % check_interval == 0 and step != 0:
            print("Batch : "+ str(step+1) +" of "+ str(len(loader))+", time : "+str(time.time() - t0))
        
        
    return total_loss / len(loader), time.time() - t0
    
trainData = ScriptDataset(srcTrain)
testData = ScriptDataset(srcTest)

trainLoader = DataLoader(trainData, 1, shuffle=True)
testLoader = DataLoader(testData, 1, shuffle=True)    


def start():
    print("======================================================")
    print("====================== Starting ======================")
    print("======================================================")
    print("")
    
    for i in range(epochs):
        print("------------------- epoch "+str(i+1)+"/"+str(epochs))
        print("")
        
        loss, t = train(trainLoader)
        
        print("")
        print("=== Results ===")
        print("Average loss : "+str(loss)+", elapsed time : "+str(t)+" -------------------")
        print("")
    
    print("------------------- Evaluation -------------------")
    print("")
    
    loss, t = eval(testLoader)
    print("")
    print("=== Results ===")
    print("Average loss : "+str(loss)+", elapsed time : "+str(t))
    print("")
    
    model.save_pretrained(savePath)
    print("======================================================")
    print("Training complete ! Model saved at location : "+savePath)