import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
import pandas as pd
import time

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model.cuda()
print("Running on : "+ str(device))


srcTest = "../Semaine 3/FixedDataset S3 with summary test.csv"
srcTrain = "../Semaine 3/FixedDataset S3 with summary train.csv"
check_interval = 200
batch_size = 8
chunk = 1024    #Nombre de tokens par bloc
loss = nn.CrossEntropyLoss() #loss function
lr = 0.01   #learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)   #optimizer : adapte les paramètres du modèle après chaque batch
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95) #scheduler : réduit le learning rate au fil du temps


class ScriptDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = pd.read_csv(data_dir)
    
    def __len__(self):
        return self.data.size
    
    def __getitem__(self, i):
        encoded = tokenizer(self.data.iloc[i][0], truncation = True, max_length = chunk)
        return torch.tensor(encoded["input_ids"]), torch.tensor(encoded["attention_mask"])

def train(loader: DataLoader):
    total_loss = 0
    model.train()
    for step, batch in enumerate(loader):
        model.zero_grad()
        inputs = batch[0].to(device)
        mask = batch[1].to(device)
        outputs = model(inputs, attention_mask = mask, labels = inputs)
        loss = outputs.loss
        
        batch_loss = loss.item()
        total_loss += batch_loss
    
trainData = ScriptDataset(srcTrain)
testData = ScriptDataset(srcTest)

trainLoader = DataLoader(trainData, batch_size=1, shuffle=True)
testLoader = DataLoader(testData, batch_size=1, shuffle=True)    #batch size 2* plus grand car evaluation moins coûteuse en ressources que le training