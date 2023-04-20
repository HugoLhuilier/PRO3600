#!/usr/bin/pnjIaEnv

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import LlamaForCausalLM, LlamaTokenizer
import pandas as pd
import time
import gc
from torch.utils.tensorboard import SummaryWriter
import os
import math
from accelerate import Accelerator

accelerator = Accelerator()

ver = "L1.0"

model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
print("Running with HuggingFace accelerate")



src = "../Data/Semaine_6/Batches4"
stories_train = 650
check_interval = 50
story_checkpoint = 100
batch_size = 4
chunk = 2048   #Nombre de tokens par bloc
loss = nn.CrossEntropyLoss() #loss function
lr = 1e-4   #learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)   #optimizer : adapte les paramètres du modèle après chaque batch
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95) #scheduler : réduit le learning rate au fil du temps
epochs = 5
savePath = r"../Models/finetuned_model_v"+ver

writer = SummaryWriter(log_dir=savePath+"/runs")

model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

stats_list = []
stats_list.append(pd.Series(epochs, name="epochs"))
stats_list.append(pd.Series(batch_size, name="batch_size"))
stats_list.append(pd.Series(chunk, name="chunk"))
nullTensor = torch.tensor([[-100] * chunk] * batch_size)

tokenizer.pad_token = tokenizer.eos_token

curEpoch = 0

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
        inputs = batch[0].type(torch.LongTensor)
        mask = batch[1]

        outputs = model(inputs, attention_mask = mask, labels = labels)
        loss = outputs[0]
        batch_loss = loss.item()
        
        if type(batch_loss) is float:
            total_loss += batch_loss
            nb_loss += 1

        accelerator.backward(loss)
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
    p = savePath+"/checkpoint "+ str(nb) + " stories epoch " + str(curEpoch)

    if not os.path.exists(p):
        os.makedirs(p)

    pd.DataFrame(storiesSaved).to_csv(p+"/listStories.csv")
    model.save_pretrained(p)
    tokenizer.save_pretrained(p)
    print("CHECKPOINT : ", p)


def evalu(loader: DataLoader):   #Evalue le modèle suivant le loader et renvoie le loss moyen et le temps d'exécution
    model.eval()
    t0 = time.time()
    total_loss = 0
    for step, batch in enumerate(loader):
        inputs = batch[0].type(torch.LongTensor)
        mask = batch[1]

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

trainLoaders = [accelerator.prepare(DataLoader(d, batch_size, shuffle = True)) for d in trainData]
testLoaders = [DataLoader(d, batch_size, shuffle = True) for d in testData]


def start():
    print("======================================================")
    print("====================== Starting ======================")
    print("======================================================")
    print("")

    tTime = 0


    for i in range(epochs):
        curEpoch = i+1
        eLoss = 0


        print("------------------- epoch "+str(i+1)+"/"+str(epochs))
        print("")

        stepStory = 0
        sSaved = []

        for trainLoader in trainLoaders:
            loss, t = train(trainLoader)
            tTime += t
            eLoss += loss
            writer.add_scalar("loss_per_story_epoch_"+str(curEpoch), loss, len(sSaved))

            sSaved.append(trainLoader.dataset.data_dir)
            stepStory += 1
            print("(Story", stepStory, ")")
            if (stepStory % story_checkpoint == 0):
                checkpoint(sSaved, stepStory)

        avLoss = eLoss/len(trainLoaders)
        print("Epoch", curEpoch, "completed. Average loss :", avLoss, "; learning rate :", scheduler.get_last_lr())

        writer.add_scalar("loss_per_epoch", avLoss, curEpoch)
        writer.add_scalar("learning_rate_per_epoch", scheduler.get_last_lr()[0], curEpoch)

        if not os.path.exists(savePath+"/epoch"+str(curEpoch)):
            os.makedirs(savePath+"/epoch"+str(curEpoch))
        model.save_pretrained(savePath+"/epoch"+str(curEpoch))
        tokenizer.save_pretrained(savePath+"/epoch"+str(curEpoch))
        print("Model saved at location", savePath+"/epoch"+str(curEpoch), "-------------------")
        print("")

        scheduler.step()

    if not os.path.exists(savePath+"/final"):
        os.makedirs(savePath+"/final")
    model.save_pretrained(savePath+"/final")
    tokenizer.save_pretrained(savePath+"/final")
    print("======================================================")
    print("Training complete ! Model saved at location : "+savePath+"/final")
    print("")

    stats_list.append(pd.Series(tTime, name="train_time"))
    stats_list.append(pd.Series(len(trainLoaders), name="train_stories"))


    print("====================== Evaluation ======================")
    totEvaLoss = 0
    for step, testLoader in testLoaders:
        loss, t = evalu(testLoader)
        totEvaLoss += loss
        print("story", step, "evaluated. Loss :", loss, " ; time :", t)

    stats_list.append(pd.Series(loss, name="test_loss"))
    stats_list.append(pd.Series(t, name="test_time"))
    stats_list.append(pd.Series(len(testLoaders), name="test_stories"))

    pd.DataFrame(stats_list).to_csv(savePath+"/stats.csv")


"""
if __name__ == "__main__":
    start()
"""