import json 
import pandas as pd
from transformers import LlamaTokenizer
from torch import tensor
import torch
import os
from torch import nn

tok = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

save_path = "../Semaine 10/Batches_lama_1024_purified"
max_tokens = 1024
nbTokens = 0
toBatch = []
mask = []
labels = []
#batches = {"text": [], "apply_loss": [], "storyID": []}
stats = {"nbBatches": 0, "nbTokens": 0, "nbPaths": 0, "nbStories": 0}

def append_batches():   
    global toBatch, nbTokens, stats, sId, toBatch, mask, labels
    
    if labels != [-100]*len(labels):   
        sPath = save_path +"/Story "+sId
        if not os.path.exists(sPath):
            os.makedirs(sPath)
            
        torch.save(nn.functional.normalize(tensor(toBatch, dtype = torch.float32), dim = 0), sPath+"/Batch " + str(stats["nbBatches"]) +".pt")
        torch.save(tensor(mask), sPath+"/Mask " + str(stats["nbBatches"]) +".pt")
        torch.save(nn.functional.normalize(tensor(labels, dtype = torch.float32), dim = 0), sPath+"/Labels " + str(stats["nbBatches"]) +".pt")
        stats["nbTokens"] += nbTokens
        stats["nbBatches"] += 1
        
    toBatch = ""
    nbTokens = 0       
    toBatch = []
    mask = []
    labels = []
    


with open("../Resources/Stories 11.03/all_paths_v2.json", "r") as r:
    with open("../Resources/Stories 11.03/all_stories.json", "r") as r2:
        i = 0   #pour parcourir all_stories.json à la recherche des bons tags
        data = json.load(r)
        tags = json.load(r2)
        lData = len(data)
        nStory = 0
        stats["nbStories"] = lData
        
        for story in data:
            sId = story["storyID"]
            
            if sId == "59840":
                continue
            
            nStory += 1
            print("Story", nStory, "of", lData)
            
            lStory = len(story["paths"])
            nPath = 0
            stats["nbPaths"] += lStory
            
            while str(tags[i]["id"]) != sId: #Les stories sont rangées dans le même ordre mais, dans le cas où l'on ignore une histoire, on fait attention à sélectionner les bons tags
                i+=1
            
            for path in story["paths"]:
                nPath += 1
                print("----- Path", nPath, "of", lStory, "(id :", sId, ")")
                tmp = str(tags[i]["tags"]) + "\n"
                
                for seg in path:
                    tmp += seg[0] + "\n\n"
                    tokens = tok(tmp, truncation = True)    #On tronque les passages potentiellement trop longs
                    segTokens = len(tokens["input_ids"])   
                    if nbTokens + segTokens > max_tokens:
                        append_batches()
                        tmp = str(tags[i]["tags"]) + "\n"
                    else:
                        tmp = ""
                    
                    toBatch += tokens["input_ids"]
                    mask += tokens["attention_mask"]
                    
                    nbTokens += segTokens
                    
                    if(seg[1] == 0):    #On applique le loss dès qu'au moins une section du batch n'a pas été rencontré
                        newLabels = [-100] * segTokens    #-100 partout si le bloc a déjà été rencontré
                    else:
                        newLabels = tokens["input_ids"]
                    
                    labels += newLabels
                    
                if nbTokens != 0:
                    append_batches()

#pd_data = pd.DataFrame(batches)
#pd_data.to_csv("batched_data.csv")

with open(save_path + "data_stats.json", "w") as f:
    json.dump(stats, f)

print("Done")