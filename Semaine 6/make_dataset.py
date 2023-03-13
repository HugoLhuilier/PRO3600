import json 
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("../Semaine 4/finetuned_model_v5.3")


nbTokens = 0
toBatch = ""
appLoss = False
batches = {"text": [], "apply_loss": [], "storyID": []}
stats = {"nbBatches": 0, "nbTokens": 0, "nbPaths": 0, "nbStories": 0}

def append_batches():
    global toBatch, nbTokens, appLoss, stats, sId
    batches["text"].append(toBatch)
    batches["apply_loss"].append(1 if appLoss else 0)   #1 si on applique le loss au batch, 0 sinon
    batches["storyID"].append(sId)
    stats["nbTokens"] += nbTokens
    toBatch = ""
    nbTokens = 0
    appLoss = False
    stats["nbBatches"] += 1

with open("../Resources/Stories 11.03/all_paths_v2.json", "r") as r:
    data = json.load(r)
    lData = len(data)
    nStory = 0
    stats["nbStories"] = lData
    
    for story in data:
        nStory += 1
        print("Story", nStory, "of", lData)
        
        lStory = len(story["paths"])
        nPath = 0
        stats["nbPaths"] += lStory
        
        sId = story["storyID"]
        for path in story["paths"]:
            nPath += 1
            print("----- Path", nPath, "of", lStory)
            
            for seg in path:
                tmp = seg[0] + "\n\n"
                segTokens = len(tok(tmp, truncation = True)["input_ids"])   #On tronque les passages potentiellement trop longs
                if nbTokens + segTokens > 1024:
                    append_batches()
                
                toBatch += tmp
                nbTokens += segTokens
                
                if(seg[1] != 0):    #On applique le loss dès qu'au moins une section du batch n'a pas été rencontré
                    appLoss = True
                
            if nbTokens != 0:
                append_batches()

pd_data = pd.DataFrame(batches)
pd_data.to_csv("batched_data.csv")

with open("data_stats.json", "w") as f:
    json.dump(stats, f)

print("Done")