import pickledb
import json
from transformers import LlamaTokenizer
import re


tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")


db_path = "D:/Documents (D)/Ecole/TSP/1A/Autres ressources PRO3600/danddbeyond.db"
out_path = "D:/Documents (D)/Ecole/TSP/1A/Autres ressources PRO3600/danddbeyondDatasetv2.json"
out_frag = "D:/Documents (D)/Ecole/TSP/1A/Autres ressources PRO3600/frag_dandbeyondDatasetv2"

def toJson(e):
    try:
        return json.loads(db.get(e))
    except:
        print(f"{e} not loaded")

db = pickledb.load(db_path, False)
dataset = [toJson(e) for e in db.getall()]
nDataset = [
    {
     "title": e['title'],
     "username": e['comments'][i]['username'],
     "index": int(e['comments'][i]["index"]),
     "comment": e['comments'][i]["comment_text"],
     "tok_size": tokenizer(e['comments'][i]["comment_text"], return_length = True).length,
     }
    for e in dataset if e['last_page'] > 5   #On ne prend les threads que s'ils sont assez longs (pas de fil de recrutement)
    for i in range(0, len(e['comments']))
    ]

s = 0
i0 = 0

for i in range(len(nDataset)):
    if nDataset[i]["index"] == 1:
        i0 = i 
        s = 0
    s += nDataset[i]["tok_size"]
    while s > 2000 and i0 < i:  #On fait en sorte que la somme du nombre de tokens des messages précédent soit inférieure à 2000
        s -= nDataset[i0]["tok_size"]
        i0 += 1
    nDataset[i]["prev_comments"] = [{
                        "username": nDataset[j]['username'],
                        "comment": nDataset[j]["comment"],
                        "begin": nDataset[j]["index"] == 1
                        } for j in range(i0, i)]

with open(out_path, 'w') as f:
    json.dump(nDataset, f)
    
name = ""
t = []
for e in nDataset:
    if e["index"] == 1:
        if len(t) != 0:
            with open(out_frag + "/" + name + ".json", 'w') as f:
                json.dump(t, f)
        t = []
        name = re.sub(r'[/:*?"<>|\\]', ' ', e["title"])
    t.append(e)

with open(out_frag + "/" + name + ".json", 'w') as f:
    json.dump(t, f)