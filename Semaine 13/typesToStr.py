import json 
import os

data_path = "../../Data/Semaine_13/frag_danddbeyondDatasetv2"

os.chdir(data_path)

for path in os.listdir("./"):
    with open(path, 'r') as f:
        j = json.load(f)
    for e in j:
        if type(e) is dict:
            print(e["title"])
            e["index"] = str(e["index"])
            e["tok_size"] = str(e["tok_size"])
        else:
            print("Element not dict")
    with open(path, 'w') as f:
        json.dump(j, f)