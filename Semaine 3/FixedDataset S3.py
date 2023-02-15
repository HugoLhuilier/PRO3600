import pandas as pd
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2") #Tokenizer

src = "D:\Documents (D)\Ecole\TSP\1A\PRO3600\Semaine 2\FixedData.txt"   #Dataset de base

data = pd.Series([], name="Scripts")    #Série pandas contenant les scripts découpés 

tmp = ""

with open(src, "r", encoding='utf-8') as fic:
    for l in fic:
        