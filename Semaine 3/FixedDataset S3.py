import pandas as pd
from transformers import AutoTokenizer
from statistics import mean

tok = AutoTokenizer.from_pretrained("gpt2") #Tokenizer

src = r"C:\Users\user\Documents\Ecole\TSP\1A\PRO3600\PRO3600\Semaine 2\FixedData.txt"   #Dataset de base

data = []    #Série pandas contenant les scripts découpés 

#Statistiques
nbScripts = 0
tokScripts = [0]     #Tableau contenant le nombre de tokens dans chaque script


#Variables de la boucle
emptLine = 0    #Nombre de lignes vides consécutives
tmp = ""    #Chaîne de caractères temporaire
prev = ""   #Chaîne de caractère précédente que l'on rajoute au début du prochain bloc
prevTok = 0     #Nombre de tokens de la chaîne précédente
nbTokens = 0    #Nombre de tokens d'un bloc


with open(src, "r", encoding='utf-8') as fic:
    for l in fic:
        if (not l.strip()) or "---" in l or "___" in l or "===" in l:   #On n'ajoute pas les lignes vides, et on compte combien il y en a d'affilée
            emptLine += 1
        else:
            if "THE ENDTHE END" in l: #On teste s'il s'agit de la fin d'un script, et on termine le bloc
                data.append(tmp)
                tmp = ""
                nbTokens = 0
                nbScripts += 1
                tokScripts.append(0)
            else:           
                lToken = len(tok(l)["input_ids"])   #Nombre de tokens dans la ligne
                if lToken + nbTokens > 1021:    #Si on dépasse le nombre de tokens max par bloc (avec tolérance de 3 pour des éventuels sauts de ligne et token spécial), on insère la ligne dans un nouveau bloc
                    data.append(tmp)
                    tokScripts[-1] += nbTokens
                    tmp = prev + l
                    nbTokens = lToken + prevTok
                    prevTok = lToken
                    prev = l
                else:
                    if (nbScripts == 0 and emptLine >= 2) or (nbScripts > 0 and emptLine >= 1):  #On teste si on est encore dans le script de Skyrim et s'il y a plus de trois sauts de ligne (le premier n'est pas compté)
                        tmp += "\n"     #On ajoute un saut de ligne entre deux répliques + on augmente d'un token
                        nbTokens += 1
                    tmp += l
                    nbTokens += lToken
                    prevTok = lToken
                    prev = l
            emptLine = 0    #Réinitialisation du nombre de lignes vides
data.append(tmp)
nbScripts += 1
nbTokensMoy = mean(tokScripts)

stats = pd.DataFrame({
    "Statistique" : [
        "Nombre de scripts",
        "Nombre moyen de tokens par script"
        ],
    "Valeur" : [nbScripts, nbTokensMoy]
    })

stats.to_csv("FixedDataset S3 stats.csv")
pd.Series(data).to_csv("FixedDataset S3.csv", index = False)