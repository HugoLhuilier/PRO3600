import pandas as pd
from transformers import AutoTokenizer, pipeline
from statistics import mean

sumPip = pipeline("summarization", model="knkarthick/MEETING_SUMMARY", truncation = True)  #Pipeline pour résumer le bloc précédent
tok = AutoTokenizer.from_pretrained("gpt2") #Tokenizer

src = r"D:\Documents (D)\Ecole\TSP\1A\PRO3600\Semaine 2\FixedData.txt"   #Dataset de base

data = []    #Liste contenant les scripts découpés

#Statistiques
nbScripts = 0
tokScripts = [0]     #Tableau contenant le nombre de tokens dans chaque script


#Variables de la boucle
emptLine = 0    #Nombre de lignes vides consécutives
tmp = ""    #Chaîne de caractères temporaire
nbTokens = 0    #Nombre de tokens d'un bloc
newScript = True   #Indique si on débute un nouveau script
nbBloc = 0


with open(src, "r", encoding='utf-8') as fic:
    for l in fic:
        
##### BLOC RAJOUTÉ APRES AVOIR REMARQUE LA DOUBLE PRESENCE DE L'APPEL A L'IA DE SUMMARY : on ne traiteque les données à partir du 24ème script #####
        if(nbScripts < 23):
            if "THE ENDTHE END" in l:
                nbScripts += 1
                print("Script count : " + str(nbScripts))
####################################################################################################################################################                
                
        elif (not l.strip()) or "---" in l or "___" in l or "===" in l:   #On n'ajoute pas les lignes vides, et on compte combien il y en a d'affilée
            emptLine += 1
        else:
            if "THE ENDTHE END" in l: #On teste s'il s'agit de la fin d'un script, et on termine le bloc
                data.append(tmp)
                tmp = ""
                nbTokens = 0
                nbScripts += 1
                tokScripts.append(0)
                newScript = True
                print("Script count : " + str(nbScripts))
            else:           
                lToken = len(tok(l)["input_ids"])   #Nombre de tokens dans la ligne
                if lToken + nbTokens > 1020:    #Si on dépasse le nombre de tokens max par bloc (avec tolérance pour des éventuels sauts de ligne et token spécial), on insère la ligne dans un nouveau bloc
                    data.append(tmp)
                    tokScripts[-1] += nbTokens
                    summary = ""
                    sumToken = 0
                    if not newScript:
                        summed = sumPip(tmp)
                        
                        summary = "[" + summed[0]["summary_text"] + "]\n\n"   #On résumé le bloc précédent et on l'inclut au début du prochain (seulement si dans le même script)
                        sumToken = len(tok(summary)["input_ids"])
                    tmp = summary + l  
                    nbTokens = lToken + sumToken
                    nbBloc += 1
                    print("Block count : " + str(nbBloc))
                else:
                    if (nbScripts == 0 and emptLine >= 2) or (nbScripts > 0 and emptLine >= 1):  #On teste si on est encore dans le script de Skyrim et s'il y a plus de trois sauts de ligne (le premier n'est pas compté)
                        tmp += "\n"     #On ajoute un saut de ligne entre deux répliques + on augmente d'un token
                        nbTokens += 1
                    tmp += l
                    nbTokens += lToken
                newScript = False
            emptLine = 0    #Réinitialisation du nombre de lignes vides
data.append(tmp)
nbScripts += 1
nbTokensMoy = mean(tokScripts)

stats = pd.DataFrame({
    "Statistique" : [
        "Nombre de scripts",
        "Nombre moyen de tokens par script",
        "Nombre de blocs"
        ],
    "Valeur" : [nbScripts, nbTokensMoy, nbBloc]
    })

stats.to_csv("FixedDataset S3 with summary stats.csv")
pd.Series(data).to_csv("FixedDataset S3 with summary.csv", index = False)