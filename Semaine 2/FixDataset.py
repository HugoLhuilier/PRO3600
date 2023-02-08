#On supprime simplement les noms de personnages qui apparaissent en double

with open("../Semaine 1/BlogspotScripts.txt", "r", encoding='utf-8') as fic:
    with open("FixedData.txt", "w", encoding='utf-8') as r:
        for l in fic:
            pos = l.find(":")
            name = l[:pos+1]
            print("pos :"+str(pos))
            print("name : "+name)
            if pos != -1:
                while name in l:
                    l = l[pos+1:]
            r.write(name+l)
            r.write('\n')