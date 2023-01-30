from bs4 import BeautifulSoup
import requests
import time
import random


headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
    }

def getHtml(url):
    return requests.get(url, headers=headers).text

def getSoup(url):
    time.sleep(random.uniform(1, 3))
    return BeautifulSoup(getHtml(url), "html.parser")

data = []

with open("sitemap.xml", "r") as f:
    xml = f.read()

soup = BeautifulSoup(xml, "xml")

for link in soup.findAll("loc"):
    url = str(link.string)
    print(url)
    soup1 = getSoup(url)
    data.append("")
    for txt in soup1.find(id="post-toc").descendants:
        if txt.string != None:
            data[-1] += str(txt.string)
    
print("Completed : "+data.length+" scripts")