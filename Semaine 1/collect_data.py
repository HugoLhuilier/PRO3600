"""NE PAS UTILISER : ban-ip

import bs4
from bs4 import BeautifulSoup
import requests
import time
import random

url = "https://gamefaqs.gamespot.com/sitemap"
url0 = "https://gamefaqs.gamespot.com"
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
    }

def getHtml(url):
    return requests.get(url, headers=headers).text

def containsScript(tag):
    return "Script" in str(tag.string)

#print(getHtml("https://gamefaqs.gamespot.com/sitemap"))

data = []

soup = BeautifulSoup(getHtml(url), "html.parser")

for faq in soup.findAll("ul")[3].children:
    if isinstance(faq, bs4.element.Tag):
        url1 = url0+faq.find("a")['href']
        print(url1)
        time.sleep(random.uniform(1, 3))
        soup1 = BeautifulSoup(getHtml(url1), "html.parser")  
        
        for letter in soup1.find("ul").children:
            if isinstance(letter, bs4.element.Tag) and letter.has_attribute("href"):
                url1 = url0+letter.find("a")['href']
                print(url1)
                time.sleep(random.uniform(1, 3))
                soup2 = BeautifulSoup(getHtml(url1), "html.parser")
                
                for game in soup2.find("ul").children:
                    if isinstance(game, bs4.element.Tag):
                        url1 = game.find("a")["href"]
                        print(url1)
                        time.sleep(random.uniform(1, 3))
                        soup3 = BeautifulSoup(getHtml(url1), "html.parser")
                        
                        for script in soup3.body.find_all(containsScript):
                            if script.has_attr("href"):
                                url1 = url0+script["href"]
                                print(url1)
                                time.sleep(random.uniform(1, 3))
                                soup4 = BeautifulSoup(getHtml(url1), "html.parser")
                                dascript = soup4.find(id="faqtext")
                                if dascript != None:                                
                                    data.append("")                           
                                    for tag in dascript.children:
                                        data[-1] += str(tag.string)

print("Completed : "+data.length+" scripts")
"""