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

#print(getHtml("https://gamefaqs.gamespot.com/sitemap"))

data = []

soup = BeautifulSoup(getHtml(url), "html.parser")

for faq in soup.findAll("ul")[2].children:
    if isinstance(faq, bs4.element.Tag):
        url1 = url0+faq.find("a")['href']
        soup1 = BeautifulSoup(getHtml(url1), "html.parser")
        time.sleep(random.uniform(1, 3))
        
        for letter in soup1.find("ul").children:
            if isinstance(letter, bs4.element.Tag):
                url1 = url0+letter.find("a")['href']
                soup2 = BeautifulSoup(getHtml(url1), "html.parser")
                time.sleep(random.uniform(1, 3))
                
                for game in soup2.find("ul").children:
                    if isinstance(game, bs4.element.Tag):
                        print(game.string)