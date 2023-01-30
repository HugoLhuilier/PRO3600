import bs4
from bs4 import BeautifulSoup
import requests
import time
import random
import re


url = "https://gamefaqs.gamespot.com/sitemap"
url0 = "https://gamefaqs.gamespot.com"
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
    }

def getHtml(url):
    return requests.get(url, headers=headers).text

def containsScript(tag):
    return "Script" in str(tag.string)

data = []

soup = BeautifulSoup(getHtml(url), "html.parser")

faq = soup.findAll("ul")[3].contents[1]


url1 = url0+faq.find("a")['href']

print(url1)

time.sleep(random.uniform(1, 3))
soup1 = BeautifulSoup(getHtml(url1), "html.parser")  

letter = soup1.find("ul").contents[1]

url1 = url0+letter.find("a")['href']

print(url1)

time.sleep(random.uniform(1, 3))
soup2 = BeautifulSoup(getHtml(url1), "html.parser")

game = soup2.find("ul").contents[1]

url1 = game.find("a")["href"]

print(url1)

time.sleep(random.uniform(1, 3))
soup3 = BeautifulSoup(getHtml("https://gamefaqs.gamespot.com/pc/615805-the-elder-scrolls-v-skyrim/faqs"), "html.parser")

script = soup3.body.find_all(containsScript)[0]


url1 = url0+script["href"]

print(url1)

time.sleep(random.uniform(1, 3))
soup4 = BeautifulSoup(getHtml(url1), "html.parser")

data.append("")

for tag in soup4.find(id="faqtext").children:
    data[-1] += str(tag.string)