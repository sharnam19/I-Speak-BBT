from bs4 import BeautifulSoup
import requests
import os
import unidecode
import sys  
import json
from nltk.tokenize import sent_tokenize
reload(sys)  
sys.setdefaultencoding('utf8')

BASE_URL = "https://bigbangtrans.wordpress.com/"
r = requests.get(BASE_URL, stream=True)
soup = BeautifulSoup(r.content,"html5lib")
lists = soup.find_all('li')
series_urls=[]
for li in lists:
    series_urls.append(li.a['href'])

series_urls = series_urls[1:]
sheldons = []
leonards = []
howards = []
raj = []
penny = []
for url in range(len(series_urls)):
    print("Fetching: "+str(url+1))
    r = requests.get(series_urls[url])
    soup = BeautifulSoup(r.content,"html5lib")
    spans = soup.find_all("span", { "style" : "font-size:small;font-family:Calibri;" })
    for span in spans:
        if span.text.startswith("Sheldon"):
            for i in  sent_tokenize(span.text.split(":")[1].strip()):
                sheldons.append(unidecode.unidecode(i))
        elif span.text.startswith("Leonard"):
            for i in sent_tokenize(span.text.split(":")[1].strip()):
                leonards.append(unidecode.unidecode(i))
        elif span.text.startswith("Howard"):
            for i in sent_tokenize(span.text.split(":")[1].strip()):
                howards.append(unidecode.unidecode(i))
        elif span.text.startswith("Raj"):
            for i in sent_tokenize(span.text.split(":")[1].strip()):
                raj.append(unidecode.unidecode(i))
        elif span.text.startswith("Penny"):
            for i in sent_tokenize(span.text.split(":")[1].strip()):
                penny.append(unidecode.unidecode(i))
data = {}
data['sheldon']=sheldons
data['leonard']=leonards
data['howard']=howards
data['raj']=raj
data['penny']=penny
json.dump(data,open("data.json","wb"))