import json
import unidecode
import numpy as np
data = json.load(open("data.json","rb"))
lists = data['sheldon']
# lists.extend(data['leonard'])
# lists.extend(data['howard'])
# lists.extend(data['penny'])
# lists.extend(data['raj'])
from nltk.tokenize import word_tokenize
mini = 1000
maxi = 0
avgs = 0
meds = []
len_dict= {}
LENGTH = 15
dialogues_15 = []
from numpy import median
TAGS = ['<SHELDON>','<LEONARD>','<HOWARD>','<RAJ>','<PENNY>']
KEYS = ['sheldon','leonard','howard','raj','penny']
for pos in range(5):
    lists = data[KEYS[pos]]
    for item in lists:
        sent = word_tokenize(unidecode.unidecode(item.lower()).replace("can't","cannot"))
        sent = ["not" if x=="n't" else x for x in sent]
        sent = ["will" if x=="'ll" else x for x in sent]
        sent = ["am" if x=="'m" else x for x in sent]
        sent = ["is" if x=="'s" else x for x in sent]
        sent = ["have" if x=="'ve" else x for x in sent]
        sent = ["would" if x=="'d" else x for x in sent]
        sent = filter(lambda a: a != "," and a!= '.' and a!='...' and a!='(' and a!=')', sent)
        if len(sent)>maxi:
            maxi = len(sent)
        if len(sent)<mini:
            mini = len(sent)
        meds.append(len(sent))
        avgs+=len(sent)
        if len(sent) not in len_dict:
            len_dict[len(sent)]=0
        len_dict[len(sent)]+=1
        sent = sent[:15]
        sent.append('<END>')
        sent.insert(0,TAGS[pos])
        dialogues_15.append(sent)

word_to_index = {}
index_encoded_string = np.zeros((len(dialogues_15),17))
index = 1
i = 0
for dialogue in dialogues_15:
    t = 0
    for word in dialogue:
        if word not in word_to_index:
            word_to_index[word]=index
            index+=1
        index_encoded_string[i,t]=word_to_index[word]
        t+=1
    i+=1
print(len(word_to_index.keys()))
print(index_encoded_string)
data = {}
data['w2ix']=word_to_index
data['ix_string']=index_encoded_string.tolist()

json.dump(data,open("processed.json","wb"))