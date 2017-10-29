import json
data = json.load(open("data.json","rb"))
sheldon = data['sheldon']
leonard = data['leonard']
raj = data['raj']
howard = data['howard']
penny = data['penny']
print(len(sheldon)+len(leonard)+len(raj)+len(howard)+len(penny))
#for x in data:
    #print(x)