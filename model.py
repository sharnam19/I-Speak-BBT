import numpy as np
from sequential import *
from layer import *
from loss import *
import json
import sys
data = json.load(open("processed.json","rb"))
w2ix=data['w2ix']
w2ix['<ZERO>']=0
ix2w = { v:k for k,v in w2ix.items() }
X  = np.array(data['ix_string'],dtype=np.int32)
mask = (X!=0)
mask = mask.astype(np.int32)

trainX  = X[:,:-1]
trainY = X[:,1:]

learning_rate = 1e-3
epoch = 1000
embed_dimension = 100
hidden_dimension = 128
batch_size = 5000
vocab_size = len(w2ix.keys())+1
bptt = 4
forward = rnn_forward
backward = rnn_backward
T = trainX.shape[1]
sentence_length = 12

Wembed = np.random.normal(loc=0.0,scale=1,size=(vocab_size,embed_dimension))
Wx = np.random.normal(loc=0.0,scale=1,size=(embed_dimension,hidden_dimension))
Wy = np.random.normal(loc=0.0,scale=1,size=(hidden_dimension,vocab_size))
Wh = np.random.normal(loc=0.0,scale=1,size=(hidden_dimension,hidden_dimension))

bh = np.zeros((hidden_dimension,))
by = np.zeros((vocab_size,))

def generate_text(character_num):
    global w2ix,Wembed,Wx,Wy,Wh,bh,by,forward,backward,hidden_dimension
    hprev = np.random.normal(loc=0.0,scale=0.01,size=(1,hidden_dimension))
    #hprev = np.zeros((1,hidden_dimension))
    inputs = np.zeros((1,sentence_length+1),dtype=np.int32)
    inputs[0,0] = w2ix[character_num]
    sentence = [ix2w[inputs[0,0]]]
    for start_index in range(sentence_length):
        Wout,_ = word_embedding_forward(inputs,Wembed)
        hprev,_ = rnn_step(Wout[0,start_index],hprev,Wx,Wh,bh)
        Aout = hprev.dot(Wy)+by
        inputs[0,start_index+1] = np.argmax(Aout)
        sentence.append(ix2w[inputs[0,start_index+1]])
        
    print(" ".join(sentence))
    sys.stdout.flush()

generate_text('<SHELDON>')
generate_text('<LEONARD>')
generate_text('<HOWARD>')
generate_text('<RAJ>')
generate_text('<PENNY>')
trainMask = mask

for e in range(epoch):
    for batch_start in range(0,trainX.shape[0],batch_size):
        batchX = trainX[batch_start:min(batch_start+batch_size,trainX.shape[0])]

        batchY = trainY[batch_start:min(batch_start+batch_size,trainX.shape[0])]
        batchMask = trainMask[batch_start:min(batch_start+batch_size,trainX.shape[0])]

        hprev = np.zeros((batchX.shape[0],hidden_dimension))
        for start_index in range(0,T,bptt):
            sent = batchX[:,start_index:min(start_index+bptt,T)]
            sentY = batchY[:,start_index:min(start_index+bptt,T)]
            sentMask = batchMask[:,start_index:min(start_index+bptt,T)]
            Wout,Wcache = word_embedding_forward(sent,Wembed)
            h,tcache = forward(Wout,hprev,Wx,Wh,bh)
            hprev = h[:,-1,:]
            Aout,Acache = temporal_affine_forward(h,Wy,by)
            loss,dx = temporal_softmax_loss(Aout,sentY,sentMask)

            dx,dWy,dby = temporal_affine_backward(dx,Acache)
            dx,dhprev,dWx,dWh,dbh = backward(dx,tcache)

            dWembed = word_embedding_backward(dx,Wcache)

            Wembed -= learning_rate*dWembed
            Wx -= learning_rate*dWx
            Wh -= learning_rate*dWh
            bh -= learning_rate*dbh
            Wy -= learning_rate*dWy
            by -= learning_rate*dby
    
    generate_text('<SHELDON>')
    generate_text('<LEONARD>')
    generate_text('<HOWARD>')
    generate_text('<RAJ>')
    generate_text('<PENNY>')
    print("Done Update")
    sys.stdout.flush()
    
data = {}
data['Wx']=Wx.tolist()
data['Wh']=Wh.tolist()
data['Wy']=Wy.tolist()
data['bh']=bh.tolist()
data['by']=by.tolist()
data['Wembed']=Wembed.tolist()

json.dump(data,open("model.json","wb"))
