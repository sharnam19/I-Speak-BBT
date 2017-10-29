import numpy as np
def rnn_step(x, h_prev, Wx, Wh, b):
    """
        Input:
            x : Input Sequence of shape (N,D)
            h_prev : Previous Hidden State (N,H)
            Wx : Input to Hidden Weight of shape (D,H)
            Wh : Hidden to Hidden Weight of shape (H,H)
            b :  Bias of shape (H,)
        Output:
            h_next : Hidden State at Next Time Step of shape (N,H)
            cache : Cached Values for Backprop 
    """
    
    h_next = np.tanh(x.dot(Wx)+h_prev.dot(Wh) + b[np.newaxis,:])
    cache = (x,h_prev,Wx,Wh,h_next)
    return h_next,cache

def rnn_step_backward(dOut,cache):
    """
        Input
            dOut: Upstream Gradients wrt h (N,H)
            cache : Cached Values useful for backprop
        Output:
            dx: Gradients wrt x
            dh_prev : Gradients wrt h_prev
            dWx : Gradients wrt Wx
            dWh : Gradients wrt Wh
            db : Gradients wrt b
    """
    x,h_prev,Wx,Wh,h_next = cache
    dSq = (1-np.square(h_next))*dOut
    
    dx = dSq.dot(Wx.T)
    dWx = x.T.dot(dSq)
    dh_prev = dSq.dot(Wh.T)
    dWh = h_prev.T.dot(dSq)
    db = np.sum(dSq,axis=0)
    
    return dx,dh_prev,dWx,dWh,db

def rnn_forward(x, h0, Wx, Wh, b):
    h, cache = None, None
    N,T,D = x.shape
    H = Wh.shape[0]
    h = np.empty((N,T,H))
    cache = []
    for i in range(x.shape[1]):
        if i == 0:
            h[:,i,:],c = rnn_step(x[:,i,:],h0,Wx,Wh,b)
        else:
            h[:,i,:],c = rnn_step(x[:,i,:],h[:,i-1,:],Wx,Wh,b)
        cache.append(c)
    cache = (cache,D)
    return h, cache


def rnn_backward(dh, cache):
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    N,T,H = dh.shape
    
    N, T, H = dh.shape
    step_cache, D = cache
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    dx = np.zeros((N, T, D))
    db = np.zeros((H))
    dprev_h = np.zeros((N, H))
    for t in reversed(range(T)):
        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dh[:, t, :] + dprev_h, step_cache[t])
        dx[:, t, :] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
    dh0 = dprev_h
    return dx, dh0, dWx, dWh, db

def word_embedding_forward(x, W):
    out, cache = None, None
    V,D = W.shape
    out = W[x]
    cache = (x,V)
    return out, cache


def word_embedding_backward(dout, cache):
    dW = None
    x,V = cache
    
    N,T = x.shape
    D = dout.shape[2]
    dW = np.zeros((V, D))

    np.add.at(dW, x.reshape(N*T), dout.reshape(N*T, D))
    return dW