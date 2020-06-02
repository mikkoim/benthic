# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:37:40 2020

@author: Mikko Impiö
"""

import tensorflow as tf
import tensorflow.keras.backend as K

import numpy as np

cce = tf.keras.losses.CategoricalCrossentropy()

y = np.array([[1., 0., 0.], 
              [1., 0., 0.], 
              [1., 0., 0.],
              [1., 0., 0.],
              [0., 1., 0.], 
              [0., 1., 0.],
              [0., 0., 1.]])

yhat = np.array([[.90, .05, .05],#0
                 [.50, .10, .40],#0
                 [.80, .15, .05],#0
                 [.90, .07, .03],#0
                 [.05, .89, .06],#1 
                 [.15, .70, .15],#1
                 [.05, .01, .94]])

loss = cce(y, yhat)

print('Loss: ', loss.numpy())

#%%

H = -np.sum(y*np.log(yhat+1e-7), axis=1, keepdims=1)
H = -y*np.log(yhat+1e-24)-(1-y)*np.log((1-yhat+1e-24))

loss1 = np.mean(H)
print('Loss: ', loss1)

#%% manual tf focal loss

y = np.array([[1.], [1.], [0.]], dtype=np.float32)
yhat = np.array([[0.97], [0.91], [0.03]], dtype=np.float32)

y = tf.convert_to_tensor(y, dtype=np.float32)
yhat = tf.convert_to_tensor(yhat, dtype=np.float32)

gamma = 2.0
alpha = 0.25

ce = -y*np.log(yhat+1e-24)-(1-y)*np.log((1-yhat+1e-24))
ce = K.binary_crossentropy(y,yhat)

p_t = (y * yhat) + ((1 - y) * (1 - yhat))
alpha_factor = (y * alpha + (1 - y) * (1 - alpha))

modulating_factor = ((1.0 - p_t)**gamma)

np.sum(alpha_factor*modulating_factor*ce, axis=-1, keepdims=1)

#%% focal multiclass

y = np.array([[1., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
yhat = np.array([[0.97, 0.03], [0.91, 0.09], [0.03, 0.97]], dtype=np.float32)


def focal_loss(gamma=2.0, alpha=[0.25, 0.75]):
    
    def loss(y,yhat):
        y = tf.convert_to_tensor(y, dtype=np.float32)
        yhat = tf.convert_to_tensor(yhat, dtype=np.float32)
        
        ce = -np.sum(y*np.log(yhat+1e-7), axis=1, keepdims=1)
        
        p_t = (y * yhat)
        alpha_factor = (y * alpha)
        
        modulating_factor = ((1.0 - p_t)**gamma)
        
        fl = tf.reduce_sum(alpha_factor*modulating_factor*ce, axis=-1, keepdims=1)
    
        return tf.reduce_mean(fl)
    return loss

import tensorflow.keras.backend as K
#%%
############## !!!!!!!COLAB VERSION!!!!! ################
def focal_loss(gamma=2.0, alpha=0.25):

    def loss(y,yhat):
        
      y = K.cast(y, tf.float32)
      yhat = tf.convert_to_tensor(yhat, dtype=y.dtype)

      ce = -K.sum(y*K.log(yhat+1e-7), axis=1, keepdims=True)
      
      p_t = (y * yhat)
      alpha_factor = (y * alpha)
      
      modulating_factor = ((1.0 - p_t)**gamma)
      
      fl = K.sum(alpha_factor*modulating_factor*ce, axis=-1)
  
      return fl

    return loss

def weighted_crossentropy(alpha=1.0):

    def loss(y,yhat):
        
      y = K.cast(y, tf.float32)
      yhat = tf.convert_to_tensor(yhat, dtype=y.dtype)

      w_ce = -K.sum(y*K.log(yhat+1e-7)*alpha, axis=1)
      
      return w_ce

    return loss

def class_balanced_loss(y_n):

    N = np.max(y_n)
    beta = (N-1)/N 

    beta = K.cast(beta, tf.float32)

    y_n = K.cast(y_n, tf.float32)
    E = (1- K.pow(beta, y_n))/(1-beta)

    alpha = (1/E)

    def loss(y,yhat):
        
      y = K.cast(y, tf.float32)
      yhat = tf.convert_to_tensor(yhat, dtype=y.dtype)
      
      CB = -K.sum( y*K.log(yhat + K.epsilon())*alpha, axis=1 )
      
      return CB

    return loss
######################################

def cross_penalizer_loss(gamma=2.0, alpha=0.25):

    def loss(y,yhat):
        
      y = K.cast(y, tf.float32)
      yhat = tf.convert_to_tensor(yhat, dtype=y.dtype)
      alpha_ = tf.convert_to_tensor(alpha, dtype=tf.float32)

      ce = -(y*K.log(yhat+1e-7)+(1-y)*K.log(1-(yhat+1e-7)))
      
      p_t = (y * yhat) + ((1 - y) * (1 - yhat))
      alpha_factor = (y * alpha_) + ((1 - y) * (1 - alpha_))
      
      modulating_factor = ((1.0 - p_t)**gamma)
      
      fl = K.sum(alpha_factor*modulating_factor*ce, axis=1)
  
      return fl

    return loss


#%% class approach
def focal_loss(y, yhat, gamma=2.0, alpha=0.25):

    y = tf.convert_to_tensor(y)
    yhat = tf.convert_to_tensor(yhat, dtype=y.dtype)

    ce = -tf.reduce_sum(y*tf.math.log(yhat+1e-7), axis=1, keepdims=True)

    p_t = (y * yhat)
    alpha_factor = (y * alpha)

    modulating_factor = ((1.0 - p_t)**gamma)

    fl = tf.reduce_sum(alpha_factor*modulating_factor*ce, axis=-1)

    return fl

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self,
                 from_logits=False,
                 alpha=0.25,
                 gamma=2.0,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name='focal_loss'):
        super(FocalLoss, self).__init__(
            name=name, reduction=reduction)

        self.from_logits = from_logits
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        return focal_loss(
            y_true,
            y_pred,
            alpha=self.alpha,
            gamma=self.gamma)
#%% weighted CE

s01 = [.90, .05, .05]
s21 = [.05, .05, .90]

s02 = [.70, .15, .15]
s22 = [.15, .15, .70]

yhat1 = np.array([s01,#0
                 [.50, .10, .40],#0
                 [.80, .15, .05],#0
                 [.90, .07, .03],#0
                 [.05, .89, .06],#1 
                 [.15, .70, .15],#1
                 s21])

yhat2 = np.array([s02,#0
                 [.50, .10, .40],#0
                 [.80, .15, .05],#0
                 [.90, .07, .03],#0
                 [.05, .89, .06],#1 
                 [.15, .70, .15],#1
                 s22])

def c(y,yhat):
    a = 1-np.mean(y,axis=0)
    H = -np.sum(a*y*np.log(yhat+1e-24), axis=1, keepdims=1)
    
    loss2 = np.mean(H)
    print('Loss bal: ', loss2)
    
    loss = cce(y, yhat)
    print('Loss: ', loss.numpy())
    
c(y,yhat1)
print()
c(y,yhat2)

#%% Focal Loss stock

def sigmoid_focal_crossentropy(y_true,
                               y_pred,
                               alpha=0.25,
                               gamma=2.0,
                               from_logits=False):
    """
    Args
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.
    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.
    """
    if gamma and gamma < 0:
        raise ValueError(
            "Value of gamma should be greater than or equal to zero")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch for y_true: {} and y_pred: {}".format(
            tf.shape(y_true), tf.shape(y_pred)))

    # Get the cross_entropy for each entry
    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
        alpha_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)


#%% Comparison CCE, FL stock vs own fl

y = [[1., 0., 0., 0.], 
     [0., 1., 0., 0.], 
     [0., 0., 1., 0.]]

yhat = [[.9, .05, .025, .025], 
        [.05, .03, .89, .03], 
        [.025, .31, .64, .025]]

yhat = [[0., 1., 0., 0.], 
         [1., 0., 0., 0.], 
         [0., 0., 0., 1.]]

yhat = [[.999, .001,   0.,    0.], 
        [  0., .999, .001,    0.], 
        [  0.,   0., .999, .001,]]
# y = [[1., 0., 0.], 
#      [0., 1., 0.], 
#      [0., 0., 1.]]

# yhat = [[.9, .05, .05], 
#         [.05, .89, .06], 
#         [.05, .01, .94]]

ybin = [[1.], [1.], [0.]]

yhatbin =  [[.9], [.89], [.06]]

# CCE
cce = tf.keras.losses.CategoricalCrossentropy()
loss = cce(y, yhat)
print('CE Loss: ', loss.numpy()) 

# FL stock
loss2 = sigmoid_focal_crossentropy(y, yhat)
print('Loss FL stock: ', loss2.numpy()) 
print('Loss FL stock mean:', loss2.numpy().mean())

# FL own
fl = focal_loss()
loss3 = fl(y, yhat)
print('Loss FL own: ', loss3.numpy()) 
print('Loss FL own mean:', loss3.numpy().mean())

# FL test
cpl = cross_penalizer_loss()
loss4 = cpl(y, yhat)
print('Loss FL test: ', loss4.numpy()) 
print('Loss FL test mean:', loss4.numpy().mean())

##################### With alpha=0.25 FL and stock very different

# CCE
cce = tf.keras.losses.CategoricalCrossentropy()
loss = cce(y, yhat)
print('CE Loss: ', loss.numpy()) 

# FL stock
loss2 = sigmoid_focal_crossentropy(y, yhat, alpha=1.0)
print('Loss FL stock: ', loss2.numpy()) 

# FL own
fl = focal_loss(alpha=1.0)
loss3 = fl(y, yhat)
print('Loss FL own: ', loss3.numpy()) 
print('Loss FL own mean:', loss3.numpy().mean())

# FL test
cpl = cross_penalizer_loss(alpha=1.0)
loss4 = cpl(y, yhat)
print('Loss FL test: ', loss4.numpy()) 
print('Loss FL test mean:', loss4.numpy().mean())

#####################with alpha=1.0 we get same res

# CCE
cce = tf.keras.losses.CategoricalCrossentropy()
loss = cce(y, yhat)
print('CE Loss: ', loss.numpy()) 

# FL stock
loss2 = sigmoid_focal_crossentropy(y, yhat, gamma=0.0, alpha=1.0)
print('Loss FL stock: ', loss2.numpy()) 

# FL own
fl = focal_loss(gamma=0.0, alpha=1.0)
loss3 = fl(y, yhat)
print('Loss FL own: ', loss3.numpy()) 
print('Loss FL own mean:', loss3.numpy().mean())

# FL test
cpl = cross_penalizer_loss(gamma=0.0, alpha=1.0)
loss4 = cpl(y, yhat)
print('Loss FL test: ', loss4.numpy()) 
print('Loss FL test mean:', loss4.numpy().mean())

#####################gamma=0 gives same as CE




#%%
##### Binary test
cce = tf.keras.losses.CategoricalCrossentropy()
loss = cce(ybin, yhatbin)
print('Loss: ', loss.numpy()) 

# FL stock
loss2 = sigmoid_focal_crossentropy(ybin, yhatbin, gamma=0, alpha=1.0)
print('Loss FL stock: ', loss2.numpy()) 

# FL own
fl = focal_loss(gamma=0, alpha=1.0)
loss3 = fl(ybin, yhatbin)
print('Loss FL own: ', loss3.numpy()) 
print('Loss FL own mean:', loss3.numpy().mean())


#%% FL vs FLstock

# FL stock
loss2 = sigmoid_focal_crossentropy(y, yhat, gamma=4.0, alpha=1.0)
print('Loss FL stock: ', loss2.numpy()) 

# FL own
fl = focal_loss(gamma=2.0, alpha=1.0)
loss3 = fl(y, yhat)
print('Loss FL own: ', loss3.numpy())


#%% CB loss

y_n = np.array([5,10,2,10])

N = np.max(y_n)
beta = (N-1)/N 
print(beta)


# FL stock
loss2 = sigmoid_focal_crossentropy(y, yhat, gamma=2.0, alpha=1.0)
print('Loss FL stock: ', loss2.numpy()) 

# FL own
fl = focal_loss(gamma=2.0, alpha=1.0)
loss3 = fl(y, yhat)
print('Loss FL own: ', loss3.numpy())

CB = class_balanced_loss(y_n, beta=0.9)
loss4 = CB(y, yhat)
print('CB loss: ', loss4.numpy())



#%%
import matplotlib.pyplot as plt

a = np.arange(0,1,0.01)

b = (1-a)/3
c = 2*b


lo = -(np.log(a) + np.log(1-b) +  np.log(1-c))

plt.plot(a, -np.log(a), linestyle='dashed')
plt.plot(a, -np.log(1-b))
plt.plot(a, -np.log(1-c))
plt.plot(a, lo, linestyle='dashdot')



a = np.arange(0,1,0.01)

b = (1-a)

lo = -(np.log(a) + np.log(1-b))

plt.plot(a, -np.log(a))
plt.plot(a, -np.log(1-b))
plt.plot(a, lo)

