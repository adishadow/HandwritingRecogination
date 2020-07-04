import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import struct
import time
import skimage.transform as trans
import skimage.color as colr
import scipy.optimize as op
#import cv2

def get_data():
    data_label = open('E:\\ipnyb\\emnist-letters-train-labels-idx1-ubyte','rb')
    data_image = open('E:\\ipnyb\\emnist-letters-train-images-idx3-ubyte','rb')

    #decoding the label from the label file     
    magic, num = struct.unpack(">II", data_label.read(8))
    lbl = np.fromfile(data_label, dtype=np.int8)

    # decoding the images form the coded i  mage file
    magic, num, rows, cols = struct.unpack(">IIII", data_image.read(16))
    img = np.fromfile(data_image, dtype=np.uint8).reshape(len(lbl), rows, cols)
    
    return lbl,img

    
def show(lbl,img):
    # to show the image produce 
    
    for i in range(6):
        im = np.random.randint(1,6000)
        plt.imshow(np.transpose(img[im]))
        plt.title('the image produce is for label '+str(lbl[im]))
        #plt.hold()
        plt.pause(2)
        
        

#Converting the image to grayscale and the 28 x 28 p image to 20 x 20
def parse_input(X):
   # X = colr.rgb2gray(X[1])
   # print(np.shape(X[1]))
   X_temp = np.zeros((np.shape(X)[0],20,20))
   for i in range(np.shape(X)[0]):
       X_temp[i] = trans.resize(X[i][:][:],[20,20])
   return X_temp
        
# random set up the weight for the network between the rang -1 to +1
def random_intialize(m,n):
    wt = np.random.randn(m,n)
    return wt
    
def intializeInput(img1):
    index_1 = 1
    index_2 = 28
    X_val = np.zeros((np.shape(img1)[0],401))
  
    for i in range(np.shape(img1)[0]):
        index_1 = 1
        index_2 = 20
        for j in range(np.shape(img1)[1]):
                    X_val[i][index_1:(index_2+1)] = np.transpose(img1[i])[j][:]
                    index_1 =index_2+1
                    index_2 = 20 * (j+2)
    X_val[:,0] = 1 
    return X_val
    

# the Sigmoid activation system for neuron 
def sigmoid(x):
    e = np.exp(-1*x)
    temp = np.ones(np.shape(e))
    e = temp + e
    e = np.divide(1,e)
    return e

def parse_label(lbl):
    parsed_label = np.zeros((np.shape(lbl)[0],26))
    for i in range(np.shape(lbl)[0]):
        parsed_label[i][lbl[i]-1] = 1
    return parsed_label   
def  nnCost(intialWeight,X_val,p_label,lambd):
    wt_1 = intailWeight[0]
    wt_2 = intailWeight[1]
    wt_3 = intailWeight[2]
#feeding forward to network
#between layer 1-)2
    z1 = X_val @ wt_1
    a = sigmoid(z1)
#between layer 2 and 3
    a1 =np.ones((np.shape(X_val)[0],41))
    a1[:,1:] = a[:,:]
    z2 = a1 @ wt_2
    a = sigmoid(z2)
# between layer 3-4
    a2 =np.ones((np.shape(X_val)[0],41))
    a2[:,1:] = a[:,:]
    z3 = a2 @ wt_3
    a3 = sigmoid(z3) 
#output of neuron
    h=a3

#   Calculating the cost
    temp = 0
    for i in range(np.shape(X_val)[0]):
       for k in range(np.shape(p_label)[1]):
          temp = temp + ((-1* p_label[i,k]) * (np.log(h[i,k])) - (1-p_label[i,k]) * (np.log(1-h[i,k])) )

    J =temp/m
    reg = (lambd/m) * np.sum(np.sum((wt_1[:,1:] @ np.transpose(wt_1[:,1:])),axis =0)) +np.sum(np.sum((wt_2[:,1:] @ np.transpose(wt_2[:,1:])),axis =0))+np.sum(np.sum((wt_3[:,1:] @ np.transpose(wt_3[:,1:])),axis =0))
#backprop

    sdelta_4 = h - p_label[:40000]
    sdelta_3 = (sdelta_4 @ np.transpose(wt_3)) * a2 *( np.ones(np.shape(a2)) -a2)
    sdelta_2 = (sdelta_3[:,1:] @ np.transpose(wt_2)) * a1 *( np.ones(np.shape(a1)) -a1)

    ldelta_3 = np.transpose(a2) @ sdelta_4 
    ldelta_2 = np.transpose(a1) @ sdelta_3[:,1:]
    ldelta_1 = np.transpose(X_val) @ sdelta_2[:,1:]

#^ UPDATING NET-WEIGHT

    wt_1 = wt_1 - np.divide(ldelta_1,m) 
    wt_2 = wt_2 - np.divide(ldelta_2,m)
    wt_3 = wt_3 - np.divide(ldelta_3,m)
    # regularized
    wt_1[:,1:] = wt_1[:,1:] - ((lambd/m) *  wt_1[:,1:])
    wt_2[:,1:] = wt_2[:,1:] - ((lambd/m) *  wt_2[:,1:])
    wt_3[:,1:] = wt_3[:,1:] - ((lambd/m) *  wt_3[:,1:])



    
[lbl,img] = get_data()
show(lbl,img)
X = parse_input(img)
show(lbl,X)
np.random.seed(142)
p_label = parse_label(lbl)
wt_1 =np.random.randn(401,40)
wt_2 = np.random.randn(41,40)
wt_3 = np.random.randn(41,26)
X_val = intializeInput(X[:40000])


m,n = X_val.shape
no_of_iteration = 2000
lambd = 14

intailWeight = np.array([wt_1,wt_2,wt_3])
otCost = op.minimize(fun = nnCost,x0 = intailWeight,args = (X_val,p_label),method='L-BFGS-B')
for q in range(no_of_iteration):
    

 
    print('The cost for itreation ',q,' is = ',J)
###






