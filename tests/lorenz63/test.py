# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:10:34 2019
@author: pmaddala
"""

import torch
import torch.nn as nn
import string

data = "i love neural networks"
EOF = "#"
#data = data+EOF
data = data.lower()

seq_len = len(data)

letters = string.ascii_lowercase+' #'
print('Letter set is '+letters)
n_letters = len(letters)
print(letters)

#letter to tensor
def ltt(ch):
    ans = torch.zeros(n_letters)
    ans[letters.find(ch)]=1
    return ans
    
def getLine(s):
    ans = []
    for c in s:
        ans.append(ltt(c))
    return torch.cat(ans,dim=0).view(len(s),1,n_letters)
    
class MyLSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(MyLSTM,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        #LSTM takes, input_dim, hidden_dim and num_layers incase of stacked LSTMs
        self.LSTM = nn.LSTM(input_dim,hidden_dim)
        self.LNN = nn.Linear(hidden_dim,input_dim)
        
    #Input must be 3 dimensional (seq_len, batch, input_dim). 
    #hc is a tuple of hidden and cell state vector. Each of them have shape (1,1,hidden_dim)
    def forward(self,inp,hc):
        #this gives outut for each input in the sequence and also (hidden and cell state vector)
        #Dimensions of output vector is (seq_len,batch,hidden_dim)
        output,_= self.LSTM(inp,hc)
        return self.LNN(output)
        #return output
        

#Dimensions of output of neural network is (seq_len, batch , hidden_dim). Since we want output dimensions to be
#the same as n_letters, hidden_dim = n_letters (**output dimensions = hidden_dimensions)
hidden_dim = n_letters     
#Invoking model. Input dimensions = n_letters i.e 28. output dimensions = hidden_dimensions = 28
model = MyLSTM(n_letters,hidden_dim)
#I'm using Adam optimizer here
optimizer = torch.optim.Adam(params = model.parameters(),lr=0.01)
#Loss function is CrossEntropyLoss
LOSS = torch.nn.CrossEntropyLoss()

#List to store targets
targets = []
#Iterate through all chars in the sequence, starting from second letter. Since output for 1st letter is the 2nd letter
for x in data[1:]+'#':
    #Find the target index. For a, it is 0, For 'b' it is 1 etc..
    targets.append(letters.find(x))
#Convert into tensor
targets = torch.tensor(targets)
    
n_iters = 400

#List to store input
inpl = []
#Iterate through all inputs in the sequence
for c in data:
    #Convert into tensor
    inpl.append(ltt(c))
#Convert list to tensor
inp = torch.cat(inpl,dim=0)
#Reshape tensor into 3 dimensions (sequence length, batches = 1, dimensions = n_letters (28))
inp = inp.view(seq_len,1,n_letters)


#Let's note down start time to track the training time
import time
start = time.time()
#Number of iterations
n_iters = 150
for itr in range(n_iters):
    #Zero the previosus gradients
    model.zero_grad()
    optimizer.zero_grad()
    #Initialize h and c vectors
    h = torch.rand(hidden_dim).view(1,1,hidden_dim)
    c = torch.rand(hidden_dim).view(1,1,hidden_dim)
    #Find the output
    output = model(inp,(h,c))
    #Reshape the output to 2 dimensions. This is done, so that we can compare with target and get loss
    output = output.view(seq_len,n_letters)
    #Find loss
    loss = LOSS(output,targets)
    #Print loss for every 10th iteration
    if itr%10==0:
        print('Iteration : '+str(itr)+' Loss : '+str(loss) )
    #Back propagate the loss
    loss.backward()
    #Perform weight updation
    optimizer.step()
    
print('Time taken to train : '+str(time.time()-start)+" seconds")
 

#This utility method predicts the next letter given the sequence   
def predict(s):
    #Get the vector for input
    inp = getLine(s)
    #Initialize h and c vectors
    h = torch.rand(1,1,hidden_dim)
    c = torch.rand(1,1,hidden_dim)
    #Get the output
    out = model(inp,(h,c))
    #Find the corresponding letter from the output
    return letters[out[-1][0].topk(1)[1].detach().numpy().item()]
         

#THis method recursively generates the sequence using the trained model
def gen(s):
    #If generated sequence length is too large, or terminate char is generated, we can print the generated output so far
    if s[-1]=='#' or len(s)>=len(data)+5:
        print(s)
        return
    #Predict with sequence s
    pred = predict(s)
    #Continue prediction with sequence s + predicted value
    print(s+pred)
    #Recurse
    gen(s+pred)
