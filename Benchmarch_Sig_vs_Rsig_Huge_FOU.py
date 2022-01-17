#!/usr/bin/env python
# coding: utf-8

# Here we import some libraries that will come handy aftwerward.

# In[1]:


import signatory
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn import linear_model
now = datetime.now()
time = now.strftime("%Y%m%d_%H%M%S")

np.random.seed(1000)
plt.rcParams.update({'font.size': 15})
plt.rcParams["figure.figsize"] = (7,5)
import seaborn as sns
sns.color_palette("colorblind")
# Here we declare some parameters:
    
import sys
sys.path.insert(0, r'C:\Users\eneam\Desktop')
from stochastic.continuous import FractionalBrownianMotion
from ttictoc import tic, toc

# In[2]:

M = 79
d = M + 1
signatures_dim = 3
dim= np.int((d**(signatures_dim+1)-1)/(d-1)-1)    

# Number of timesteps in which the split the time span [0, T]
N_T = 100
# Number of rain Sample
N_S = 10000
N_C = 10000


hurst = 0.3
FBM = FractionalBrownianMotion(hurst=hurst, t=1)

Theta = np.ones((M,M))

for i in range(M):
    for j in range(M):
        Theta[i,j] = np.round((j+1)/(i+1),2)
        Theta[j,i] = np.round((j+1)/(i+1),2)

# Theta = np.array([[1, 0.5],[0.5, 1]])
G = np.diag([1 for x in range(M)])
mu = np.ones((M,))


k = d
Z0 = np.random.normal(0.0,1.0,size=(k,1)) 
# Decided where to put the outputs. You have to change this...

# In[3]:


quality = 1000
# Target_Path_Folder = r"C:\Users\eneam\Dropbox\Research\Thesis\GBM_Signal_Extraction_GBM_GBM_Few_Shit_Student_" + str(mu) + "_" + str(sigma) + "_"  + str(mu_2) + "_" + str(sigma_2) + "_" + str(N_T) + "_" + str(M) + "_" + str(today).replace("-", "_")
Target_Path_Folder = r"C:\Users\eneam\Dropbox\Research\Rough_Paper\Outputs\ICLR\Benchmark_Signature_FOU_" + str(N_T) + "_" + str(N_S) + "_" + str(signatures_dim) + "_" + str(k) + "_" + str(d)
Path(Target_Path_Folder).mkdir(parents=True, exist_ok=True)
path_print = Path(Target_Path_Folder)


# Now we define some utilities

# In[22]:
    
def Tonio_Measure(serie_1, serie_2):
    numerator = np.sum(np.square(serie_1-serie_2),axis=0)
    denominator = np.sum(np.square(serie_1),axis=0)
    return np.mean(numerator/denominator)
    
    
def Tonio_Measure_all(df1,df2):
    df_tonio = np.empty((df1.shape[0],0))
    for i in range(df1.shape[0]):
        df_tonio = np.insert(df_tonio,0,Tonio_Measure(df1[i,:], df2[i,:]))
    return df_tonio


    

def randomAbeta(d,k):
    A = []
    beta = []
    for i in range(d):
        # B = 0.0*nilpotent(M) + np.random.standard_t(2,size=(M,M)) 
        B = np.random.normal(0.0,1.0,size=(k,k)) 
        # B = np.random.permutation(B)
        A = A + [B]
        # beta = beta + [0.0*canonical(i,M)+np.random.standard_t(2,size=(M,1))]
        beta = beta + [np.random.normal(0.0,1.0,size=(k,1))]
    return [A,beta]


CDeta = randomAbeta(d,k)
C = CDeta[0]
deta = CDeta[1]


def path():
    
    t = np.linspace(0,1,N_T+1)
    dt = t[1] - t[0]
    FBMpath  = FBM.sample(N_T, True)
    SDEpath = np.zeros((N_T+1,M))
    SDEpath[0,:] = 1
    
    for i in range(1,d-1):
        FBMpath_temp = FBM.sample(N_T, True)
        FBMpath = np.c_[FBMpath, FBMpath_temp]
    dB = np.transpose(np.diff(FBMpath, axis=0))
    
    
    for tt in np.arange(1,N_T+1):
        SDEpath[tt,:] = SDEpath[tt-1,:] + np.matmul(Theta, mu - SDEpath[tt-1,:] )*dt + np.matmul(G,dB[:,tt-1])

    Control_Path = FBMpath = np.c_[t, FBMpath]

    return [Control_Path, SDEpath.reshape((N_T+1,M))]


def sigmoid(x):
    return x/(np.sqrt(k)*d)
    # return np.divide(1, 1 + np.exp(-x))
    #return np.maximum(x/10,0)



def reservoirfield_Y(state,increment):
    value = np.zeros((k,1))
    for i in range(d):
        value = value + sigmoid(np.matmul(C[i],state) + deta[i])*increment[i]
    return value



def reservoir_Y(Control_Path):
    reservoirpath = [Z0]
    Increment_Storage = np.diff(Control_Path,axis=1)
    for i in range(N_T):
        increment = Increment_Storage[:,i]
        reservoirpath = reservoirpath + [(reservoirpath[-1]+reservoirfield_Y(reservoirpath[-1],increment))]
    return reservoirpath   

# Decleare the RDE Object and plot the Random Signature, jsut to see how they look.

# In[23]:
# sig_X=torch.zeros([N_S,N_T,dim])
# Reservoir_Y_Sig = np.zeros((1,M))
# print("Inizio Estrazione")
# tic()
# np.random.seed(1000)
# for l in range(N_S):
    
#     if np.mod(l,10)==0:
#         print(l)

#     Joint_Path = path()
#     Y_Path = Joint_Path[1]
#     Control_Path = Joint_Path[0]
#     x=torch.Tensor(np.expand_dims(Control_Path,0))
#     sig_X[l,:,:] = signatory.signature(x, signatures_dim,stream=True)
#     Reservoir_Y_Sig = np.r_[Reservoir_Y_Sig, Y_Path[1:,]]

# Extraction_Time_Sig_Train =toc()

# print(Extraction_Time_Sig_Train)

# # In[23]:

# Signatures =np.reshape(sig_X,(-1,dim))
# Reservoir_Sig_Train = Signatures.numpy()

# Reservoir_Y_Sig = np.delete(Reservoir_Y_Sig, (0), axis=0)
# Reservoir_Y_Sig = Reservoir_Y_Sig.reshape((Reservoir_Y_Sig.shape[0],M))

# Reservoir_Sig_Train=Reservoir_Sig_Train
# Reservoir_Y_Sig=Reservoir_Y_Sig
    
# # In[23]:
    
    
    
# lm_Y_Sig = linear_model.Ridge(alpha=0.001, solver='svd')#
# print("Inizio Allenamento")
# tic()

# model_Y_Sig  = lm_Y_Sig.fit(Reservoir_Sig_Train,Reservoir_Y_Sig)

# Training_Time_Sig_Train =toc()
# print(Training_Time_Sig_Train)


    
# In[23]:

Reservoir_RSig_Train=np.zeros([N_S,N_T,k])
Reservoir_Y_RSig = np.zeros((1,M))
    

print("Inizio Estrazione")
tic()
np.random.seed(1000)
for l in range(N_S):
    
    if np.mod(l,10)==0:
        print(l)
    
    Joint_Path = path()
    Y_Path = Joint_Path[1]
    Control_Path = Joint_Path[0]
    
    Control_Path_RSig = []
    for i in range(d):
        Control_Path_RSig = Control_Path_RSig + [Control_Path[:,i]]
    
    Features = np.squeeze(reservoir_Y(Control_Path_RSig))
    Reservoir_RSig_Train[l,:,:] = Features[1:,:]
    Reservoir_Y_RSig = np.r_[Reservoir_Y_RSig, Joint_Path[1][1:,:]]
    
Extraction_Time_RSig_Train =toc()
print(Extraction_Time_RSig_Train)

Reservoir_Y_RSig = np.delete(Reservoir_Y_RSig, (0), axis=0)
Reservoir_Y_RSig = Reservoir_Y_RSig.reshape((Reservoir_Y_RSig.shape[0],M))
Reservoir_RSig_Train =np.reshape(Reservoir_RSig_Train,(-1,k))


    
# In[23]:
    
    
    
lm_Y_RSig = linear_model.Ridge(alpha=0.001, solver='svd')#
print("Inizio Allenamento")
tic()
model_Y_RSig  = lm_Y_RSig.fit(Reservoir_RSig_Train,Reservoir_Y_RSig)

Training_Time_RSig_Train =toc()
print(Training_Time_RSig_Train)


##############################################################################
    
# In[23]:
    
# Reservoir_Pred_Y_Sig = np.zeros((N_C,N_T,d-1))
# Reservoir_Y_True_Sig = np.zeros((N_C,N_T,d-1))
    
# print("Inizio Estrazione e Valutazione")

# np.random.seed(10000000)
# for l in range(N_C):
    
#     if np.mod(l,10)==0:
#         print(l)
    
#     Joint_Path_Test = path()
#     Y_Path_Test  = Joint_Path_Test [1]
#     Control_Path_Test  = Joint_Path_Test [0]
#     x_Test =torch.Tensor(np.expand_dims(Control_Path_Test ,0))
    
#     sig_X_Test  = signatory.signature(x_Test , signatures_dim,stream=True)
#     Signatures_Test  =np.reshape(sig_X_Test ,(-1,dim))
#     Signatures_Test  = Signatures_Test.numpy()
#     Y_Pred_Sig = lm_Y_Sig.predict(Signatures_Test)
    
#     Reservoir_Y_True_Sig[l,:,:] =  Y_Path_Test[1:,:]
#     Reservoir_Pred_Y_Sig[l,:,:]  = Y_Pred_Sig
    




# In[23]:
    

Reservoir_Pred_Y_RSig = np.zeros((N_C,N_T,d-1))
Reservoir_Y_True_RSig = np.zeros((N_C,N_T,d-1))  
    
print("Inizio Estrazione e Valutazione")

np.random.seed(10000000)
for l in range(N_C):
    
    if np.mod(l,10)==0:
        print(l)
    
    
    
    Joint_Path_Test = path()
    Y_Path_Test  = Joint_Path_Test [1]
    Control_Path_Test  = Joint_Path_Test [0]

    Control_Path_RSig_Test = []
    for i in range(d):
        Control_Path_RSig_Test = Control_Path_RSig_Test + [Control_Path_Test[:,i]]
    
    Features_Test = np.squeeze(reservoir_Y(Control_Path_RSig_Test))
    Y_Pred_RSig = lm_Y_RSig.predict(Features_Test)
    
    
    Reservoir_Y_True_RSig[l,:,:] = Y_Path_Test[1:,:]
    Reservoir_Pred_Y_RSig[l,:,:] = Y_Pred_RSig[1:,:]
    



    
# In[23]:
    
index = 1
    
Path_to_Plot = Y_Path_Test[1:,:][:,index]
Pred_RSig = Y_Pred_RSig[1:,:][:,index]
# Pred_Sig = Y_Pred_Sig[:,index]
        
plt.figure()
# We plot
line_up, = plt.plot(Control_Path_Test[1:,0],Path_to_Plot, color = (0.138, 0.484, 0.782),linewidth=4, label='True')
line_down, = plt.plot(Control_Path_Test[1:,0],Pred_RSig, color = (0.93, 0.525, 0.219),linewidth=3, linestyle='dashed', label='RSig')
# line_up2, = plt.plot(Control_Path_Test[1:,0],Pred_Sig, 'g',linewidth=3, linestyle='dashed', label='Sig')
# line_err_down, = plt.plot(Joint_Path_Test[0],Y_Extracted-2*Y_Extracted_err, 'g',linewidth=1, linestyle='dashed', label='True')
# plt.legend([line_up, line_down,line_err_up,line_err_down], ['Extracted ' + r'$\hat{Y}_{t}$', 'True ' + r'$Y_{t}$'],fontsize=15)
# plt.legend([line_up, line_down,line_up2], ['True', 'RSig', 'Sig' ],fontsize=15)
plt.title("Out Of Sample",fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Value',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.savefig(path_print / "Out_of_Sample_Comparison_of_True_vs_Extracted_Y_Levels.pdf", bbox_inches='tight', dpi=quality)
plt.show()
   
    
    
    
    
    
# In[23]:
    
    

# Tonio_Measure_Sig=  Tonio_Measure_all(Reservoir_Y_True_Sig,Reservoir_Pred_Y_Sig)     
Tonio_Measure_RSig= Tonio_Measure_all(Reservoir_Y_True_RSig,Reservoir_Pred_Y_RSig)     
  

    
# In[23]:
    
    
print("Extraction Time Sig")
# print(Extraction_Time_Sig_Train)

print()
print()


print("Training Time Sig")
# print(Training_Time_Sig_Train)

print()
print()


print("All Time Sig")
# print(Extraction_Time_Sig_Train + Training_Time_Sig_Train)

print()
print()

print("Extraction Time RSig")
print(Extraction_Time_RSig_Train)

print()
print()


print("Training Time RSig")
print(Training_Time_RSig_Train)

print()
print()

print("All Time Sig")
print(Extraction_Time_RSig_Train + Training_Time_RSig_Train)

print()
print()

    
print("Mean Tonio Sig")
# print(np.mean(Tonio_Measure_Sig))

print()
print()


print("Mean Tonio RSig")
print(np.mean(Tonio_Measure_RSig))


    

# print(np.median(Tonio_Measure_Sig))
# print(np.median(Tonio_Measure_RSig))

    
# In[23]:
    

import sys

print('This message will be displayed on the screen.')

original_stdout = sys.stdout # Save a reference to the original standard output

with open(path_print /'filename.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    
 
        
    print("Extraction Time Sig")
    # print(Extraction_Time_Sig_Train)
    
    print()
    print()
    
    
    print("Training Time Sig")
    # print(Training_Time_Sig_Train)
    
    print()
    print()
    
    
    print("All Time Sig")
    # print(Extraction_Time_Sig_Train + Training_Time_Sig_Train)
    
    print()
    print()
    
    print("Extraction Time RSig")
    print(Extraction_Time_RSig_Train)
    
    print()
    print()
    
    
    print("Training Time RSig")
    print(Training_Time_RSig_Train)
    
    print()
    print()
    
    print("All Time Sig")
    print(Extraction_Time_RSig_Train + Training_Time_RSig_Train)
    
    print()
    print()
    
        
    print("Mean Tonio Sig")
    # print(np.mean(Tonio_Measure_Sig))
    
    print()
    print()
    
    
    print("Mean Tonio RSig")
    print(np.mean(Tonio_Measure_RSig))
    

    


    
    
    sys.stdout = original_stdout # Reset the standard output to its original value
