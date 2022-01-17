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

np.random.seed(10)
plt.rcParams.update({'font.size': 15})
plt.rcParams["figure.figsize"] = (7,5)
import seaborn as sns
sns.color_palette("colorblind")
# Here we declare some parameters:

# In[2]:

d = 10
signatures_dim = 6
dim= np.int((d**(signatures_dim+1)-1)/(d-1)-1)    

max_k = 200

# Number of controls
# Size of reservoir

epsilon = 1

# Number of timesteps in which the split the time span [0, T]
N_T = 200
# Number of rain Sample
N_S = 1


# For a given reconstruction error epsilon and N_T, it will tell the minimum k to use.

print((24*np.log(N_T))/(3*epsilon**2 - 2*epsilon**3))
JL_Limit = np.ceil((24*np.log(N_T))/(3*epsilon**2 - 2*epsilon**3))

passo=1
scala=0.3


# Decided where to put the outputs. You have to change this...

# In[3]:


quality = 1000
# Target_Path_Folder = r"C:\Users\eneam\Dropbox\Research\Thesis\GBM_Signal_Extraction_GBM_GBM_Few_Shit_Student_" + str(mu) + "_" + str(sigma) + "_"  + str(mu_2) + "_" + str(sigma_2) + "_" + str(N_T) + "_" + str(M) + "_" + str(today).replace("-", "_")
Target_Path_Folder = r"C:\Users\eneam\Dropbox\Research\Rough_Paper\Outputs\ICLR\Learn_Signatures_Multi_Bigger_20210930_" + str(N_T) + "_" + str(max_k) + "_" + str(N_S) + "_" + str(signatures_dim) + "_" + str(scala)
Path(Target_Path_Folder).mkdir(parents=True, exist_ok=True)
path = Path(Target_Path_Folder)


# Now we define some utilities

# In[22]:



def randomAbeta(d,M):
    A = []
    beta = []
    for i in range(d):
        B = np.random.normal(0.0,1.0,size=(M,M))
        # B = np.random.standard_t(2,size=(M,M)) 
        A = A + [B]
        beta = beta + [np.random.normal(0.0,1.0,size=(M,1))]
        # beta =  beta + [np.random.standard_t(2,size=(M,1))  ]
    return [A,beta]



def sigmoid(x,alpha):
    # return x/12 #Luca
    return x*alpha



def reservoirfield_Y(state,increment, C, deta, alpha, k):
    value = np.zeros((k,1))
    for i in range(d):
        value = value + sigmoid(np.matmul(C[i],state) + deta[i],alpha)*increment[i]
    return value




class RDE:
    def __init__(self,timehorizon,initialvalue,timesteps,):
        self.timehorizon = timehorizon
        self.initialvalue = initialvalue # np array
        self.timesteps = timesteps

    def path(self):
        
        t = np.arange(0, self.timehorizon + self.timehorizon/self.timesteps, self.timehorizon/self.timesteps)
        dB= np.sqrt(self.timehorizon/self.timesteps) * np.random.randn(self.timesteps)
        BMpath  = np.insert(np.cumsum(dB),0,0)
        
        for i in range(1,d-1):
            dB_temp = np.sqrt(self.timehorizon/self.timesteps) * np.random.randn(self.timesteps)
            BMpath_temp  = np.insert(np.cumsum(dB_temp),0,0)
            BMpath = np.c_[BMpath, BMpath_temp]
        
        return [t, BMpath]

    
    def reservoir_Y(self,Control_Path, C, deta, alpha, Z0, k):
        reservoirpath = [Z0]
        Increment_Storage = np.diff(Control_Path,axis=1)
        for i in range(self.timesteps):
            increment = Increment_Storage[:,i]
            reservoirpath = reservoirpath + [(reservoirpath[-1]+reservoirfield_Y(reservoirpath[-1], increment, C, deta, alpha, k))]
        return reservoirpath   



def Error_Calculator(Features_Reservoir,Signatures_Train):
    
    lm_Y = linear_model.Ridge(alpha=0.001)#
    model_Y  = lm_Y.fit(Features_Reservoir,Signatures_Train)
    Sig_Extracted = model_Y.predict(Features_Reservoir)

    
    L2_Error = 0
    L2_Rel_Error = 0
    for l in range(dim):
    
        L2_Error = L2_Error + np.sum(np.square(Sig_Extracted[:,l]-Signatures_Train[:,l]))
        L2_Rel_Error = L2_Rel_Error + np.sum(np.square(Sig_Extracted[:,l]-Signatures_Train[:,l]))/np.sum(np.square(Signatures_Train[:,l]))

    return [L2_Error, L2_Rel_Error]


# Decleare the RDE Object and plot the Random Signature, jsut to see how they look.

# In[23]:
    
OU_RDE = RDE(1,1.0,N_T)
Joint_Path = OU_RDE.path()
Y_Path = Joint_Path[1]
Control_Path = [Joint_Path[0]]
for i in range(d-1):
    Control_Path = Control_Path + [Y_Path[:,i]]
Z0_Source = np.random.normal(0.0,1.0,size=(max_k,1))

# In[23]:
    

sig_X=torch.zeros([1,N_T,dim])
x_test = np.vstack(Control_Path)
X_te=np.expand_dims(x_test.T,0)


for j in range(2,N_T+2):
    x=torch.Tensor(X_te[0,:j,:])
    sig_X[0,j-2,:]=signatory.signature(x.unsqueeze(0), signatures_dim)

Signatures =np.reshape(sig_X,(-1,dim))
Signatures_Train = Signatures.numpy()
    


# In[24]:
# List_L2_Errors_Quad = []
List_L2_Errors_Sqrt = []

asse_x = [i for i in range(2,max_k,passo)]  

for j in asse_x:
    print("Ordine " + str(j))
    Z0 = Z0_Source[:j,0].reshape((j,1))
    CDeta = randomAbeta(d,j)
    C = CDeta[0]
    deta = CDeta[1]
    Features_Sqrt= np.squeeze(OU_RDE.reservoir_Y(Control_Path, C, deta, 1/np.sqrt(j) , Z0, j))[1:,:]
    # Features_Quad = np.squeeze(OU_RDE.reservoir_Y(Control_Path, C, deta, scala/np.sqrt(np.sqrt(j)) , Z0, j))[1:,:]
    
    # plt.figure()
    # plt.plot(Joint_Path[0][1:],Features_Quad)
    # plt.show()
    # plt.figure()
    # plt.plot(Joint_Path[0][1:],Features_Sqrt)
    # plt.show()
    
    # List_L2_Errors_Quad = List_L2_Errors_Quad + [Error_Calculator(Features_Quad,Signatures_Train)[0]]
    List_L2_Errors_Sqrt = List_L2_Errors_Sqrt + [Error_Calculator(Features_Sqrt,Signatures_Train)[0]]







# In[24]:

    
# plt.figure()
# # We plot
# # line_up, = plt.plot(asse_x,np.log(List_L2_Errors_Quad), color = (0.138, 0.484, 0.782),linewidth=4, label='Quad')
# line_down, = plt.plot(asse_x,np.log(List_L2_Errors_Sqrt), color = (0.138, 0.666, 0.222),linewidth=4, label='Sqrt')
# plt.legend([line_up,line_down], ['Quad', 'Sqrt'])
# plt.title("Log L2 Error",fontsize=15)
# plt.xlabel('k',fontsize=15)
# plt.ylabel('Value',fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.savefig(path / "Log_L2_Error_Across_Signatures.pdf", bbox_inches='tight', dpi=quality)
# plt.show()

# In[24]:

    
plt.figure()
# We plot
line_up, = plt.plot(asse_x[:200],np.log(List_L2_Errors_Sqrt[:200]), color = (0.138, 0.484, 0.782),linewidth=4, label='Quad')
plt.title("L2 Log-Error",fontsize=15)
plt.xlabel('k',fontsize=15)
plt.ylabel('Value',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(path / "Log_L2_Error_Across_Signatures_Only_1_Error.pdf", bbox_inches='tight', dpi=quality)
plt.show()



# In[24]:
# List_L2_Errors_Subset_Quad = []
# List_L2_Errors_Subset_Sqrt = []

# asse_x = [i for i in range(2,max_k,passo)]  

# for j in asse_x:
#     print("Ordine " + str(j))
#     List_L2_Errors_Subset_Quad = List_L2_Errors_Subset_Quad + [Error_Calculator(Features_Quad[:,:j],Signatures_Train)[0]]
#     List_L2_Errors_Subset_Sqrt = List_L2_Errors_Subset_Sqrt + [Error_Calculator(Features_Sqrt[:,:j],Signatures_Train)[0]]
    
    

# In[24]:

# import seaborn as sns
# with sns.color_palette("colorblind"):
    
#     plt.figure()
#     # We plot
#     line_up, = plt.plot(asse_x,np.log(List_L2_Errors_Sqrt),linewidth=4, label='Full Sqrt')
#     line_down, = plt.plot(asse_x,np.log(List_L2_Errors_Subset_Sqrt),linewidth=4, label='Sub Sqrt')
#     line_up2, = plt.plot(asse_x,np.log(List_L2_Errors_Quad),linewidth=4, label='Full Quad')
#     line_down2, = plt.plot(asse_x,np.log(List_L2_Errors_Subset_Quad),linewidth=4, label='Sub Quad')
#     plt.legend([line_up, line_down,line_up2, line_down2], ['Full Sqrt','Sub Sqrt','Full Quad','Sub Quad',])
#     plt.title("Error Comparison",fontsize=15)
#     plt.xlabel('k',fontsize=15)
#     plt.ylabel('Value',fontsize=15)
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.savefig(path / "Log_L2_Error_Across_Signatures_Comparison.pdf", bbox_inches='tight', dpi=quality)
#     plt.show()

File_Name = "Errors.txt"
Name = path / File_Name
np.savetxt(Name, [np.log(List_L2_Errors_Sqrt)])
Prova = np.loadtxt(Name)