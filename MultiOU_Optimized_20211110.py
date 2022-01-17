#!/usr/bin/env python
# coding: utf-8

# Here we import some libraries that will come handy aftwerward.

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import pandas as pd
from pathlib import Path
from datetime import datetime
#from scipy.stats import shapiro
from scipy.stats import shapiro
now = datetime.now()
time = now.strftime("%Y%m%d_%H%M%S")
import seaborn as sns
plt.rcParams.update({'font.size': 15})
plt.rcParams["figure.figsize"] = (7,5)
import seaborn as sns
sns.color_palette("colorblind")
from ttictoc import tic, toc

np.random.seed(0)

# Here we declare some parameters:

# In[2]:


# Number of controls

# Size of reservoir
k = 111

M = 19
d = M + 1

epsilon = 1

# Number of timesteps in which the split the time span [0, T]
N_T = 100
N_S = 1
N_C = 10000



Theta = np.ones((M,M))

for i in range(M):
    for j in range(M):
        Theta[i,j] = np.round((j+1)/(i+1),2)
        Theta[j,i] = np.round((j+1)/(i+1),2)

# Theta = np.array([[1, 0.5],[0.5, 1]])
G = np.diag([1 for x in range(M)])
mu = np.ones((M,))

print((24*np.log(N_T))/(3*epsilon**2 - 2*epsilon**3))
print(k > (24*np.log(N_T))/(3*epsilon**2 - 2*epsilon**3))



Z0 = np.random.uniform(-1, 1, size=(k,1))

# print(2*speed*mean>vol**2)


# In[14]:


quality = 1000
# Target_Path_Folder = r"C:\Users\eneam\Dropbox\Research\Thesis\GBM_Signal_Extraction_GBM_GBM_Few_Shit_Student_" + str(mu) + "_" + str(sigma) + "_"  + str(mu_2) + "_" + str(sigma_2) + "_" + str(N_T) + "_" + str(k) + "_" + str(today).replace("-", "_")
Target_Path_Folder = r"C:\Users\eneam\Dropbox\Research\Rough_Paper\Outputs\ICLR\Multidim_OU_20211115_Tempi_Updated_" + str(N_T) + "_" + str(k) + "_" + str(N_S) + "_" + str(M)
Path(Target_Path_Folder).mkdir(parents=True, exist_ok=True)
path = Path(Target_Path_Folder)


# Now we define some utilities

# In[4]:



# This funcion calculates the autocorrelation of elements in x for a given lag
def auto_correlation(x,lag = 1):
	a = pd.Series(np.reshape(x,(-1)))
	b = a.autocorr(lag = lag)
	if np.isnan(b) or np.isinf(b):
		return 0
	return b

# This funcion returns a correlogram of x
def acf(x,max_lag=13):
	acf = []
	for i in range(max_lag):
		acf.append(auto_correlation(x,lag=i+1))
	return np.array(acf)


def canonical(i,M):
    e = np.zeros((M,1))
    e[i,0]=1.0
    return e


# def randomAbeta(d,M):
#     A = []
#     beta = []
#     for i in range(d):
#         # B = 0.0*nilpotent(M) + np.random.standard_t(2,size=(M,M)) 
#         B = np.zeros((M,M))
#         Temp = np.random.normal(0.0,1.0,size=(1,M)) 
#         Temp = (Temp-np.mean(Temp))/np.std(Temp)
#         for i in range(M):
#             B[i,i] = Temp[0,i]
            

#         # B = 0.0*nilpotent(M) + np.random.normal(0.0,1.0,size=(M,M)) 
#         # B = np.random.permutation(B)
#         A = A + [B]
#         beta = beta + [0.0*canonical(i,M)+np.random.uniform(-1, 1, size=(M,1))]
#         # beta = beta + [0.0*canonical(i,M)+np.random.normal(0.0,1.0,size=(M,1))]
#     return [A,beta]

def randomAbeta(d,M):
    A = []
    beta = []
    for i in range(d):
        # B = 0.0*nilpotent(M) + np.random.standard_t(2,size=(M,M)) 
        B = np.random.normal(0.0,1.0,size=(M,M)) 
        # B = np.random.permutation(B)
        A = A + [B]
        # beta = beta + [0.0*canonical(i,M)+np.random.standard_t(2,size=(M,1))]
        beta = beta + [np.random.normal(0.0,1.0,size=(M,1))]
    return [A,beta]


CDeta = randomAbeta(d,k)
C = CDeta[0]
deta = CDeta[1]


def sigmoid(x):
    return x/(np.sqrt(k)*d)
    # return np.divide(1, 1 + np.exp(-x))
    #return np.maximum(x/10,0)



def reservoirfield_Y(state,increment):
    value = np.zeros((k,1))
    for i in range(d):
        value = value + sigmoid(np.matmul(C[i],state) + deta[i])*increment[i]
    return value


    
class RDE:
    def __init__(self,timehorizon,initialvalue,dimensionR,timesteps,):
        self.timehorizon = timehorizon
        self.initialvalue = initialvalue # np array
        self.dimensionR = dimensionR
        self.timesteps = timesteps

    def path(self):
        
        t = np.arange(0, self.timehorizon + self.timehorizon/self.timesteps, self.timehorizon/self.timesteps)
        dB= np.sqrt(self.timehorizon/self.timesteps) * np.random.randn(self.timesteps)
        BMpath  = np.insert(np.cumsum(dB),0,0)
        
        for i in range(1,M):
            dB_temp = np.sqrt(self.timehorizon/self.timesteps) * np.random.randn(self.timesteps)
            BMpath_temp  = np.insert(np.cumsum(dB_temp),0,0)
            BMpath = np.c_[BMpath, BMpath_temp]
        dB = np.transpose(np.diff(BMpath, axis=0))
        
        
        dt = t[1] - t[0]
        SDEpath = np.empty((N_T+1,M))
        SDEpath[0,:] = 1
        
        for tt in np.arange(1,N_T+1):
            SDEpath[tt,:] = SDEpath[tt-1,:] + np.matmul(Theta, mu - SDEpath[tt-1,:] )*dt + np.matmul(G,dB[:,tt-1])
            # SDEpath[:,tt]  = np.maximum(SDEpath[:,tt], 0.00001)


        return [t, BMpath, SDEpath.reshape((N_T+1,M))]


    
    def reservoir_Y(self,Control_Path):
        reservoirpath = [Z0]
        Increment_Storage = np.diff(Control_Path,axis=1)
        for i in range(self.timesteps):
            increment = Increment_Storage[:,i]
            reservoirpath = reservoirpath + [(reservoirpath[-1]+reservoirfield_Y(reservoirpath[-1],increment))]
        return reservoirpath   

    
def Tonio_Measure(serie_1, serie_2):
    numerator = np.sum(np.square(serie_1-serie_2),axis=0)
    denominator = np.sum(np.square(serie_1),axis=0)
    return np.mean(numerator/denominator)
    
    
def Tonio_Measure_all(df1,df2):
    df_tonio = np.empty((df1.shape[0],0))
    for i in range(df1.shape[0]):
        df_tonio = np.insert(df_tonio,0,Tonio_Measure(df1[i,:], df2[i,:]))
    return df_tonio


def Average_Increment_Calculator(df):
    Increments = np.diff(df,axis=1)
    return np.mean(Increments, axis=1)
    
def Std_Increment_Calculator(df):
    Increments = np.diff(df,axis=1)
    return np.std(Increments, axis=1) 

def Autocorr_Increment_Calculator(df):
    Increments = np.diff(df,axis=1)
    
    df_acf = np.empty((Increments.shape[0],0))

    
    for i in range(Increments.shape[0]):
        acf_temp = acf(Increments[i,:],nlags=1,fft=True)
        df_acf = np.insert(df_acf,0,acf_temp[1])
    
    return df_acf

def pvalue_normality_Increment_Calculator(df):
    Increments = np.diff(df,axis=1)
    df_p = np.empty((Increments.shape[0],0))

    for i in range(Increments.shape[0]):
        stat, p = shapiro(Increments[i,:])
        df_p = np.insert(df_p,0,p)
    
    return df_p


def Quadratic_Variation_Calculator(df):
    return np.sum(np.square(np.diff(df,axis=1)), axis=1) 


# Now we actually generate the trjectory and relative controls & reservoirs from which we will try to learn the r-Sig

# In[5]:


# Declare the object
OU_RDE = RDE(1,1.0,k,N_T)
# Generate the control paths and the target trajectory
Joint_Path = OU_RDE.path()


# # Plot RDE path along with controls
# labels = ["Time", "BM_Path", "OU_Path"]
# colors = ["r","g","b"]

# f,axs = plt.subplots(3, sharex=True, sharey=True)

# # ---- loop over axes ----
# for i,ax in enumerate(axs):
#   axs[i].plot(Joint_Path[i],color=colors[i],label=labels[i])
#   axs[i].legend(loc='best')
# f.suptitle('Evolution of Control and Output Paths', fontsize=16)
# plt.savefig(path / "Example_Time_BM_OU_Paths.pdf", dpi=quality)
# plt.show()

Joint_Path = OU_RDE.path()
Y_Path = Joint_Path[1]


############# NOW WE EXTRACT FEATURES OF THE TARGET PATH

Control_Path = [Joint_Path[0]]
for i in range(M):
    Control_Path = Control_Path + [Y_Path[:,i]]

Features = np.squeeze(OU_RDE.reservoir_Y(Control_Path))

with sns.color_palette("colorblind", n_colors=8):
    plt.plot(Joint_Path[0],Features[:,[0,1,2,3,4,5,6,7]],linewidth=3)
plt.title("Random Signatures (8/444 plotted)",fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Value',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(path / "Random_Signature.pdf",bbox_inches='tight', dpi=quality)
plt.show()

with sns.color_palette("colorblind", n_colors=8):
    plt.plot(Joint_Path[0],Features,linewidth=3)
plt.title("Random Signatures (8/444 plotted)",fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Value',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(path / "Random_Signature_All.pdf",bbox_inches='tight', dpi=quality)
plt.show()


# In[5]:

tic()

Features_Reservoir=np.zeros([N_S,N_T+1,k],dtype=np.float32)
Y_Reservoir = np.zeros((1,M))

for i in range(N_S):
    
    if np.mod(i,10)==0:
        print(i)

    Joint_Path = OU_RDE.path()
    Y_Path = Joint_Path[1]

    Control_Path = [Joint_Path[0]]
    for l in range(M):
        Control_Path = Control_Path + [Y_Path[:,l]]
    
    Features_Reservoir[i,:,:] = np.squeeze(OU_RDE.reservoir_Y(Control_Path))
    Y_Reservoir = np.r_[Y_Reservoir, Joint_Path[2]]
    
Generation_Time = toc()

    ### Train lin model to learn how to extract BM

Y_Reservoir = np.delete(Y_Reservoir, (0), axis=0)

Features_Reservoir =np.reshape(Features_Reservoir,(-1,k))



print("Fatto")

# In[5]:
    
tic()

lm_Y = linear_model.Ridge(alpha=0.001, solver='svd')#
model_Y  = lm_Y.fit(Features_Reservoir,Y_Reservoir)

Training_Time = toc()

print("Trained")


# In[6]:


# Joint_Path = OU_RDE.path()
# Y_Path = Joint_Path[1]


# ############# NOW WE EXTRACT FEATURES OF THE TARGET PATH

# Control_Path = [Joint_Path[0]]
# for i in range(M):
#     Control_Path = Control_Path + [Y_Path[:,i]]
# Features = np.squeeze(OU_RDE.reservoir_Y(Control_Path))
# Y_Extracted = model_Y.predict(Features)
# Y_True = Joint_Path[2]


# labels_True = [ 'True ' + r'$Y^{1}_{t}$', 'True ' + r'$Y^{2}_{t}$', 'True ' + r'$Y^{3}_{t}$', 'True ' + r'$Y^{4}_{t}$']
# labels_Extracted = [ 'Extracted ' + r'$\hat{Y}^{1}_{t}$', 'Extracted ' + r'$\hat{Y}^{2}_{t}$', 'Extracted ' + r'$\hat{Y}^{3}_{t}$', 'Extracted ' + r'$\hat{Y}^{4}_{t}$']

# fig = plt.figure(figsize=(32, 8))
# coord = []# create coord array from 241, 242, 243, ..., 248 
# for i in range(1, 5): # in python, 9 is not included
#     row = 1
#     column = 4
#     coord.append(str(row)+str(column)+str(i))
    
# # create subplot 241, 242, 243, ..., 248
# for i in range(len(coord)):
#     plt.subplot(coord[i])
#     line_up, = plt.plot(Joint_Path[0],Y_Extracted[:,i], color = (0.138, 0.484, 0.782),linewidth=6, label='LTL')
#     line_down, = plt.plot(Joint_Path[0],Y_True[:,i], color = (0.93, 0.525, 0.219),linewidth=4, linestyle='dashed', label='True')
#     if i==0:
#         plt.legend([line_up, line_down], [labels_Extracted[i], labels_True[i]],fontsize=25)
#         plt.ylabel('Value',fontsize=25)
#     plt.xticks(fontsize=25)
#     plt.yticks(fontsize=25)
#     plt.xlabel('Time',fontsize=25)
# plt.savefig(path / str("Out_of_Sample_Comparison_of_True_vs_Extracted_BM_Levels.pdf"), bbox_inches='tight', dpi=quality)
# plt.show()


# In[6]:

Joint_Path = OU_RDE.path()
Y_Path = Joint_Path[1]


############# NOW WE EXTRACT FEATURES OF THE TARGET PATH

Control_Path = [Joint_Path[0]]
for i in range(M):
    Control_Path = Control_Path + [Y_Path[:,i]]
Features = np.squeeze(OU_RDE.reservoir_Y(Control_Path))
Y_Extracted = model_Y.predict(Features)
Y_True = Joint_Path[2]


labels_True = [ 'True ' + r'$Y^{1}_{t}$', 'True ' + r'$Y^{2}_{t}$', 'True ' + r'$Y^{3}_{t}$', 'True ' + r'$Y^{4}_{t}$', 'True ' + r'$Y^{5}_{t}$', 'True ' + r'$Y^{6}_{t}$', 'True ' + r'$Y^{7}_{t}$', 'True ' + r'$Y^{8}_{t}$', 'True ' + r'$Y^{9}_{t}$']
labels_Extracted = [ 'Extracted ' + r'$\hat{Y}^{1}_{t}$', 'Extracted ' + r'$\hat{Y}^{2}_{t}$', 'Extracted ' + r'$\hat{Y}^{3}_{t}$', 'Extracted ' + r'$\hat{Y}^{4}_{t}$', 'Extracted ' + r'$\hat{Y}^{5}_{t}$', 'Extracted ' + r'$\hat{Y}^{6}_{t}$', 'Extracted ' + r'$\hat{Y}^{7}_{t}$', 'Extracted ' + r'$\hat{Y}^{8}_{t}$', 'Extracted ' + r'$\hat{Y}^{9}_{t}$']

    
# create subplot 241, 242, 243, ..., 248
for i in range(4):
    plt.figure()
    line_up, = plt.plot(Joint_Path[0],Y_Extracted[:,i], color = (0.138, 0.484, 0.782),linewidth=6, label='LTL')
    line_down, = plt.plot(Joint_Path[0],Y_True[:,i], color = (0.93, 0.525, 0.219),linewidth=4, linestyle='dashed', label='True')
    plt.legend([line_up, line_down], [labels_Extracted[i], labels_True[i]])
    plt.ylabel('Value',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Time',fontsize=15)
    Name = "Out_of_Sample_Comparison_of_True_vs_Extracted_Y_Levels" + str(i+1)
    plt.savefig(path / Name, bbox_inches='tight', dpi=quality)
    plt.show()


# In[6]:

Reservoir_Y_Extracted = np.zeros((N_C,N_T+1,d-1))
Reservoir_Y_True = np.zeros((N_C,N_T+1,d-1))



for i in range(N_C):
    
    if np.mod(i,10)==0:
        print(i)
    
    Joint_Path_Test = OU_RDE.path()
    Y_Path = Joint_Path_Test[1]


    ############# NOW WE EXTRACT FEATURES OF THE TARGET PATH

    Control_Path = [Joint_Path_Test[0]]
    for l in range(M):
        Control_Path = Control_Path + [Y_Path[:,l]]
    
    Features_Test_Extracted = np.squeeze(OU_RDE.reservoir_Y(Control_Path))
    Y_Test_Extracted = model_Y.predict(Features_Test_Extracted)
      

    ############### SU Y Real
    
    Reservoir_Y_Extracted[i,:,:] = Y_Test_Extracted  
    Reservoir_Y_True[i,:,:] = Joint_Path_Test[2] 
    

print("Fatto")
    

# In[11]:


Tonio_Measure_Extracted = Tonio_Measure_all(Reservoir_Y_True,Reservoir_Y_Extracted)


# Compare starting value

# In[11]:

    
print()
print()

print("Tonio mean Measure:")
print(np.mean(Tonio_Measure_Extracted))



print()
print()


print("Tonio median Measure:")
print(np.median(Tonio_Measure_Extracted))

print()
print()


print("Generation_Time:")
print(Generation_Time)


print("Training_Time:")
print(Training_Time)

print()
print()

    




import sys

print('This message will be displayed on the screen.')

original_stdout = sys.stdout # Save a reference to the original standard output

with open(path /'filename.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    
    
    print()
    print()
    
    print("Tonio mean Measure:")
    print(np.mean(Tonio_Measure_Extracted))
    
    
    
    print()
    print()
    
    
    print("Tonio median Measure:")
    print(np.median(Tonio_Measure_Extracted))
    
    print()
    print()
    
    
    print("Generation_Time:")
    print(Generation_Time)
    
    
    print("Training_Time:")
    print(Training_Time)
    
    print()
    print()


    
    
    sys.stdout = original_stdout # Reset the standard output to its original value