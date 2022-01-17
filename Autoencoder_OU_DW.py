#!/usr/bin/env python
# coding: utf-8

# Here we import some libraries that will come handy aftwerward.

# In[1]:


import numpy as np
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.stats import shapiro
from statsmodels.tsa.stattools import acf
from scipy.stats import ttest_1samp
from sklearn import linear_model
plt.rcParams.update({'font.size': 15})
plt.rcParams["figure.figsize"] = (7,5)
import seaborn as sns
sns.color_palette("colorblind")
now = datetime.now()
time = now.strftime("%Y%m%d_%H%M%S")

np.random.seed(0)
# Here we declare some parameters:

# In[2]:


# Number of controls
d = 1
# Size of reservoir
k = 50

epsilon = 1

# Number of timesteps in which the split the time span [0, T]
N_T = 150
# Number Train Samples
N_S = 1
N_C = 1

epsilon = 1

# Param to generate the series on which we train the autoencoder
speed_train = 1
mean_train  = 2
vol_train = 1

print((24*np.log(N_T))/(3*epsilon**2 - 2*epsilon**3))
print(k > (24*np.log(N_T))/(3*epsilon**2 - 2*epsilon**3))

# Parameters of the processes we use to test it.

# WE WILL SEE HOW IT REACTS IF WE THOUGH IN A DW AND A OU WITH THE FOLLOWING PARAMETWRS.
speed_test = 3
mean_test  = 6
vol_test   = 3

Z0 = np.random.normal(0.0,1.0,size=(k,1)) 

alpha = np.sqrt(k)

# Decided where to put the outputs. You have to change this...

# In[3]:


quality = 1000
# Target_Path_Folder = r"C:\Users\eneam\Dropbox\Research\Thesis\GBM_Signal_Extraction_GBM_GBM_Few_Shit_Student_" + str(mu) + "_" + str(sigma) + "_"  + str(mu_2) + "_" + str(sigma_2) + "_" + str(N_T) + "_" + str(M) + "_" + str(today).replace("-", "_")
Target_Path_Folder = r"C:\Users\eneam\Dropbox\Research\Rough_Paper\Outputs\ICLR\Autoencoder_OU_DW_Luca_" + str(speed_train) + "_" + str(mean_train) + "_"  + str(vol_train) + "_" + str(N_T) + "_" + str(k) + "_" + str(N_S) + "_" + str(alpha)
Path(Target_Path_Folder).mkdir(parents=True, exist_ok=True)
path = Path(Target_Path_Folder)


# Now we define some utilities

# In[4]:


def randomAbeta(d,M):
    A = []
    beta = []
    for i in range(d):
        B = np.random.normal(0.0,1.0,size=(M,M)) 
        A = A + [B]
        beta = beta + [np.random.normal(0.0,1.0,size=(M,1))]
    return [A,beta]


CDeta = randomAbeta(d,k)
C = CDeta[0]
deta = CDeta[1]


def sigmoid(x):
    return x/alpha


def reservoirfield_Y(state,increment):
    value = np.zeros((k,1))
    for i in range(d):
        value = value + sigmoid(np.matmul(C[i],state) + deta[i])*increment
    return value

    
class RDE:
    def __init__(self,timehorizon,initialvalue,dimensionR,timesteps,):
        self.timehorizon = timehorizon
        self.initialvalue = initialvalue # np array
        self.dimensionR = dimensionR
        self.timesteps = timesteps

    def path(self, mean, vol, speed, mean2, vol2, speed2):
        
        t = np.arange(0, self.timehorizon + self.timehorizon/self.timesteps, self.timehorizon/self.timesteps)
        dB = np.random.randn(self.timesteps)*np.sqrt(self.timehorizon/self.timesteps) 
        BMpath  = np.insert(np.cumsum(dB),0,0)

        dt = t[1] - t[0]
        SDEpath = np.empty((1,N_T+1))
        SDEpath[:, 0] = 1
        
        SDEpath2 = np.empty((1,N_T+1))
        SDEpath2[:, 0] = 1
        
        SDEpath3 = np.empty((1,N_T+1))
        SDEpath3[:, 0] = 1
        
        for tt in np.arange(1,N_T+1):
            SDEpath[:,tt]  = SDEpath [:,tt-1] + speed *                 (mean -SDEpath [:,tt-1]   )*dt + vol *dB[tt-1,]
            SDEpath2[:,tt] = SDEpath2[:,tt-1] + speed2*SDEpath2[:,tt-1]*(mean2-SDEpath2[:,tt-1]**2)*dt + vol2*dB[tt-1,]
            SDEpath3[:,tt] = SDEpath3[:,tt-1] + speed2*                 (mean2-SDEpath3[:,tt-1]   )*dt + vol2*dB[tt-1,]
            


        SDEpath  = SDEpath.reshape((N_T+1,))
        SDEpath2 = SDEpath2.reshape((N_T+1,))
        SDEpath3 = SDEpath3.reshape((N_T+1,))

        return [t, BMpath, SDEpath, SDEpath2, SDEpath3]


    
    def reservoir_Y(self,Control_Path):
        reservoirpath = [Z0]
        Increment_Storage = np.diff(Control_Path,axis=1)
        for i in range(self.timesteps):
            increment = Increment_Storage[:,i]
            reservoirpath = reservoirpath + [(reservoirpath[-1]+reservoirfield_Y(reservoirpath[-1],increment))]
        return reservoirpath   


    
def Tonio_Measure(serie_1, serie_2):
    numerator = np.sum(np.square(serie_1-serie_2))
    denominator = np.sum(np.square(serie_1))
    return numerator/denominator

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

def Tonio_Measure_all(df1,df2):
    df_tonio = np.empty((df1.shape[0],0))
    for i in range(df1.shape[0]):
        df_tonio = np.insert(df_tonio,0,Tonio_Measure(df1[i,:], df2[i,:]))
    return df_tonio

def Quadratic_Variation_Calculator(df):
    return np.sum(np.square(np.diff(df,axis=1)), axis=1) 


# Now we actually generate the trjectory and relative controls & reservoirs from which we will try to learn the r-Sig

# In[5]:


# Declare the object
OU_RDE = RDE(1,1.0,k,N_T)
Joint_Path = OU_RDE.path(mean_train, vol_train, speed_train, mean_test, vol_test, speed_test)  

t = np.arange(0, 1 + 1/N_T, 1/N_T)

# Here I just plot the RSig of a path from the law of the training (OU(mean_train, vol_train, speed_train))
plt.plot(np.squeeze(OU_RDE.reservoir_Y([Joint_Path[2]])))
plt.show()
# Here I just plot the RSig of a path from NOT the law of the training (DW(mean_test, vol_test, speed_test))
plt.plot(np.squeeze(OU_RDE.reservoir_Y([Joint_Path[3]])))
plt.show()
# Here I just plot the RSig of a path from NOT the law of the training (OU(mean_test, vol_test, speed_test))
plt.plot(np.squeeze(OU_RDE.reservoir_Y([Joint_Path[4]])))
plt.show()


# Generate the Reservoir of Signature and Targets.
# Explaination in the code

# In[6]:


Features_Reservoir=np.zeros([N_S,N_T+1,k])
Y_Reservoir = np.zeros((1,))

for i in range(N_S):
    
    if np.mod(i,10)==0:
        print(i)
    

    Joint_Path = OU_RDE.path(mean_train, vol_train, speed_train, mean_test, vol_test, speed_test)
    Features_Reservoir[i,:,:] = np.squeeze(OU_RDE.reservoir_Y([Joint_Path[2]]))
    Y_Reservoir = np.r_[Y_Reservoir, Joint_Path[2]]

print("Done")
Features_Reservoir =np.reshape(Features_Reservoir,(-1,k))

Y_Reservoir = np.delete(Y_Reservoir, (0), axis=0)
Y_Reservoir = Y_Reservoir.reshape((Y_Reservoir.shape[0],1))




lm_BM = linear_model.Ridge(alpha=0.001)#
model_BM  = lm_BM.fit(Features_Reservoir,Y_Reservoir)

coefficienti = model_BM.coef_

# plt.plot(np.sort(coefficienti).reshape((-1,1)))

plt.plot(np.argsort(coefficienti).reshape((-1,1)))

# In[7]:


plt.figure()
# We plot
line_up, = plt.plot(Joint_Path[0],model_BM.predict(Features_Reservoir), color = (0.138, 0.484, 0.782),linewidth=4, label='LTL')
line_down, = plt.plot(Joint_Path[0],Y_Reservoir, color = (0.93, 0.525, 0.219),linewidth=3, linestyle='dashed', label='True')
plt.legend([line_up, line_down], ['Extracted ' + r'$\hat{X}_{t}$', 'True ' + r'$X_{t}$'])
plt.title("Out of Sample",fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Value',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(path / "Out_of_Sample_Comparison_of_True_vs_Extracted_OU_True_Levels.pdf", bbox_inches='tight', dpi=quality)
plt.show()


errore = model_BM.predict(Features_Reservoir).reshape((-1,1))-Y_Reservoir.reshape((-1,1))

plt.figure()
# We plot
plt.plot(Joint_Path[0], errore, color = (0.138, 0.484, 0.782),linewidth=4, label='LTL')
plt.title("Out of Sample",fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Value',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(path / "Out_of_Sample_Comparison_of_True_vs_Extracted_OU_True_Levels.pdf", bbox_inches='tight', dpi=quality)
plt.show()

# Let us plot an OOS example

# In[9]:


Joint_Path_Test = OU_RDE.path(mean_train, vol_train, speed_train, mean_test, vol_test, speed_test) 

############# NOW WE EXTRACT FEATURES OF THE TARGET PATH

Features_True_OU =  np.squeeze(OU_RDE.reservoir_Y([Joint_Path_Test[2]]))
Y_True_OU_Extracted = model_BM.predict(Features_True_OU)

# Features_Fake_DW =  np.squeeze(OU_RDE.reservoir_Y([Joint_Path_Test[3]]))
# Y_Fake_DW_Extracted = model_BM.predict(Features_Fake_DW)

# Features_Fake_OU =  np.squeeze(OU_RDE.reservoir_Y([Joint_Path_Test[4]]))
# Y_Fake_OU_Extracted = model_BM.predict(Features_Fake_OU)



plt.figure()
# We plot
line_up, = plt.plot(Joint_Path_Test[0], Y_True_OU_Extracted, color = (0.138, 0.484, 0.782),linewidth=4, label='LTL')
line_down, = plt.plot(Joint_Path_Test[0],Joint_Path_Test[2], color = (0.93, 0.525, 0.219),linewidth=3, linestyle='dashed', label='True')
plt.legend([line_up, line_down], ['Extracted ' + r'$\hat{X}_{t}$', 'True ' + r'$X_{t}$'])
plt.title("Out of Sample",fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Value',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(path / "Out_of_Sample_Comparison_of_True_vs_Extracted_OU_True_Levels.pdf", bbox_inches='tight', dpi=quality)
plt.show()


errore = Y_True_OU_Extracted.reshape((-1,1))-Joint_Path_Test[2].reshape((-1,1))

plt.figure()
# We plot
plt.plot(Joint_Path_Test[0], errore, color = (0.138, 0.484, 0.782),linewidth=4, label='LTL')
plt.title("Out of Sample",fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Value',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(path / "Out_of_Sample_Comparison_of_True_vs_Extracted_OU_True_Levels.pdf", bbox_inches='tight', dpi=quality)
plt.show()


# plt.figure()
# # We plot
# line_up, = plt.plot(Joint_Path[0],Y_Fake_DW_Extracted, color = (0.138, 0.484, 0.782),linewidth=4, label='LTL')
# line_down, = plt.plot(Joint_Path[0],Joint_Path_Test[3], color = (0.93, 0.525, 0.219),linewidth=3, linestyle='dashed', label='True')
# plt.legend([line_up, line_down], ['Extracted ' + r'$\hat{Y}_{t}$', 'True ' + r'$Y_{t}$'])
# plt.title("Out of Sample",fontsize=15)
# plt.xlabel('Time',fontsize=15)
# plt.ylabel('Value',fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.savefig(path / "Out_of_Sample_Comparison_of_True_vs_Extracted_DW_Fake_Levels.pdf", bbox_inches='tight', dpi=quality)
# plt.show()

# plt.figure()
# # We plot
# line_up, = plt.plot(Joint_Path[0],Y_Fake_OU_Extracted, color = (0.138, 0.484, 0.782),linewidth=4, label='LTL')
# line_down, = plt.plot(Joint_Path[0],Joint_Path_Test[4], color = (0.93, 0.525, 0.219),linewidth=3, linestyle='dashed', label='True')
# plt.legend([line_up, line_down], ['Extracted ' + r'$\hat{Y}_{t}$', 'True ' + r'$Y_{t}$'])
# plt.title("Out of Sample",fontsize=15)
# plt.xlabel('Time',fontsize=15)
# plt.ylabel('Value',fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.savefig(path / "Out_of_Sample_Comparison_of_True_vs_Extracted_OU_Fake_Levels.pdf", bbox_inches='tight', dpi=quality)
# plt.show()



# In[9]:


Reservoir_Y_Extracted = np.zeros((1,N_T+1))
Reservoir_Y_True = np.zeros((1,N_T+1))

for i in range(N_C):
    
    if np.mod(i,100)==0:
        print(i)
    
     
    ############ TEST THE AUTOENCODER #############
    
    
    Joint_Path_Test = OU_RDE.path(mean_train, vol_train, speed_train, mean_test, vol_test, speed_test) 
    Y_Extracted = model_BM.predict(np.squeeze(OU_RDE.reservoir_Y([Joint_Path_Test[2]])))
    Reservoir_Y_Extracted = np.r_[Reservoir_Y_Extracted, Y_Extracted.reshape((1,N_T+1))]   
    Reservoir_Y_True = np.r_[Reservoir_Y_True, Joint_Path_Test[2].reshape((1,N_T+1))]   


print("Fatto")
    
Reservoir_Y_Extracted =  np.delete(Reservoir_Y_Extracted, 0, axis=0)
Reservoir_Y_True = np.delete(Reservoir_Y_True, 0, axis=0)


# In[10]:

Tonio_Measure_Extracted = Tonio_Measure_all(Reservoir_Y_True,Reservoir_Y_Extracted)


# Compare starting value

# In[11]:


print("Tonio Measure:")
print(np.mean(Tonio_Measure_Extracted))
print("Tonio Measure:")
print(np.median(Tonio_Measure_Extracted))
