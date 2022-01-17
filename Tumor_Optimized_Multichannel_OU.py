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
now = datetime.now()
time = now.strftime("%Y%m%d_%H%M%S")
import seaborn as sns

np.random.seed(2**31)


# Here we declare some parameters:

# In[2]:



# Number of controls
d = 2
# Size of reservoir
k = 222

epsilon = 1

# Number of timesteps in which the split the time span [0, T]
N_T = 100
# Number of Train Sample
N_S = 100
# Number of Test Samples
N_C = 1

N_Channels = 1


k1 = 10
k2 = 0.5
l0 = 0.9
l1 = 0.7
p = 20

mean = 2
speed = 2
vol = 0.25


epsilon = 1

# For a given reconstruction error epsilon and N_T, it will tell the minimum k to use.

print((24*np.log(N_T))/(3*epsilon**2 - 2*epsilon**3))
print(k > (24*np.log(N_T))/(3*epsilon**2 - 2*epsilon**3))

Z0 = np.random.uniform(-1, 1, size=(k,1))

bias = 0.01


# Decided where to put the outputs. You have to change this...

# In[3]:


quality = 1000
# Target_Path_Folder = r"C:\Users\eneam\Dropbox\Research\Thesis\GBM_Signal_Extraction_GBM_GBM_Few_Shit_Student_" + str(mu) + "_" + str(sigma) + "_"  + str(mu_2) + "_" + str(sigma_2) + "_" + str(N_T) + "_" + str(M) + "_" + str(today).replace("-", "_")
Target_Path_Folder = r"C:\Users\eneam\Dropbox\Research\Rough_Paper\Outputs\ICLR\Tumor_Classic_OU_Different_Stimula_" + str(k1) + "_" + str(k2) + "_"  + str(l0)  + "_"  + str(l1) + "_" + str(N_T) + "_" + str(k) + "_" + str(N_S) + "_" + str(N_Channels)
Path(Target_Path_Folder).mkdir(parents=True, exist_ok=True)
path = Path(Target_Path_Folder)


# Now we define some utilities

# In[22]:

    
def Path_Integrator(t, BM, mean, speed, vol):
    dt = t[1] - t[0]
    dB = np.diff(BM)
    SDEpath = np.empty((1,N_T+1))
    SDEpath[:, 0] = 0
        
    for tt in np.arange(1,N_T+1):
        SDEpath[:,tt] = SDEpath[:,tt-1] + speed*(mean-SDEpath[:,tt-1])*dt + vol*dB[tt-1,]
    return SDEpath.reshape((N_T+1,))


def nilpotent(M):
    B = np.zeros((M,M))
    for i in range(2,M):
        B[i,i-1]=1.0
    return B

def canonical(i,M):
    e = np.zeros((M,1))
    e[i,0]=1.0
    return e



# def randomAbeta(d,M):
#     A = []
#     beta = []
#     for i in range(d):
#         # B = 0.0*nilpotent(M) + np.random.standard_t(2,size=(M,M)) 
#         B = np.triu(np.random.uniform(-1, 1, size=(M,M))).astype('float')
#         for i in range(M):
#             B[i,i] = 1
#         # B = 0.0*nilpotent(M) + np.random.normal(0.0,1.0,size=(M,M)) 
#         # B = np.random.permutation(B)
#         A = A + [B]
#         beta = beta + [0.0*canonical(i,M)+np.random.uniform(-1, 1, size=(M,1))]
#         # beta = beta + [0.0*canonical(i,M)+np.random.normal(0.0,1.0,size=(M,1))]
#     return [A,beta]


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
#         beta = beta + [0.0*canonical(i,M)+np.random.uniform(-1, 1, size=(M,1))*0]
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





def sigmoid(x):
    return x/40
    # return np.divide(1, 1 + np.exp(-x))
    #return np.maximum(x/10,0)


def reservoirfield_Y(state,increment, C, deta):
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

    def path(self, k1, k2, l0, l1, p, u):
        
        t = np.arange(0, self.timehorizon + self.timehorizon/self.timesteps, self.timehorizon/self.timesteps)
        dt = t[1] - t[0]
        
        switch_tem = 0
        x1 = np.empty((1,N_T+1))
        x2 = np.empty((1,N_T+1))
        x3 = np.empty((1,N_T+1))
        x4 = np.empty((1,N_T+1))
        w = np.empty((1,N_T+1))
        
        u = u.reshape((1,N_T+1))
        
        x1[0] = 2
        x2[0] = 0
        x3[0] = 0
        x4[0] = 0
        w [0] = x1[0]
        

        for tt in np.arange(1,N_T+1):
            switch_tem = l0*x1[0,tt-1]/((1+((l0/l1)*w[0,tt-1])**p)**(1/p))
            x1[0,tt] = x1[0,tt-1] + dt * (switch_tem - k2 * u[0,tt-1] * x1[0,tt-1] )
            x2[0,tt] = x2[0,tt-1] + dt * ( k2 * u[0,tt-1] * x1[0,tt-1] - k1 * x2[0,tt-1])
            x3[0,tt] = x3[0,tt-1] + dt * k1 * (x2[0,tt-1]-x3[0,tt-1])
            x4[0,tt] = x4[0,tt-1] + dt * k1 * (x3[0,tt-1]-x4[0,tt-1])
            w[0,tt]  = x1[0,tt]+ x2[0,tt] + x3[0,tt] + x4[0,tt]
            



        return [t, u.reshape((N_T+1,)), w.reshape((N_T+1,))]


    
    def reservoir_Y(self,Control_Path, C, deta):
        reservoirpath = [Z0]
        Increment_Storage = np.diff(Control_Path,axis=1)
        for i in range(self.timesteps):
            increment = Increment_Storage[:,i]
            reservoirpath = reservoirpath + [(reservoirpath[-1]+reservoirfield_Y(reservoirpath[-1],increment, C, deta))]
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


# Decleare the RDE Object and plot the Random Signature, jsut to see how they look.

# In[23]:

CDeta = randomAbeta(d,k)
C = CDeta[0]
deta = CDeta[1]

t = np.arange(0, 10 + 10/N_T, 10/N_T)

# Declare the object
OU_RDE = RDE(10,1.0,k,N_T)
dB = np.sqrt(10/N_T) * np.random.randn(N_T)
# u  = 0.25*(np.insert(np.cumsum(dB),0,0))**2
u = Path_Integrator(t, np.insert(np.cumsum(dB),0,0), mean, speed, vol)
Joint_Path = OU_RDE.path(k1, k2, l0, l1, p, u) 

plt.figure()
# We plot
line_up, = plt.plot(Joint_Path[0],Joint_Path[1], color = (0.138, 0.484, 0.782),linewidth=4, label='LTL')
line_down, = plt.plot(Joint_Path[0],Joint_Path[2], color = (0.93, 0.525, 0.219),linewidth=3, linestyle='dashed', label='True')
plt.legend([line_up, line_down], ['u', 'w'],fontsize=15)
plt.title("Out Of Sample",fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Value',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.savefig(path / "Out_of_Sample_Comparison_of_True_vs_Extracted_Y_Levels.pdf", bbox_inches='tight', dpi=quality)
plt.show()



Control_Path = [Joint_Path[0],Joint_Path[1]]
plt.plot(np.squeeze(OU_RDE.reservoir_Y(Control_Path, C, deta)))
plt.savefig(path / "Random_Signature.pdf", dpi=quality)
plt.show()


# In[24]:

with sns.color_palette("colorblind", n_colors=8):
    plt.plot(Joint_Path[0],np.squeeze(OU_RDE.reservoir_Y(Control_Path, C, deta))[:,[0,1,2,3,4,5,6,7]],linewidth=3)
plt.title("Random Signatures (8/111 plotted)",fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Value',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(path / "Random_Signature.pdf", bbox_inches='tight', dpi=quality)
plt.show()


# Generate the Reservoir of Signature and Targets.
# Explaination in the code

# In[24]:

CDeta = randomAbeta(d*N_Channels,k)

C = []
deta = []

for l in range(0,d*N_Channels,2):
    C = C + [[CDeta[0][l],CDeta[0][l+1]]]
    deta = deta + [[CDeta[1][l],CDeta[1][l+1]]]


Y_Reservoir = np.zeros((1,))
Features_Reservoir=np.zeros([N_Channels,N_S,N_T+1,k],dtype=np.float32)


for i in range(N_S):

    if np.mod(i,100)==0:
        print(i)
    dB = np.sqrt(10/N_T) * np.random.randn(N_T)
    # u_train  = 0.25*(np.insert(np.cumsum(dB),0,0))**2
    u_train = Path_Integrator(t, np.insert(np.cumsum(dB),0,0), mean, speed, vol)
    Joint_Path = OU_RDE.path(k1, k2, l0, l1, p, u_train) 
    Control_Path = [Joint_Path[0],Joint_Path[1]]
    
    for l in range(0,N_Channels):
        Features_Reservoir[l,i,:,:] = np.squeeze(OU_RDE.reservoir_Y(Control_Path, C[l], deta[l]))
    
    # Here we save the target: SDEpath
    Y_Reservoir = np.r_[Y_Reservoir, np.log(Joint_Path[2]+bias)]

Y_Reservoir = np.delete(Y_Reservoir, (0), axis=0)
Y_Reservoir = Y_Reservoir.reshape((Y_Reservoir.shape[0],1))

# In[25]:
    
model_list = []

Y_Pred = np.zeros((Y_Reservoir.shape[0],N_Channels))

for l in range(0,N_Channels):
    print(l)
    Features = Features_Reservoir[l,:,:,:].astype(np.float32)
    Features =np.reshape(Features,(-1,k))
    lm_Y = linear_model.Ridge(alpha=0.001)#
    model_Y  = lm_Y.fit(Features,Y_Reservoir)
    
    Y_Pred[:,l] = model_Y.predict(Features).reshape((Y_Reservoir.shape[0],))
    
    model_list = model_list + [model_Y]


# Let us plot an OOS example

# In[26]:
    
Features_Test=np.zeros([N_Channels,1,N_T+1,k])
dB = np.sqrt(10/N_T) * np.random.randn(N_T)
# u_test  = 0.25*(np.insert(np.cumsum(dB),0,0))**2
u_test = Path_Integrator(t, np.insert(np.cumsum(dB),0,0), mean, speed, vol)
Joint_Path_Test = OU_RDE.path(k1, k2, l0, l1, p, u_test)
Control_Path = [Joint_Path_Test[0],Joint_Path_Test[1]]
Y_Pred_Test = np.zeros((Joint_Path_Test[2].shape[0],N_Channels))
    
for l in range(0,N_Channels):
    Y_Pred_Test[:,l] = model_list[l].predict(np.squeeze(OU_RDE.reservoir_Y(Control_Path, C[l], deta[l]))).reshape((Joint_Path_Test[2].shape[0],))
    

Y_Extracted = np.mean(np.exp(Y_Pred_Test)-bias,axis=1)
# Y_Extracted_err = np.std(Y_Pred_Test,axis=1)/np.sqrt(N_Channels)
    

############# NOW WE EXTRACT the FEATURES of the Controls: Time and BM 

# Map through linear layer
import seaborn as sns

plt.figure()
# We plot
line_up, = plt.plot(Joint_Path_Test[0],Y_Extracted, color = (0.138, 0.484, 0.782),linewidth=4, label='LTL')
line_down, = plt.plot(Joint_Path_Test[0],Joint_Path_Test[2], color = (0.93, 0.525, 0.219),linewidth=3, linestyle='dashed', label='True')
line_down2, = plt.plot(Joint_Path_Test[0],Joint_Path_Test[1], 'g',linewidth=2, label='True')

plt.legend([line_up, line_down, line_down2], ['Extracted ' + r'$\hat{Y}_{t}$', 'True ' + r'$Y_{t}$, ',r'$X_{t}$'],fontsize=10)
plt.title("Out Of Sample",fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Value',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(path / "Out_of_Sample_Comparison_Correct_Stimula_OU.pdf", bbox_inches='tight', dpi=quality)
plt.show()

#########################################################


# In[9]:


Features_Test=np.zeros([N_Channels,1,N_T+1,k])
dB = np.sqrt(10/N_T) * np.random.randn(N_T)
# u_test  = 0.25*(np.insert(np.cumsum(dB),0,0))**2
# u_test = Path_Integrator(t, np.insert(np.cumsum(dB),0,0), mean+0.5, speed-0.5, vol+0.5)

u_test = Path_Integrator(t, np.insert(np.cumsum(dB),0,0)**2, mean, speed, vol)

Joint_Path_Test = OU_RDE.path(k1, k2, l0, l1, p, u_test)
Control_Path = [Joint_Path_Test[0],Joint_Path_Test[1]]
Y_Pred_Test = np.zeros((Joint_Path_Test[2].shape[0],N_Channels))
    
for l in range(0,N_Channels):
    Y_Pred_Test[:,l] = model_list[l].predict(np.squeeze(OU_RDE.reservoir_Y(Control_Path, C[l], deta[l]))).reshape((Joint_Path_Test[2].shape[0],))
    

Y_Extracted = np.mean(np.exp(Y_Pred_Test)-bias,axis=1)

# Map through linear layer
import seaborn as sns

plt.figure()
# We plot
line_up, = plt.plot(Joint_Path_Test[0],Y_Extracted, color = (0.138, 0.484, 0.782),linewidth=4, label='LTL')
line_down, = plt.plot(Joint_Path_Test[0],Joint_Path_Test[2], color = (0.93, 0.525, 0.219),linewidth=3, linestyle='dashed', label='True')
line_down2, = plt.plot(Joint_Path_Test[0],Joint_Path_Test[1], 'g',linewidth=2, label='True')

plt.legend([line_up, line_down, line_down2], ['Extracted ' + r'$\hat{Y}_{t}$', 'True ' + r'$Y_{t}$, ',r'$X_{t}$'],fontsize=10)
plt.title("Out Of Sample",fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Value',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(path / "Out_of_Sample_Comparison_Wrong_Stimula_OU.pdf", bbox_inches='tight', dpi=quality)
plt.show()

# Now we extract some statistics from the extracted and real paths and compare them.
# The fact is that for each new OOS path that we check, we know exactly which Y we should extract and we have one that we extract ourselves.
# Therefore, we compare in a fair way because we compare statistics on N_C Extracted Ys with the statistics that we would have observed on the correct Ys.
# This gives an idea of how much fucked up our Extracted Ys are wrt their true counterpaty.

# In[9]:




Reservoir_Y_Extracted = np.zeros((1,N_T+1))
Reservoir_Y_True = np.zeros((1,N_T+1))

for i in range(N_C):
    
    if np.mod(i,100)==0:
        print(i)
    
     
    ############ TEST THE AUTOENCODER #############
    
    Features_Test=np.zeros([N_Channels,1,N_T+1,k])
    
    dB = np.sqrt(10/N_T) * np.random.randn(N_T)
    # u_test  = 0.25*(np.insert(np.cumsum(dB),0,0))**2
    u_test = Path_Integrator(t, np.insert(np.cumsum(dB),0,0), mean, speed, vol)
    Joint_Path_Test = OU_RDE.path(k1, k2, l0, l1, p, u_test) 
    Control_Path = [Joint_Path_Test[0],Joint_Path_Test[1]]
    Y_Pred_Test = np.zeros((Joint_Path_Test[2].shape[0],N_Channels))
        
    for l in range(0,N_Channels):
        Y_Pred_Test[:,l] = np.exp(model_list[l].predict(np.squeeze(OU_RDE.reservoir_Y(Control_Path, C[l], deta[l]))).reshape((Joint_Path_Test[2].shape[0],)))-bias
        
    Y_Test_Extracted = np.mean(Y_Pred_Test,axis=1)
    
    ############### SU PATH CORRETTO
    ############### SU Y Real
    
    Reservoir_Y_Extracted = np.r_[Reservoir_Y_Extracted, Y_Test_Extracted.reshape((1,N_T+1))]   
    Reservoir_Y_True = np.r_[Reservoir_Y_True, Joint_Path_Test[2].reshape((1,N_T+1))]   


print("Fatto")
    
Reservoir_Y_Extracted =  np.delete(Reservoir_Y_Extracted, 0, axis=0)
Reservoir_Y_True = np.delete(Reservoir_Y_True, 0, axis=0)


# In[10]:



Starting_Values_True = Reservoir_Y_True[:,0]
Average_Increments_True = Average_Increment_Calculator(Reservoir_Y_True)
Std_Increments_True = Std_Increment_Calculator(Reservoir_Y_True)
p_value_normality_increments_True = pvalue_normality_Increment_Calculator(Reservoir_Y_True)
Autocorrelation_increments_True = Autocorr_Increment_Calculator(Reservoir_Y_True)
Quadratic_Variation_True = Quadratic_Variation_Calculator(Reservoir_Y_True)



Starting_Values_Extracted  = Reservoir_Y_Extracted [:,0]
Average_Increments_Extracted  = Average_Increment_Calculator(Reservoir_Y_Extracted )
Std_Increments_Extracted  = Std_Increment_Calculator(Reservoir_Y_Extracted )
p_value_normality_increments_Extracted  = pvalue_normality_Increment_Calculator(Reservoir_Y_Extracted )
Autocorrelation_increments_Extracted  = Autocorr_Increment_Calculator(Reservoir_Y_Extracted )
Quadratic_Variation_Extracted  = Quadratic_Variation_Calculator(Reservoir_Y_Extracted )
Tonio_Measure_Extracted = Tonio_Measure_all(Reservoir_Y_True,Reservoir_Y_Extracted)


# Compare starting value

# In[11]:


print("Tonio Measure:")
print(np.mean(Tonio_Measure_Extracted))
print("Tonio Measure:")
print(np.median(Tonio_Measure_Extracted))


# Compare Distribution of Average of the Increments. Interesting is that the average is relevant but... The distribution of the averages is more informative.

# In[12]:



plt.figure()
bins = np.linspace(-0.05, 0.05, np.int(np.sqrt(N_C)))

plt.hist(Average_Increments_True, bins, alpha=0.5, label='True',density=True)
plt.hist(Average_Increments_Extracted, bins, alpha=0.5, label='Extracted',density=True)
plt.legend(loc='upper right')
plt.title("Distribution of Average of Increments")
plt.savefig(path / "Distribution of Average of Increments.pdf", dpi=quality)
plt.show()

tset, pval_avg = ttest_1samp(Average_Increments_Extracted, 0)
print("p-values",pval_avg)

if pval_avg > 0.05:    # alpha value is 0.05 or 5%
    print("Average of the averages of Increments is NOT 0")
else:
  print("Average of the averages of Increments is 0")  

plt.figure()
labels = ('Extracted', 'True') 
data = [Average_Increments_Extracted, Average_Increments_True]
fig7, ax7 = plt.subplots()
ax7.set_title('Average of Increments')
ax7.boxplot(data)
plt.xticks(np.arange(len(labels))+1,labels)
plt.savefig(path / "Boxplot of Average of Increments.pdf", dpi=quality)
plt.show()


# Compare Distribution of Stds of the Increments. Interesting is that the average is relevant but... The distribution of the Stds is more informative.

# In[13]:



plt.figure()
bins = np.linspace(0.8, 1.3, np.int(np.sqrt(N_C)))

plt.hist(Std_Increments_True**2*N_T, bins, alpha=0.5, label='True',density=True)
plt.hist(Std_Increments_Extracted**2*N_T, bins, alpha=0.5, label='Extracted',density=True)
plt.legend(loc='upper right')
plt.title("Distribution of Std of Increments")
plt.savefig(path / "Distribution of Std of Increments.pdf", dpi=quality)
plt.show()

print(np.mean(Std_Increments_True)**2*N_T)
print(np.mean(Std_Increments_Extracted)**2*N_T)

tset, pval_std = ttest_1samp(Std_Increments_Extracted, np.sqrt(1/N_T))
print("p-values",pval_std)

if pval_std > 0.05:    # alpha value is 0.05 or 5%
    print("Average of the Std of Increments is NOT " + str(np.sqrt(1/N_T)))
else:
  print("Average of the Std of Increments is "+ str(np.sqrt(1/N_T)))
  
plt.figure()
labels = ('Extracted', 'True') 
data = [Std_Increments_Extracted, Std_Increments_True]
fig7, ax7 = plt.subplots()
ax7.set_title('Std of Increments')
ax7.boxplot(data)
plt.xticks(np.arange(len(labels))+1,labels)
plt.savefig(path / "Boxplot of Std of Increments.pdf", dpi=quality)
plt.show()


# Compare Distribution of pvalues. Interesting is that the average is relevant but... The distribution of the pvalues is more informative.

# In[14]:



plt.figure()
bins = np.linspace(0, 1, np.int(np.sqrt(N_C)))

plt.hist(p_value_normality_increments_True, bins, alpha=0.5, label='True',density=True)
plt.hist(p_value_normality_increments_Extracted, bins, alpha=0.5, label='Extracted',density=True)
plt.legend(loc='upper right')
plt.title("Distribution of Pvalue of Normality of Increments")
plt.savefig(path / "Distribution of Pvalue of Normality of Increments.pdf", dpi=quality)
plt.show()

print(np.mean(p_value_normality_increments_True))
print(np.mean(p_value_normality_increments_Extracted))

plt.figure()
labels = ('Extracted', 'True') 
data = [p_value_normality_increments_Extracted, p_value_normality_increments_True]
fig7, ax7 = plt.subplots()
ax7.set_title('Pvalue of Normality of Increments')
ax7.boxplot(data)
plt.xticks(np.arange(len(labels))+1,labels)
plt.savefig(path / "Boxplot of Pvalue of Normality of Increments.pdf", dpi=quality)
plt.show()


# Compare Distribution of Autocorrelation of the Increments. Interesting is that the average is relevant but... The distribution of the Autocorrelation is more informative.

# In[15]:



bins = np.linspace(-0.2, 0.2, np.int(np.sqrt(N_C)))
plt.figure()
plt.hist(Autocorrelation_increments_True, bins, alpha=0.5, label='True',density=True)
plt.hist(Autocorrelation_increments_Extracted, bins, alpha=0.5, label='Extracted',density=True)
plt.legend(loc='upper right')
plt.title("Distribution of Autocorrelation of Increments")
plt.savefig(path / "Distribution of Autocorrelations of Increments.pdf", dpi=quality)
plt.show()

print(np.mean(Autocorrelation_increments_True))
print(np.mean(Autocorrelation_increments_Extracted))


tset, pval_aut = ttest_1samp(Autocorrelation_increments_True,0)
print("p-values",pval_aut)

if pval_aut > 0.05:    # alpha value is 0.05 or 5%
    print("Average of the Autocorrelations of Increments is NOT 0")
else:
  print("Average of the Autocorrelations of Increments is 0")
  
plt.figure()
labels = ('Extracted', 'True') 
data = [Autocorrelation_increments_Extracted, Autocorrelation_increments_True]
fig7, ax7 = plt.subplots()
ax7.set_title('Autocorrelations of Increments')
ax7.boxplot(data)
plt.xticks(np.arange(len(labels))+1,labels)
plt.savefig(path / "Boxplot of Autocorrelations of Increments.pdf", dpi=quality)
plt.show()
 


# Compare Distribution of Quadratic Variation of the Increments. Interesting is that the average is relevant but... The distribution of the Quadratic Variation is more informative.

# In[16]:



plt.figure()
bins = np.linspace(0.8, 1.3, np.int(np.sqrt(N_C)))
plt.figure()
plt.hist(Quadratic_Variation_True, bins, alpha=0.5, label='True',density=True)
plt.hist(Quadratic_Variation_Extracted, bins, alpha=0.5, label='Extracted',density=True)
plt.legend(loc='upper right')
plt.title("Distribution of Second Variation of BMs")
plt.savefig(path / "Distribution of Second Variation of BMs.pdf", dpi=quality)
plt.show()

print(np.mean(Quadratic_Variation_True))
print(np.mean(Quadratic_Variation_Extracted))

tset, pval_qv = ttest_1samp(Autocorrelation_increments_True,1)
print("p-values",pval_qv)

if pval_qv > 0.05:    # alpha value is 0.05 or 5%
    print("Average of the Second Variation of BMs is NOT 1")
else:
  print("Average of the Second Variation of BMs is 1")

plt.figure()
labels = ('Extracted', 'True') 
data = [Quadratic_Variation_Extracted, Quadratic_Variation_True]
fig7, ax7 = plt.subplots()
ax7.set_title('Second Variation of BMs')
ax7.boxplot(data)
plt.xticks(np.arange(len(labels))+1,labels)
plt.savefig(path / "Boxplot of Second Variation of BMs.pdf", dpi=quality)
plt.show()




# In[17]:



import sys

print('This message will be displayed on the screen.')

original_stdout = sys.stdout # Save a reference to the original standard output

with open(path /'filename.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    
    print("Mean of Starting_Values_True:")
    print(np.mean(Starting_Values_True))
    print("Mean of Starting_Values_Extracted:")
    print(np.mean(Starting_Values_Extracted))
    print()
    

    
    print("Mean of Tonio Measure:")
    print(np.mean(Tonio_Measure_Extracted))
    print()
    
    
    print("Median of Tonio Measure:")
    print(np.median(Tonio_Measure_Extracted))
    print()
    
    
    print("Mean of Average_Increments_True:")
    print(np.mean(Average_Increments_True))
    print("Mean of Average_Increments_Extracted:")
    print(np.mean(Average_Increments_Extracted))
    
    tset, pval_avg = ttest_1samp(Average_Increments_Extracted, 0)
    print("p-values",pval_avg)
    
    if pval_avg > 0.05:    # alpha value is 0.05 or 5%
        print("Average of the averages of Increments is NOT 0")
    else:
      print("Average of the averages of Increments is 0")  
        
    print()
    
    
    
    print("Mean of Std_Increments_True:")
    print(np.mean(Std_Increments_True))
    print("Mean of Std_Increments_Extracted:")
    print(np.mean(Std_Increments_Extracted))
    
    tset, pval_std = ttest_1samp(Std_Increments_Extracted, np.sqrt(1/N_T))
    print("p-values",pval_std)
    
    if pval_std > 0.05:    # alpha value is 0.05 or 5%
        print("Average of the Std of Increments is NOT " + str(np.sqrt(1/N_T)))
    else:
      print("Average of the Std of Increments is "+ str(np.sqrt(1/N_T)))
    
    
    print()
    
    
    print("Mean of p_value_normality_increments_True:")
    print(np.mean(p_value_normality_increments_True))
    print("Mean of p_value_normality_increments_Extracted:")
    print(np.mean(p_value_normality_increments_Extracted))
    print()
    
    
    print("Mean of Autocorrelation_increments_True:")
    print(np.mean(Autocorrelation_increments_True))
    print("Mean of Autocorrelation_increments_Extracted:")
    print(np.mean(Autocorrelation_increments_Extracted))
    
    tset, pval_aut = ttest_1samp(Autocorrelation_increments_True,0)
    print("p-values",pval_aut)
    
    if pval_aut > 0.05:    # alpha value is 0.05 or 5%
        print("Average of the Autocorrelations of Increments is NOT 0")
    else:
      print("Average of the Autocorrelations of Increments is 0")
    
    print()
    
    
    print("Mean of Quadratic_Variation_True:")
    print(np.mean(Quadratic_Variation_True))
    print("Mean of Quadratic_Variation_Extracted:")
    print(np.mean(Quadratic_Variation_Extracted))
    
    tset, pval_qv = ttest_1samp(Quadratic_Variation_Extracted,1)
    print("p-values",pval_qv)
    
    if pval_qv > 0.05:    # alpha value is 0.05 or 5%
        print("Average of the Second Variation of BMs is NOT 1")
    else:
      print("Average of the Second Variation of BMs is 1")
    
    
    
    sys.stdout = original_stdout # Reset the standard output to its original value

