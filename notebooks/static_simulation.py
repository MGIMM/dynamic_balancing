#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import matplotlib.pyplot as plt
# from MMDBalancing import MMDBalancing as MMDB
# from OptimalTransportBalancing import OptimalTransportBalancing as OTB
# from NeuralAdversarialBalancing import NeuralAdversarialBalancing as NAB
#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# utils
from utils_balancing import *


# In[2]:


def static_simulation():
    n = 5000
    m = 5000
    d = 1
    r = lambda x:(x-3).square() + (x>-2)*(x+3).square() +x.abs()
    #r = lambda x:x.square()
    def get_data(n = 500,m = 500, r = r, d = d):
        def pi(x):
            return torch.sin(x)+ 2*torch.rand(x.shape)-1
        def pi_ring(x):
            return torch.sin(x)+ 1*torch.rand(x.shape)-0.5
        
        
        xi = torch.normal(mean = -1, std = 2, size = (n,d))
        xi_ring = torch.zeros(size = (m,d))
        for i in range(m):
            if torch.rand(1).item()>0.3:
                xi_ring[i,0] = torch.normal(mean = -4, std = 2, size = (1,)).item()
            else:
                xi_ring[i,0] = torch.normal(mean = 3, std = 0.2, size = (1,)).item()
        w = torch.ones(n)
        w_ring = torch.ones(m)
        
        
        
        
        xi_natural = torch.cat((xi, pi(xi)),axis = 1)
        xi_ring_natural = torch.cat((xi_ring, pi_ring(xi_ring)), axis = 1)
        Z =xi_natural[:,0]+xi_natural[:,1] + torch.rand((n,)) 
        Z_ring =xi_ring_natural[:,0]+xi_ring_natural[:,1]+torch.rand((m,))
        R = r(Z)
        return xi_natural,xi_ring_natural,R,Z,Z_ring
    
    # ## Reference value
    
    # In[7]:
    
    
    xi_natural, xi_ring_natural,R,Z,Z_ring = get_data(n = 50000, m = 50000)
    ref = r(Z_ring).mean()
    
    
    # ### Re-generate data set with $n=m=500$.
    
    # In[8]:
    
    
    n = 500
    m = 500
    xi_natural, xi_ring_natural,R,Z,Z_ring = get_data(n = n, m = m, r = r)
    
    
    # # GIPWE: DE and DRE
    # 
    # 1. Data splitting (K-folds with K = 3)
    
    # In[9]:
    
    
    def get_split_ind(n,K = 3):
        I_n = torch.arange(n, dtype = float)
        
        rand_ind_n = torch.multinomial(I_n,len(I_n),replacement = False)
        num_folds_n = int(n/K)
        Ind = []
        for i in range(K):
            if (i+1)*num_folds_n <= n:
                Ind.append(list(rand_ind_n[i*num_folds_n:(i+1)*num_folds_n].detach().numpy()))
            else:
                Ind.append(list(rand_ind_n[i*num_folds_n:].detach().numpy()))
        
        Ind_split = []
        for i in range(K):
            list_n = []
            for j in range(n):
                if j >= i*num_folds_n and j < (i+1)*num_folds_n:
                    pass
                else:
                    list_n.append(rand_ind_n[j].item())
                
            Ind_split.append(list_n)
        return Ind_split,Ind
    
    
    # In[10]:
    
    
    K = 3
    Ind_out, Ind_in = get_split_ind(n,K)
    
    
    # 2. Get GIPW weights
    
    # In[11]:
    
    
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    
    
    # In[12]:
    
    
    XGB = xgb.XGBRegressor(gamma = 5e0)
    RF = RandomForestRegressor(n_estimators = 20, min_samples_split = 20)
    LR = LogisticRegression()
    def get_GIPW_weights(model):
        eta = np.zeros(n)
        for k in range(K):
            SGIPW = Shallow_GIPW(xi_natural[Ind_out[k],:], xi_ring_natural)
            
            SGIPW.train(model,xi = np.array(xi_natural[Ind_in[k],:]),log=False)
            eta[Ind_in[k]] = SGIPW.weights*(SGIPW.weights>0)
        return eta
    
    eta_XGB = get_GIPW_weights(XGB)
    eta_RF = get_GIPW_weights(RF)
    eta_LR = get_GIPW_weights(LR)
    
    
    # In[13]:
    
    # OT
    OTB = OptimalTransportBalancing()
    eta_OT = OTB.get_weights(xi_natural,xi_ring_natural)
    eta_OT = eta_OT.detach().numpy()
    
    
    # In[17]:
    
    
    # MMD weights
    lambda_RKHS = 1e2
    lambda_l2 = 1e-3
    MMDB = MMDBalancing(xi_natural,xi_ring_natural,sigma = 5e-1,D = 2000)
    eta_MMD = MMDB.get_weights(lambda_RKHS = lambda_RKHS, lambda_l2 = lambda_l2)
    eta_MMD = eta_MMD.to("cpu").detach().numpy()
    
    
    # In[18]:
    
    
    
    
    
    # In[20]:
    
    
    # Neural Adversarial Balancing
    class NeuralNetwork(nn.Module):
        def __init__(self,input_dim = 1, num_nodes = 32):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_dim, num_nodes),
                nn.ReLU(),
                #nn.Dropout(0.3),
                #nn.BatchNorm1d(num_nodes), 
                
                nn.Linear(num_nodes, num_nodes),
                nn.ReLU(),
                nn.Linear(num_nodes, num_nodes),
                nn.ReLU(),
                #nn.Dropout(0.3),
                #nn.BatchNorm1d(num_nodes), 
                
                #nn.Linear(num_nodes, num_nodes),
                #nn.ReLU(),
                # # #nn.Dropout(0.3),
                # # nn.BatchNorm1d(num_nodes), 
                
                nn.Linear(num_nodes, 1),
            )
    
        def forward(self, x):
            x = self.flatten(x)
            target = self.linear_relu_stack(x)
            return target
    
    
    # In[21]:
    
    
    AB = Adversarial_Balancing(xi_natural,xi_ring_natural)
    num_nodes_IPM = 24
    model_IPM = NeuralNetwork(input_dim = d*2,num_nodes = 2*num_nodes_IPM).to(AB.dev)
    model_reweighting = NeuralNetwork(input_dim = d*2, num_nodes = num_nodes_IPM).to(AB.dev)
    learning_rate = 1e-3
    optimizer_IPM = torch.optim.Adam(model_IPM.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
    optimizer_reweighting = torch.optim.Adam(model_reweighting.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
    
    
    # In[22]:
    
    
    epochs = 50
    loss_trace = []
    for t in range(epochs):
        #print(f"Epoch {t+1}\n-------------------------------")
        current_test_loss = AB.train_loop(model_IPM = model_IPM,
                                          model_reweighting = model_reweighting,
                                          optimizer_IPM = optimizer_IPM,
                                          optimizer_reweighting = optimizer_reweighting,
                                          IPM_steps = 3,
                                          reweight_steps = 3,
                                          lambda_l2_weight = 5e-2,
                                          lambda_l2_IPM = 1e-2,
                                          lambda_l1_IPM = 1e-2,
                                         )
        loss_trace.append(current_test_loss.to("cpu").detach().numpy())
    
    
    
    
    weights = model_reweighting(xi_natural.to("cuda:0"))
    #weights /=weights.mean()
    eta_NAB = weights.to("cpu").detach().numpy()
    
    
    
    
    
    # 4. Get $r^{\natural}$ estimation with the same K-fold splitting
    
    # In[26]:
    
    
    from sklearn.linear_model import LinearRegression
    RF_R = RandomForestRegressor(n_estimators = 20, min_samples_split = 5)
    #model_r = RF_R
    model_r = LinearRegression()
    
    
    # In[27]:
    
    
    def get_r_estimation(model, K = 3):
        r_hat = np.zeros(n)
        r_hat_ring = np.zeros(m)
        for k in range(K):
            SGIPW = Shallow_GIPW(xi_natural[Ind_out[k],:], xi_ring_natural)
            model_k = model
            model_k.fit(xi_natural[Ind_out[k],:].detach().numpy(), R[Ind_out[k]].detach().numpy())
            
            r_hat[Ind_in[k]] = model_k.predict(xi_natural[Ind_in[k]].detach().numpy())
            r_hat_ring += model_k.predict(xi_ring_natural.detach().numpy())
        r_hat_ring /= K
            
        return r_hat, r_hat_ring
    
    
    # In[28]:
    
    
    r_hat,r_hat_ring = get_r_estimation(model_r)
    
    
    # In[29]:
    
    
    
    
    # ## Estimators
    
    # In[30]:
    
    
    def get_DE(eta, R = R, ref= ref):
        try:
            eta = torch.from_numpy(eta)
        except:
            pass
        pred = (eta*R).mean().item()
        error  = torch.abs(pred - ref).item()
        return pred, error 
    def get_DRE(eta,r_hat, r_hat_ring, R = R, ref = ref):
        try:
            eta = torch.from_numpy(eta)
            r_hat = torch.from_numpy(r_hat)
        except:
            pass
        pred = (eta*(R -r_hat)).mean() + r_hat_ring.mean()
        error  = torch.abs(pred - ref).item()
        return pred.item(), error 
        
        
        
        
    
    
    # In[31]:
    
    
    #pd.set_option("display.precision", 2)
    #pd.set_option('display.float_format', lambda x: '%.2f' % x)
    table_bad_reg = pd.DataFrame([[get_DE(eta_OT)[1],get_DRE(eta_OT,r_hat,r_hat_ring)[1]],[get_DE(eta_MMD)[1],get_DRE(eta_MMD,r_hat,r_hat_ring)[1]],                          [get_DE(eta_NAB)[1],get_DRE(eta_NAB,r_hat,r_hat_ring)[1]],                         [get_DE(eta_RF)[1],get_DRE(eta_RF,r_hat,r_hat_ring)[1]],[get_DE(eta_XGB)[1],get_DRE(eta_XGB,r_hat,r_hat_ring)[1]],                          [get_DE(eta_LR)[1],get_DRE(eta_LR,r_hat,r_hat_ring)[1]],[None, torch.abs(r_hat_ring.mean()-ref).item()]],                        columns = ("DE","DRE"), index = ("OT", "MMD","NAB", "GIPW-RF","GIPW-XGB","GIPW-LR","G-computation"))
    
    
    # ## Bad regression model: Linear regression
    
    # In[32]:
    
    
    
    
    # In[ ]:
    
    
    
    
    
    # ## Good regression model: XGBoosting
    
    # In[33]:
    
    
    XGB_R = xgb.XGBRegressor(n_estimators = 20, gamma = 1e-0)
    model_r = XGB_R
    r_hat,r_hat_ring = get_r_estimation(model_r)
    
    
    # In[34]:
    
    
    pd.set_option("display.precision", 2)
    table_good_reg = pd.DataFrame([[get_DE(eta_OT)[1],get_DRE(eta_OT,r_hat,r_hat_ring)[1]],[get_DE(eta_MMD)[1],get_DRE(eta_MMD,r_hat,r_hat_ring)[1]],                          [get_DE(eta_NAB)[1],get_DRE(eta_NAB,r_hat,r_hat_ring)[1]],                         [get_DE(eta_RF)[1],get_DRE(eta_RF,r_hat,r_hat_ring)[1]],[get_DE(eta_XGB)[1],get_DRE(eta_XGB,r_hat,r_hat_ring)[1]],                          [get_DE(eta_LR)[1],get_DRE(eta_LR,r_hat,r_hat_ring)[1]],[None, torch.abs(r_hat_ring.mean()-ref).item()]],                        columns = ("DE","DRE"), index = ("OT", "MMD","NAB", "GIPW-RF","GIPW-XGB","GIPW-LR","G-computation"))
    
    
    # In[35]:
    
    
    return table_bad_reg, table_good_reg


