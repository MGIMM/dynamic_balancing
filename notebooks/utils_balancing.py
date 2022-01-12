import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


# GIPWE
class Shallow_GIPW:
    def __init__(self, xi, xi_ring, w =None, w_ring = None, epsilon = 1e-8, test_split_ratio = 0.2, bootstrap_multiplier = 2):
        # parameter used to avoid division by 0
        self.epsilon = epsilon
        # import support of source and target measures
        self.xi = xi
        self.xi_ = xi # backup
        self.xi_ring = xi_ring
        
        self.n = self.xi.shape[0]
        self.m = self.xi_ring.shape[0]
        # backup
        self.n_ = self.xi.shape[0]
        self.m_ = self.xi_ring.shape[0]
        # init weights
        if w is not None:
            self.w_bootstrap_indicator = True
            self.w = w
        else:
            self.w = np.ones(self.n)
            self.w_bootstrap_indicator = False
        if w_ring is not None:
            self.w_ring_bootstrap_indicator = True
            self.w_ring = w_ring
        else:
            self.w_ring = np.ones(self.m)
            self.w_ring_bootstrap_indicator = False
        ## setup artificial supervised learning problem
        if self.w_bootstrap_indicator:
            self.xi = self.xi[np.random.choice(self.n, size = int(self.n*bootstrap_multiplier), p = self.w/np.sum(self.w))]
            self.n *= bootstrap_multiplier
        if self.w_ring_bootstrap_indicator:
            self.xi_ring = self.xi_ring[np.random.choice(self.m, size = int(self.m*bootstrap_multiplier), p = self.w_ring/np.sum(self.w_ring))]
            self.m *= bootstrap_multiplier
        X = np.concatenate((self.xi,self.xi_ring), axis = 0)
        Y = np.concatenate((np.zeros(self.n), np.ones(self.m)), axis = 0)
        # apply a random permutation
        random_indices = np.random.choice(self.n+self.m, self.n+self.m, replace = False)
        X = X[random_indices]
        Y = Y[random_indices]
        # train/test splitting
        self.Xtrain = X[random_indices[0:int((self.n+self.m)*test_split_ratio)],:]
        self.Ytrain = Y[random_indices[0:int((self.n+self.m)*test_split_ratio)]]
        self.Xtest = X[random_indices[int((self.n+self.m)*test_split_ratio)+1:],:]
        self.Ytest = Y[random_indices[int((self.n+self.m)*test_split_ratio)+1:]]
        
    def train(self,model,xi, log = False):
        model.fit(self.Xtrain, self.Ytrain.reshape((self.Xtrain.shape[0],)))
        test_loss = np.mean((model.predict(self.Xtest) - self.Ytest)**2)
        train_loss = np.mean((model.predict(self.Xtrain) - self.Ytrain)**2)
        if log:
            print("train loss:", train_loss)
            print("test loss:", test_loss)
        try:
            e = model.predict_proba(xi)[:,1] 
        except:
            e = model.predict(xi) 
            
        self.weights = e/(1.-e+self.epsilon) *self.n_/self.m_
        
        
        
    
    
class OptimalTransportBalancing():
    def __init__(self, metric = None):
        """
        Attributes
        ----------
        metric: function (x,y)
            Compute distance between two elements x and y, that are both assumed to be tensors.
        """
        if metric != None:
            self.metric = metric
        else:
            self.metric = lambda x,y : torch.linalg.norm(x-y)
    def get_nearest_neighbor_index (self, x,Y):
        self.n = Y.shape[0]
        distance_list = torch.zeros(self.n)
        for i in range(self.n):
            distance_list[i] = self.metric(x, Y[i])
        return torch.argmin(distance_list).item() 
    def get_weights(self, source, target, source_weights = None, target_weights = None):
        self.n = len(source)
        self.m = len(target)
        if source_weights is not None:
            self.w = source_weights.reshape((self.n,))
        else:
            self.w = torch.ones(self.n)
        if target_weights is not None:
            self.w_ring = target_weights.reshape((self.m,))
        else:
            self.w_ring = torch.ones(self.m)
        sum_nn_w_ring = torch.zeros(self.n)
        nearest_neighbor_index = torch.zeros(self.m, dtype = int)
        for j in range(self.m):
            nearest_neighbor_index[j] = self.get_nearest_neighbor_index(target[j], source)
        for i in range(self.n):
            for j in range(self.m):
                if int(nearest_neighbor_index[j].item()) == i:
                    sum_nn_w_ring[i] += self.w_ring[j]
        eta_ring = self.w * sum_nn_w_ring
        return eta_ring *self.n/self.m
    
class MMDBalancing():
    def __init__(self,
                 xi, xi_ring,
                 w = None, w_ring = None,
                 sigma = 1, D = 500,
                 k = None, KXX = None, KXY = None, KYY = None,
                 dev = "cuda:0"):
        # init device
        self.dev = dev
        # init support of source and target measures
        self.xi = xi.to(self.dev) 
        self.xi_ring = xi_ring.to(self.dev)
        self.n,self.d = xi.shape
        self.m = xi_ring.shape[0]
        # init weights of source and target measures
        if w is not None:
            self.w = w.to(self.dev)
        else:
            self.w = torch.ones(self.n, device =self.dev)
        if w_ring is not None:
            self.w_ring = w_ring.to(self.dev)
        else:
            self.w_ring = torch.ones(self.m, device =self.dev)
        # init kernel matrices, when not provided, Gaussian kernel with random Fourier features is implemented
        if (KXX and KXY and KYY) is not None:
            self.KXX,self.KXY,self.KYY = KXX,KXY,KYY
        else:
            W = torch.normal(0,sigma,(D,self.d), device = self.dev)
            theta = torch.rand(D, device  = self.dev)*2*torch.pi
            Phi_X = torch.zeros((self.n,D), device = self.dev)
            Phi_Y = torch.zeros((self.m,D), device = self.dev)
            for j in range(D):
                Phi_X[:,j] = torch.cos(torch.matmul(W[j,:].T,self.xi.T) + theta[j])
                Phi_Y[:,j] = torch.cos(torch.matmul(W[j,:].T,self.xi_ring.T) + theta[j])
            
            self.KXX = torch.matmul(Phi_X,Phi_X.T)
            self.KYY = torch.matmul(Phi_Y,Phi_Y.T)
            self.KXY = torch.matmul(Phi_X,Phi_Y.T)
        
        self.alpha = torch.rand(self.n,device = self.dev,requires_grad=True)
            
        
        
        
        
    def get_weights(self, 
           lambda_l2 = 1e-1,
           lambda_RKHS = 1e2,
           ):
                
        #self.weights = torch.matmul(self.KXX,self.alpha)
        
        
      #direct solves
        y = self.w_ring.type(torch.cdouble)
        KXX = self.KXX.type(torch.cdouble)
        KXX_2 = torch.matmul(KXX.T,KXX)
        KXY = self.KXY.type(torch.cdouble)
        self.alpha = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(KXX.T, KXX)+lambda_RKHS*KXX + lambda_l2*KXX_2),KXY),y)
        self.weights = torch.real(torch.matmul(KXX,self.alpha))
        return self.weights
    
dev = "cuda:0"    
class Deep_GIPW():
    def __init__(self, xi, xi_ring, w = None, w_ring = None, epsilon = 1e-8, dev = dev, test_split_ratio = 0.2, bootstrap_multiplier = 2):
        self.epsilon = epsilon
        self.dev = dev
        with torch.no_grad():
            self.xi = xi.clone().to(self.dev)
            self.xi_ring = xi_ring.clone().to(self.dev)
            # backup
            self.xi_ = xi.clone().to(self.dev)
            self.xi_ring_ = xi_ring.clone().to(self.dev)
            if w is not None:
                self.w = w.clone().to(self.dev)
            else:
                self.w = torch.ones(len(xi)).clone().to(self.dev)
            if w_ring is not None:
                self.w_ring = w_ring.clone().to(self.dev)
            else:
                self.w_ring = torch.ones(len(xi_ring)).to(self.dev)
        self.n = len(xi)
        self.m = len(xi_ring)
        # backup
        self.n_ = len(xi)
        self.m_ = len(xi_ring)
        
        if w is not None:
            self.w_bootstrap_indicator = True
            self.w = w.to(self.dev)
        else:
            self.w = torch.ones(self.n).to(self.dev)
            self.w_bootstrap_indicator = False
        if w_ring is not None:
            self.w_ring_bootstrap_indicator = True
            self.w_ring = w_ring.to(self.dev)
        else:
            self.w_ring = torch.ones(self.m).to(self.dev)
            self.w_ring_bootstrap_indicator = False
        ## setup artificial supervised learning problem
        if self.w_bootstrap_indicator:
            rand_ind = torch.multinomial(self.w,  int(self.n*bootstrap_multiplier), replacement = True)
            self.xi = self.xi[rand_ind]
            self.n *= bootstrap_multiplier
        if self.w_ring_bootstrap_indicator:
            rand_ind = torch.multinomial(self.w_ring,  int(self.m*bootstrap_multiplier), replacement = True)
            self.xi_ring = self.xi_ring[rand_ind]
            self.m *= bootstrap_multiplier
        X = torch.cat((self.xi,self.xi_ring), axis = 0)
        Y = torch.cat((torch.zeros((self.n,1)), torch.ones((self.m,1))), axis = 0)
        # apply a random permutation
        random_indices = np.random.choice(self.n+self.m, self.n+self.m, replace = False)
        X = X[random_indices]
        Y = Y[random_indices]
        Y = Y.to(self.dev)
        # train/test splitting
        self.Xtrain = X[random_indices[0:int((self.n+self.m)*test_split_ratio)],:]
        self.Ytrain = Y[random_indices[0:int((self.n+self.m)*test_split_ratio)]]
        self.Xtest = X[random_indices[int((self.n+self.m)*test_split_ratio)+1:],:]
        self.Ytest = Y[random_indices[int((self.n+self.m)*test_split_ratio)+1:]]
    
    def train(self,
              model,
              optimizer,
              batch_size = 64,
              epochs = 500):
        loss_fn = nn.MSELoss()
        dataset_train = TensorDataset(self.Xtrain,self.Ytrain.type(torch.float))
        dataset_test = TensorDataset(self.Xtest,self.Ytest.type(torch.float))
        train_dataloader = DataLoader(dataset_train, batch_size=batch_size)
        test_dataloader = DataLoader(dataset_test, batch_size=batch_size)
        def train_loop():
            size = len(train_dataloader.dataset)
            for batch, (X, y) in enumerate(train_dataloader):
                # Compute prediction and loss
                pred = model(X)
                loss = loss_fn(pred, y)
        
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            return loss


        def test_loop():
            size = len(test_dataloader.dataset)
            num_batches = len(test_dataloader)
            #test_loss, correct = 0, 0
            test_loss = 0
        
            with torch.no_grad():
                for X, y in test_dataloader:
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
            test_loss /= num_batches
            return test_loss
        test_loss_trace = []
        train_loss_trace = []
        for t in range(epochs):
            current_train_loss = train_loop()
            current_test_loss = test_loop()
            test_loss_trace.append(current_test_loss)
            train_loss_trace.append(current_train_loss.to("cpu").detach().numpy())
        return train_loss_trace, test_loss_trace
        
class Adversarial_Balancing():
    def __init__(self,
                 source_sample,
                 target_sample,
                 source_weight = None,
                 target_weight = None):
        """
        Attributes
        ----------
        source_sample: tensor
        Support of the source measure.
        
        target_sample: tensor
        target of the source measure.
        
        source_weight: tensor/np.array or None
        Weights of the source weighted empirical measure, i.e., 
        source measure = \frac{1}{n} \sum_{i=1}^{n} source_weight[i] \delta_{source_sample[i]}. 
        When not assigned, the weight 1 is being implemented.
        
        target_weight: tensor/np.array or None
        Weights of the target weighted empirical measure, i.e., 
        $$
        target measure = \frac{1}{m} \sum_{j=1}^{m} target_weight[j] \delta_{target_sample[j]}. 
        $$
        When not assigned, the weight 1 is being implemented.
        """
        # use GPU when possible
        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:  
            dev = "cpu"  
        self.dev = torch.device(dev)
        # xi denotes the source measure
        # xi_ring denotes the target measure
        with torch.no_grad():
            self.xi = source_sample.clone().to(self.dev)
            self.xi_ring = target_sample.clone().to(self.dev)
            if source_weight:
                self.w = source_weight.clone().to(self.dev)
            else:
                self.w = torch.ones(len(source_sample)).clone().to(self.dev)
            if target_weight:
                self.w_ring = target_weight.clone().to(self.dev)
            else:
                self.w_ring = torch.ones(len(target_sample)).to(self.dev)
        self.n  = len(self.xi)
                
  
            
    def train_loop(self,
                   model_IPM,
                  model_reweighting,
                  optimizer_IPM,
                  optimizer_reweighting,
                  IPM_steps = 1,
                  reweight_steps = 1,
                  lambda_l2_weight = 1e-1,
                  lambda_l2_IPM = 1e-1,
                  lambda_l1_IPM = 1e-1,
                  ):
        for t in range(IPM_steps):
            with torch.no_grad():
                weights = model_reweighting(self.xi).clone()
            weights.to(self.dev)
            mean_source = torch.mean(model_IPM(self.xi)*weights*self.w)
            #mean_source = torch.mean(model_IPM(self.xi)*weights/weights.sum()*self.w)
            mean_target = torch.mean(model_IPM(self.xi_ring)*self.w_ring)
            loss_IPM = -torch.abs(mean_source - mean_target)
            # # l2-regularization
            # lambda_l2_IPM = 1e2
            # for p in model_IPM.parameters():
            #     l2 += p.square().sum()
            #loss_IPM += lambda_l2_IPM * model_IPM(self.xi).square().mean() 
            loss_IPM_reg = loss_IPM + lambda_l2_IPM * (model_IPM(self.xi).square()*self.w).mean() 
            loss_IPM_reg +=  lambda_l1_IPM*(model_IPM(self.xi_ring).square()*self.w_ring).mean()
            # Backpropagation
            optimizer_IPM.zero_grad()
            loss_IPM_reg.backward()
            optimizer_IPM.step() 
            
            #shaker = ParameterShaker(self.n)
            #model_IPM.apply(shaker)
        # optimization for weight function estimation    
        
        for t in range(reweight_steps):
            with torch.no_grad():
                mean_target_ = torch.mean(model_IPM(self.xi_ring)*self.w_ring).clone()
                values = model_IPM(self.xi).clone()
            mean_source_ =  torch.mean(values*model_reweighting(self.xi)*self.w)
            #mean_source_ =  torch.mean(values*model_reweighting(self.xi)/model_reweighting(self.xi).sum()*self.w)
            loss_reweighting = torch.abs(mean_source_ - mean_target_)
            
            #loss_reweighting_reg = loss_reweighting + lambda_l2_weight * model_reweighting(self.xi).square().mean() 
            loss_reweighting_reg = loss_reweighting + lambda_l2_weight * (model_reweighting(self.xi).square()*self.w).mean() 
            
            optimizer_reweighting.zero_grad()
            loss_reweighting_reg.backward()
            optimizer_reweighting.step() 
            #clipper = ParameterClipper()
            #model_reweighting.apply(clipper)
            
            #shaker = ParameterShaker(self.n)
            #model_reweighting.apply(shaker)
        return loss_reweighting
    
   
    
    
            

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


def get_X_natural(X,X_ring_0, B = 3):
    T = len(X) -1 
    A_ring = torch.zeros((T,m*B,1))
    X_ring = X.clone()
    _X_ring = X.clone()
    
    # covariate shifts
    _X_ring[0] = X_ring_0.clone()
    X_ring[0] = X_ring_0.clone()
    
    for b in range(B-1):
        X_ring = torch.cat((X_ring,_X_ring), dim = 1)
    
        
    for t in range(T):
        A_ring[t] = pi_ring(X_ring[t],torch.tensor([t]))+ torch.normal(0,0.3,(m*B,1))
    X_ring_natural = torch.cat((X_ring[0:-1], A_ring),dim = 2)
    return X_ring_natural,X_ring,A_ring

