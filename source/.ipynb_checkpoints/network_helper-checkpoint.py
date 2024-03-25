import numpy as np
import matplotlib.pyplot as plt
import sympy
import pickle
import torch
import torch.nn as nn
import time
import sympy as sy
from knw_helper import coeff

# Base neural network class
class Net(nn.Module):
    def __init__(self, layers, body, act):
        super(Net, self).__init__()
        self.act = act; self.body = body
        self.fc = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(self.fc[-1].weight)
    
    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            if self.act == 'tanh':
                x = torch.tanh(x)
            elif self.act == 'sin':
                x = torch.sin(x)
        x = self.fc[-1](x)
        if self.body == 1: # If using net class as multi-headed body, perform final activation function as this is not the final output but represents the basis
            if self.act == 'tanh':
                x = torch.tanh(x)
            elif self.act == 'sin':
                x = torch.sin(x)
        return x
    
class MH_PINN:
    def __init__(self, xf, ff, xb, body_layers, head_layers, verbose, epoch, number_of_cases, act, PDE):
        self.epoch = epoch; self.act = act; self.PDE = PDE
        self.verbose = verbose
        self.number_of_cases = number_of_cases
        
        if self.PDE == 'poisson': 
            # boundary points --- training
            self.xb = torch.unsqueeze(torch.tensor(xb, dtype=torch.float32),-1)
            # collocation points --- residual
            self.xf = torch.unsqueeze(torch.tensor(xf, dtype=torch.float32, requires_grad=True),-1)
            self.ff = torch.unsqueeze(torch.tensor(ff, dtype=torch.float32, requires_grad=True),-1)
        else:
            # boundary points --- training
            self.xb = torch.tensor(xb, dtype=torch.float32)
            # collocation points --- residual
            self.xf = torch.tensor(xf[:, 0:1], dtype=torch.float32, requires_grad=True)
            self.yf = torch.tensor(xf[:, 1:2], dtype=torch.float32, requires_grad=True)
            self.ff = torch.tensor(ff, dtype=torch.float32)
        
        self.u_net_head_ls = []; self.net_params = []
        self.u_net_body = Net(body_layers, 1, self.act)
        self.net_params += list(self.u_net_body.parameters())
        
        for i in range(self.number_of_cases):
            u_net = Net(head_layers, 0, self.act)
            self.u_net_head_ls.append(u_net)
            self.net_params += list(u_net.parameters())
        
    def get_loss(self):
        
        # Predict the basis (from body network)
        self.body_basis_xb = self.u_net_body(self.xb)
        if self.PDE == 'poisson':
            self.body_basis_xf = self.u_net_body(self.xf)
        else:
            self.body_basis_xf = self.u_net_body(torch.cat((self.xf,self.yf), 1))
        
        loss = 0
        for i in range(self.number_of_cases):

            # BC & IC loss
            mse_ub = (self.u_net_head_ls[i](self.body_basis_xb)).square().mean() # Dirichlet = 0

            # Residual/collocation loss
            u = self.u_net_head_ls[i](self.body_basis_xf)
            if self.PDE == 'poisson':
                u_sum = u.sum()
                u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
                u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
                # Residuals
                f = u_xx - self.ff[i]
            elif self.PDE == 'allencahn':
                u_sum = u.sum()
                u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
                u_y = torch.autograd.grad(u_sum, self.yf, create_graph=True)[0]
                u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
                u_yy = torch.autograd.grad(u_y.sum(), self.yf, create_graph=True)[0]
                # Residuals
                f = 0.1*(u_xx.flatten() + u_yy.flatten()) + u.flatten()*(u.flatten()**2 -1) - self.ff[i].flatten() 
            mse_f = f.square().mean()

            # Loss weights
            w_f = 1
            w_bc = 10
            net_loss = w_bc*mse_ub + w_f*mse_f

            loss += net_loss 
        return loss
    
    def train_adam(self):
        optimizer = torch.optim.Adam(self.net_params, lr=1e-3)
        
        for n in range(self.epoch):
            loss = self.get_loss()
            if n%100==0:
                if self.verbose == 1:
                    print('epoch %d, loss: %g'%(n, loss.item()))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

    def train_lbfgs(self, lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size):
        global iter_count
        iter_count = 0
        
        optimizer = torch.optim.LBFGS(self.net_params, lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size)
        def closure():
            global iter_count
            optimizer.zero_grad()
            loss = self.get_loss()
            loss.backward(retain_graph=True)
            global iter_count 
            iter_count += 1
            if self.verbose == 1 and iter_count%500 == 0:
                print('loss:%g'%loss.item())
            return loss
        optimizer.step(closure)
            
    def predict(self, x_star):
        
        with torch.no_grad():
            if self.PDE == 'poisson': 
                self.body_basis_xf = self.u_net_body(torch.unsqueeze(torch.tensor(x_star, dtype=torch.float32),-1))
            else:
                self.body_basis_xf = self.u_net_body(torch.tensor(x_star, dtype=torch.float32))
                      
        u_pred_ls = []
        for i in range(self.number_of_cases):
            with torch.no_grad():
                u_pred = self.u_net_head_ls[i](self.body_basis_xf)
            u_pred_ls.append(u_pred)
        return u_pred_ls, self.body_basis_xf
    
class MH_PINN_reg:
    def __init__(self, xf, ff, xb, body_layers, head_layers, verbose, epoch, number_of_cases, act, PDE, *args):
        #torch.autograd.set_detect_anomaly(True) ### NO DOT USE without huge computational cost penalty 
        self.epoch = epoch; self.act = act; self.PDE = PDE
        self.verbose = verbose
        self.number_of_cases = number_of_cases
        self.body_layers = body_layers
        self.head_layers = head_layers
        self.X = args[0]
        self.Y = args[1]
        
        if self.PDE == 'poisson': 
            # boundary points --- training
            self.xb = torch.unsqueeze(torch.tensor(xb, dtype=torch.float32),-1)
            # collocation points --- residual
            self.xf = torch.unsqueeze(torch.tensor(xf, dtype=torch.float32, requires_grad=True),-1)
            self.ff = torch.unsqueeze(torch.tensor(ff, dtype=torch.float32, requires_grad=True),-1)
        else:
            # boundary points --- training
            self.xb = torch.tensor(xb, dtype=torch.float32)
            # collocation points --- residual
            self.xf = torch.tensor(xf[:, 0:1], dtype=torch.float32, requires_grad=True)
            self.yf = torch.tensor(xf[:, 1:2], dtype=torch.float32, requires_grad=True)
            self.ff = torch.tensor(ff, dtype=torch.float32)
        
        self.u_net_head_ls = []; self.net_params = []
        self.u_net_body = Net(body_layers, 1, self.act)
        self.net_params += list(self.u_net_body.parameters())
        
        for i in range(self.number_of_cases):
            u_net = Net(head_layers, 0, self.act)
            self.u_net_head_ls.append(u_net)
            self.net_params += list(u_net.parameters())
            
        self.knw_regularization_init()
        
    def get_loss(self):
        
        # Predict the basis (from body network)
        self.body_basis_xb = self.u_net_body(self.xb)
        if self.PDE == 'poisson':
            self.body_basis_xf = self.u_net_body(self.xf)
        else:
            self.body_basis_xf = self.u_net_body(torch.cat((self.xf,self.yf), 1))
            
        loss = 0
        for i in range(self.number_of_cases):

            # BC & IC loss
            mse_ub = (self.u_net_head_ls[i](self.body_basis_xb)).square().mean() # Dirichlet = 0

            # Residual/collocation loss
            u = self.u_net_head_ls[i](self.body_basis_xf)
            if self.PDE == 'poisson':
                u_sum = u.sum()
                u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
                u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
                # Residuals
                f = u_xx - self.ff[i]
            elif self.PDE == 'allencahn':
                u_sum = u.sum()
                u_x = torch.autograd.grad(u_sum, self.xf, create_graph=True)[0]
                u_y = torch.autograd.grad(u_sum, self.yf, create_graph=True)[0]
                u_xx = torch.autograd.grad(u_x.sum(), self.xf, create_graph=True)[0]
                u_yy = torch.autograd.grad(u_y.sum(), self.yf, create_graph=True)[0]
                # Residuals
                f = 0.1*(u_xx.flatten() + u_yy.flatten()) + u.flatten()*(u.flatten()**2 -1) - self.ff[i].flatten() 
            mse_f = f.square().mean() 

            # Loss weights
            w_f = 1
            w_bc = 10

            net_loss = w_bc*mse_ub + w_f*mse_f

            loss += net_loss 
            
        # Add in Knw regularization
        self.regularization = self.get_knw_regularization()
        self.metric = self.get_knw()
        loss += self.regularization
        
        return loss

    def knw_regularization_init(self):
        if self.PDE == 'poisson':
            x = self.xf.detach().numpy()[:,0]
            self.solution_basis = np.array([np.sin(x*np.pi), np.sin(2*x*np.pi), np.sin(3*x*np.pi), np.sin(4*x*np.pi), np.sin(5*x*np.pi)])   
        elif self.PDE == 'allencahn':
            self.solution_basis = np.array([(np.sin(self.X*np.pi)*np.sin(self.Y*np.pi)).flatten(), (np.sin(2*self.X*np.pi)*np.sin(2*self.Y*np.pi)).flatten(), (np.sin(3*self.X*np.pi)*np.sin(3*self.Y*np.pi)).flatten(), (np.sin(4*self.X*np.pi)*np.sin(4*self.Y*np.pi)).flatten(), (np.sin(5*self.X*np.pi)*np.sin(5*self.Y*np.pi)).flatten()])
            
        self.sol_coeff = coeff(len(self.solution_basis), 1, 0 , None)
        self.pred_coeff = coeff(self.body_layers[-1], 0, 0, None)
        
        self.parameters_sol = list(self.sol_coeff.parameters())
        self.parameters_pred = list(self.pred_coeff.parameters())
        
        self.optimizer_sol = torch.optim.Adam(self.parameters_sol, lr=1e-3)
        self.optimizer_pred = torch.optim.Adam(self.parameters_pred, lr=1e-3)
        
    def get_knw_regularization(self): # This function includes the model basis in the computational graph for updating using the model optimizer
        
        u_c = torch.sum((torch.unsqueeze(self.sol_coeff(),-1).clone().detach()*torch.tensor(self.solution_basis)),0)
        u_W1 = torch.sum((torch.unsqueeze(self.pred_coeff(),-1).clone().detach()*self.body_basis_xf.T),0)

        loss_reg = (u_c - u_W1).square().sum().sqrt()
        
        return loss_reg*10
    
    def get_knw(self): # This function does not include the model basis in the computational graph
                
        u_c = torch.sum((torch.unsqueeze(self.sol_coeff(),-1)*torch.tensor(self.solution_basis)),0)
        u_W1 = torch.sum((torch.unsqueeze(self.pred_coeff(),-1)*self.body_basis_xf.T.clone().detach()),0)

        loss_reg = (u_c - u_W1).square().sum().sqrt()
        
        return loss_reg*10
    
    def train_adam(self):
        optimizer_model = torch.optim.Adam(self.net_params, lr=1e-3)
        
        for n in range(self.epoch):
            loss = self.get_loss()
            
            if n%100==0:
                if self.verbose == 1:
                    print('epoch %d, total loss: %g'%(n, loss.item()))
                    print('epoch %d, Kwn loss: %g'%(n, self.metric))
                    
            ### Tri-optimization
            # Model Optimization
            optimizer_model.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_model.step()
            
            self.train_knw()
        
        self.compute_full_knw()
            
    def train_knw(self):
        loss_pred = self.metric
        loss_sol = -self.metric
        
        # Knw solution coefficent optimization
        self.optimizer_sol.zero_grad()
        loss_sol.backward(retain_graph=True)
        self.optimizer_sol.step()
        
        # Knw model coefficent optimization
        self.optimizer_pred.zero_grad()
        loss_pred.backward()
        self.optimizer_pred.step()
        
    def compute_full_knw(self):
        for n in range(5000):
            self.metric = self.get_knw()
            self.train_knw()
        print('Knw:', self.get_knw().detach())
        
    def return_coeff(self):
        return self.sol_coeff(), self.pred_coeff()
    
    def train_lbfgs(self, lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size):
        global iter_count
        iter_count = 0
        total_params = self.net_params + self.parameters_pred
        optimizer = torch.optim.LBFGS(total_params, lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size)
        def closure():
            global iter_count
            optimizer.zero_grad()
            loss = self.get_loss()
            loss.backward(retain_graph=True)
            global iter_count 
            iter_count += 1
            if self.verbose == 1 and iter_count%500 == 0:
                print('total loss:%g'%loss.item())
                print('Kwn loss: %g'%self.metric)
            
            return loss
        optimizer.step(closure)
            
    def predict(self, x_star):
        
        with torch.no_grad():
            if self.PDE == 'poisson': 
                self.body_basis_xf = self.u_net_body(torch.unsqueeze(torch.tensor(x_star, dtype=torch.float32),-1))
            else:
                self.body_basis_xf = self.u_net_body(torch.tensor(x_star, dtype=torch.float32))
                      
        u_pred_ls = []
        for i in range(self.number_of_cases):
            with torch.no_grad():
                u_pred = self.u_net_head_ls[i](self.body_basis_xf)
            u_pred_ls.append(u_pred)
        return u_pred_ls, self.body_basis_xf

class PI_DON():
    def __init__(self, input_u, physicsInformed_u, input_y_f, input_y_b, output_b, branch_layers, trunk_layers, num_of_cases, act, PDE, *args):
        super(PI_DON,self).__init__()
        self.num_of_cases = num_of_cases
        self.act = act
        self.PDE = PDE
        self.branch = Net(branch_layers, 0, 'tanh')
        self.trunk = Net(trunk_layers, 1, self.act)
        
        if self.PDE == 'poisson': 
            self.input_u = torch.tensor(input_u, dtype = torch.float32, requires_grad=True)
            self.physicsInformed_u = torch.tensor(physicsInformed_u, dtype = torch.float32)
            self.input_y_f = torch.unsqueeze(torch.tensor(input_y_f, dtype = torch.float32, requires_grad=True),-1)
            self.input_y_b = torch.unsqueeze(torch.tensor(input_y_b, dtype = torch.float32, requires_grad=True),-1)
            self.output_b = torch.tensor(output_b, dtype = torch.float32, requires_grad=True)
        else:
            self.input_u = torch.tensor(input_u, dtype = torch.float32, requires_grad=True)
            self.physicsInformed_u = torch.tensor(physicsInformed_u, dtype = torch.float32)
            self.input_y_fx = torch.tensor(input_y_f[:, 0:1], dtype = torch.float32, requires_grad=True) # x dim
            self.input_y_fy = torch.tensor(input_y_f[:, 1:2], dtype = torch.float32, requires_grad=True) # y dim
            self.input_y_b = torch.tensor(input_y_b, dtype = torch.float32)
            self.output_b = torch.tensor(output_b, dtype = torch.float32)
            
        self.params = list(self.branch.parameters()) + list(self.trunk.parameters())
        
    def get_loss(self):
        
        loss = 0
        if self.PDE == 'poisson':
            trunk_out_f = self.trunk(self.input_y_f)
        else:
            trunk_out_f = self.trunk(torch.cat((self.input_y_fx,self.input_y_fy), 1))   
        trunk_out_b = self.trunk(self.input_y_b)
        
        for i in range(self.num_of_cases):
            
            coeff = self.branch(self.input_u[i])

            # BC & IC
            pred_b = torch.sum(coeff * trunk_out_b, dim=-1, keepdim=True)
            #mse_b = (self.output_b - pred_b.flatten()).square().mean()
            mse_b = (pred_b.flatten()).square().mean()
            
            # Residuals
            pred_f = torch.sum(coeff * trunk_out_f, dim=-1, keepdim=True)
            if self.PDE == 'poisson':
                pred_f_sum = pred_f.sum()
                pred_f_x = torch.autograd.grad(pred_f_sum, self.input_y_f, create_graph=True)[0]
                pred_f_xx = torch.autograd.grad(pred_f_x.sum(), self.input_y_f, create_graph=True)[0]
                f = pred_f_xx.flatten() -  self.physicsInformed_u[i][1:-1].flatten()
            if self.PDE == 'allencahn':
                pred_f_sum = pred_f.sum()
                pred_f_x = torch.autograd.grad(pred_f_sum, self.input_y_fx, create_graph=True)[0]
                pred_f_y = torch.autograd.grad(pred_f_sum, self.input_y_fy, create_graph=True)[0]
                pred_f_xx = torch.autograd.grad(pred_f_x.sum(), self.input_y_fx, create_graph=True)[0]
                pred_f_yy = torch.autograd.grad(pred_f_y.sum(), self.input_y_fy, create_graph=True)[0]
                # fix forcing [1:-1] for 2D
                f = 0.1*(pred_f_xx.flatten() + pred_f_yy.flatten()) + pred_f.flatten()*(pred_f.flatten()**2 - 1) -  self.physicsInformed_u[i].flatten() 
                
            mse_f = f.square().mean()
            
            loss += mse_b*10 + mse_f
        
        return loss
    
    def train(self, epochs):
        optimizer = torch.optim.Adam(self.params , lr=1e-3)
        
        for i in range(epochs):
            loss = self.get_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100 == 0:
                print('Epoch: ', i)
                print('loss:%g'%loss.item())
        return 
    
    def train_lbfgs(self, lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size):
        loss_list = []
        global iter_count
        iter_count = 0
        
        optimizer = torch.optim.LBFGS(self.params, lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size)
        def closure():
            global iter_count
            optimizer.zero_grad()
            loss = self.get_loss()
            loss.backward(retain_graph=True)
            global iter_count 
            iter_count += 1
            loss_list.append(loss.detach().numpy())
            if iter_count%500 == 0:
                print('L-BFGS iter: ', iter_count)
                print('loss:%g'%loss.item())
            return loss
        optimizer.step(closure)
        return             
                
    def pred(self, input_u, input_y):
        pred = []
        coeff_ls = []
        with torch.no_grad():
            if self.PDE == 'poisson':
                    trunk_out = self.trunk(torch.unsqueeze(torch.tensor(input_y, dtype = torch.float32),-1))
            else:
                trunk_out = self.trunk(torch.tensor(input_y, dtype = torch.float32))
            
        for i in range(self.num_of_cases):
            with torch.no_grad():
                branch_out = self.branch(torch.tensor(input_u[i], dtype = torch.float32))
            coeff_ls.append(branch_out)
            
            pred.append(torch.sum(branch_out * trunk_out, dim=-1, keepdim=True))
            
        return pred, trunk_out, coeff_ls
    
    
class PI_DON_reg():
    def __init__(self, input_u, physicsInformed_u , input_y_f, input_y_b, output_b, branch_layers, trunk_layers, num_of_cases, act, PDE, *args):
        super(PI_DON_reg,self).__init__()
        self.num_of_cases = num_of_cases
        self.act = act
        self.PDE = PDE
        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.branch = Net(branch_layers, 0, 'tanh')
        self.trunk = Net(trunk_layers, 1, self.act)
        self.X = args[0]
        self.Y = args[1]
        
        if self.PDE == 'poisson': 
            self.input_u = torch.tensor(input_u, dtype = torch.float32, requires_grad=True)
            self.physicsInformed_u = torch.tensor(physicsInformed_u, dtype = torch.float32)
            self.input_y_f = torch.unsqueeze(torch.tensor(input_y_f, dtype = torch.float32, requires_grad=True),-1)
            self.input_y_b = torch.unsqueeze(torch.tensor(input_y_b, dtype = torch.float32, requires_grad=True),-1)
            self.output_b = torch.tensor(output_b, dtype = torch.float32, requires_grad=True)
        else:
            self.input_u = torch.tensor(input_u, dtype = torch.float32, requires_grad=True)
            self.physicsInformed_u = torch.tensor(physicsInformed_u, dtype = torch.float32)
            self.input_y_fx = torch.tensor(input_y_f[:, 0:1], dtype = torch.float32, requires_grad=True) # x dim
            self.input_y_fy = torch.tensor(input_y_f[:, 1:2], dtype = torch.float32, requires_grad=True) # y dim
            self.input_y_b = torch.tensor(input_y_b, dtype = torch.float32)
            self.output_b = torch.tensor(output_b, dtype = torch.float32)        

        self.params = list(self.branch.parameters()) + list(self.trunk.parameters())
        
        self.knw_regularization_init()
        
    def get_loss(self):
        
        loss = 0
        
        if self.PDE == 'poisson':
            self.trunk_out_f = self.trunk(self.input_y_f)
        else:
            self.trunk_out_f = self.trunk(torch.cat((self.input_y_fx,self.input_y_fy), 1))   
        self.trunk_out_b = self.trunk(self.input_y_b)
        
        for i in range(self.num_of_cases):
            
            coeff = self.branch(self.input_u[i])

            # BC & IC
            pred_b = torch.sum(coeff * self.trunk_out_b, dim=-1, keepdim=True)
            #mse_b = (self.output_b - pred_b.flatten()).square().mean()
            mse_b = (pred_b.flatten()).square().mean()
            
            # Residuals
            pred_f = torch.sum(coeff * self.trunk_out_f, dim=-1, keepdim=True)
            if self.PDE == 'poisson':
                pred_f_sum = pred_f.sum()
                pred_f_x = torch.autograd.grad(pred_f_sum, self.input_y_f, create_graph=True)[0]
                pred_f_xx = torch.autograd.grad(pred_f_x.sum(), self.input_y_f, create_graph=True)[0]
                f = pred_f_xx.flatten() -  self.physicsInformed_u[i][1:-1].flatten()
            if self.PDE == 'allencahn':
                pred_f_sum = pred_f.sum()
                pred_f_x = torch.autograd.grad(pred_f_sum, self.input_y_fx, create_graph=True)[0]
                pred_f_y = torch.autograd.grad(pred_f_sum, self.input_y_fy, create_graph=True)[0]
                pred_f_xx = torch.autograd.grad(pred_f_x.sum(), self.input_y_fx, create_graph=True)[0]
                pred_f_yy = torch.autograd.grad(pred_f_y.sum(), self.input_y_fy, create_graph=True)[0]
                # fix forcing [1:-1] for 2D
                f = 0.1*(pred_f_xx.flatten() + pred_f_yy.flatten()) + pred_f.flatten()*(pred_f.flatten()**2 - 1) -  self.physicsInformed_u[i].flatten() 
                
            mse_f = f.square().mean()
            
            loss += mse_b*10 + mse_f
            
        # Add in Knw regularization
        self.regularization = self.get_knw_regularization()
        self.metric = self.get_knw()
        loss += self.regularization
        
        return loss
    
    def knw_regularization_init(self):
        if self.PDE == 'poisson':
            x = self.input_y_f.detach().numpy()[:,0]
            self.solution_basis = np.array([np.sin(x*np.pi), np.sin(2*x*np.pi), np.sin(3*x*np.pi), np.sin(4*x*np.pi), np.sin(5*x*np.pi)])
        elif self.PDE == 'allencahn':
            self.solution_basis = np.array([(np.sin(self.X*np.pi)*np.sin(self.Y*np.pi)).flatten(), (np.sin(2*self.X*np.pi)*np.sin(2*self.Y*np.pi)).flatten(), (np.sin(3*self.X*np.pi)*np.sin(3*self.Y*np.pi)).flatten(), (np.sin(4*self.X*np.pi)*np.sin(4*self.Y*np.pi)).flatten(), (np.sin(5*self.X*np.pi)*np.sin(5*self.Y*np.pi)).flatten()])
            
        self.sol_coeff = coeff(len(self.solution_basis),1, 0 , None)
        self.pred_coeff = coeff(self.trunk_layers[-1],0, 0, None)
        
        self.parameters_sol = list(self.sol_coeff.parameters())
        self.parameters_pred = list(self.pred_coeff.parameters())
        
        self.optimizer_sol = torch.optim.Adam(self.parameters_sol, lr=1e-3)
        self.optimizer_pred = torch.optim.Adam(self.parameters_pred, lr=1e-3)
        
    def get_knw_regularization(self): # This function includes the model basis in the computational graph for updating using the model optimizer
        
        u_c = torch.sum((torch.unsqueeze(self.sol_coeff(),-1).clone().detach()*torch.tensor(self.solution_basis)),0)
        u_W1 = torch.sum((torch.unsqueeze(self.pred_coeff(),-1).clone().detach()*self.trunk_out_f.T),0)
        #u_W1 = torch.sum((torch.unsqueeze(self.pred_coeff(),-1)*self.body_basis_xf.T),0)
        loss_reg = (u_c - u_W1).square().sum().sqrt()
        
        return loss_reg*10
    
    def get_knw(self): # This function does not include the model basis in the computational graph
                
        u_c = torch.sum((torch.unsqueeze(self.sol_coeff(),-1)*torch.tensor(self.solution_basis)),0)
        u_W1 = torch.sum((torch.unsqueeze(self.pred_coeff(),-1)*self.trunk_out_f.T.clone().detach()),0)
        #u_W1 = torch.sum((torch.unsqueeze(self.pred_coeff(),-1).clone().detach()*self.body_basis_xf.T.clone().detach()),0)
        loss_reg = (u_c - u_W1).square().sum().sqrt()
        
        return loss_reg*10
    
    def train(self, epochs):
        loss_list = []
        test_error = []

        optimizer = torch.optim.AdamW(self.params , lr=1e-3)
        for i in range(epochs):
            loss = self.get_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.detach().numpy())
            if i%100 == 0:
                print('Epoch: ', i)
                print('loss:%g'%loss.item())
                print('Kwn loss: %g'%self.metric)
                
            self.train_knw()
    
        self.compute_full_knw()
        
    def train_knw(self):
        loss_pred = self.metric
        loss_sol = -self.metric
        
        # Knw solution coefficent optimization
        self.optimizer_sol.zero_grad()
        loss_sol.backward(retain_graph=True)
        self.optimizer_sol.step()
        
        # Knw model coefficent optimization
        self.optimizer_pred.zero_grad()
        loss_pred.backward()
        self.optimizer_pred.step()
        
    def compute_full_knw(self):
        for n in range(5000):
            self.metric = self.get_knw()
            self.train_knw()
        print('Knw:', self.get_knw().detach())
        
    def return_coeff(self):
        return self.sol_coeff(), self.pred_coeff()
    
    def train_lbfgs(self, lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size):
        loss_list = []
        global iter_count
        iter_count = 0
        total_params = self.params + self.parameters_pred
        optimizer = torch.optim.LBFGS(total_params, lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size)
        def closure():
            global iter_count
            optimizer.zero_grad()
            loss = self.get_loss()
            loss.backward(retain_graph=True)
            global iter_count 
            iter_count += 1
            loss_list.append(loss.detach().numpy())
            if iter_count%500 == 0:
                print('L-BFGS iter: ', iter_count)
                print('loss:%g'%loss.item())
                print('Kwn loss: %g'%self.metric)
            return loss
        optimizer.step(closure)
        return
    
    def pred(self, input_u, input_y):
        pred = []
        coeff_ls = []
        with torch.no_grad():
            if self.PDE == 'poisson':
                    trunk_out = self.trunk(torch.unsqueeze(torch.tensor(input_y, dtype = torch.float32),-1))
            else:
                trunk_out = self.trunk(torch.tensor(input_y, dtype = torch.float32))
            
        for i in range(self.num_of_cases):
            with torch.no_grad():
                branch_out = self.branch(torch.tensor(input_u[i], dtype = torch.float32))
            coeff_ls.append(branch_out)
            
            pred.append(torch.sum(branch_out * trunk_out, dim=-1, keepdim=True))
            
        return pred, trunk_out, coeff_ls