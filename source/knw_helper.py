import numpy as np
import matplotlib.pyplot as plt
import sympy
import pickle
import torch
import torch.nn as nn
import time
import sympy as sy

# Class for learnable coefficent parameters
class coeff(nn.Module):
    def __init__(self, num, bound, init, init_params):
        super().__init__()
        self.num = num; self.bound = bound
        if init == 1:
            self.params = nn.Parameter(init_params)
        else:
            self.params = nn.Parameter(torch.randn(num))

    def forward(self):
        if self.bound == 1:
            return torch.sigmoid(self.params)
        else:
            return self.params
    
class K_n_width:
    def __init__(self, solution_basis, model_basis, adam_lr, verbose, exact, init, init_params_sol, init_params_pred, *args):
        self.exact = exact
        self.adam_lr = adam_lr
        self.verbose = verbose
        self.solution_basis = solution_basis
        self.model_basis = model_basis
        self.sol_coeff = coeff(len(solution_basis),1, init, init_params_sol)
        self.pred_coeff = coeff(len(model_basis),0, init, init_params_pred)
        self.normalize = args[0]
        self.forcing_basis = args[1]
        self.unitball = args[2]
        
        self.parameters_sol = list(self.sol_coeff.parameters())
        self.parameters_pred = list(self.pred_coeff.parameters())
        
    def get_loss(self):
        # Compute Kolmogorov n-width
        
        # Numerator
        u_c = torch.sum((torch.unsqueeze(self.sol_coeff(),-1)*torch.tensor(self.solution_basis)),0)
        u_W1 = torch.sum((torch.unsqueeze(self.pred_coeff(),-1)*torch.tensor(self.model_basis)),0)
        numerator = (u_c - u_W1).square().sum().sqrt()
        
        if self.normalize == 1:
            f_c = torch.sum((torch.unsqueeze(self.sol_coeff(),-1)*torch.tensor(self.forcing_basis)),0)
            denominator = (f_c).square().sum().sqrt()
            loss = numerator/denominator
        else:
            loss = numerator
            
        return loss
    
    def get_unitball_loss(self):
        if self.unitball == 1:
            unitball_residual = 10*torch.absolute(0.5-torch.sqrt(torch.sum(torch.square((0.5-self.sol_coeff())))))
        else:
            unitball_residual = 0
        return unitball_residual
    
    def train_adam(self, epoch):
        optimizer_sol = torch.optim.Adam(self.parameters_sol, lr=self.adam_lr)
        optimizer_pred = torch.optim.Adam(self.parameters_pred, lr=self.adam_lr)
        for n in range(epoch):
            loss_pred = self.get_loss() 
            loss_sol = -loss_pred + self.get_unitball_loss()
            if n%100==0:
                if self.verbose == 1:
                    print('epoch %d, loss: %g'%(n, loss_pred.item()))
            optimizer_sol.zero_grad()
            loss_sol.backward(retain_graph=True)
            optimizer_sol.step()
            
            optimizer_pred.zero_grad()
            loss_pred.backward()
            optimizer_pred.step()
            
    def metric(self):
        return self.get_loss() 
    
    def solution(self):
        u_c = torch.sum((torch.unsqueeze(self.sol_coeff(),-1)*torch.tensor(self.solution_basis)),0)
        u_W1 = torch.sum((torch.unsqueeze(self.pred_coeff(),-1)*torch.tensor(self.model_basis)),0)
        return u_c, u_W1, self.sol_coeff(), self.pred_coeff()