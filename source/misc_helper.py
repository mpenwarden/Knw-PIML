import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
import pickle

def animation(path, model_name, fig_num, x, u_c_optimized, u_W1_optimized, *args):
    PDE = args[0]
    y = args[1]
    triang = args[2]
    
    if PDE == 'poisson':
        plt.figure(figsize = (6.4,4.8))
        plt.plot(x,u_c_optimized.detach().numpy(),'k--',linewidth=2)
        plt.plot(x,u_W1_optimized.detach().numpy(),'r')
        plt.ylabel('u', fontsize=20); plt.xlabel('x', fontsize=20);
        plt.xticks(fontsize=10); plt.yticks(fontsize=10); plt.legend(['u(c)', 'u(W1,W2)'])

        plt.savefig(path + '\\figures\\' + model_name + '_animate_' + str(fig_num), dpi = 300)
        plt.close()
        return
    else:
        plt.figure(figsize = (17.5,5))
        plt.subplot(1, 3, 1)
        plt.title('Prediction')
        plt.tricontourf(triang, u_W1_optimized.reshape(x.shape[0],y.shape[0]).detach().numpy().flatten(), 100, cmap='jet')
        plt.colorbar()
        plt.xlabel('x'); plt.ylabel('y')

        plt.subplot(1, 3, 2)
        plt.title('Solution')
        plt.tricontourf(triang, u_c_optimized.reshape(x.shape[0],y.shape[0]).detach().numpy().flatten(), 100, cmap='jet')
        plt.colorbar()
        plt.xlabel('x'); plt.ylabel('y')

        plt.subplot(1, 3, 3)
        plt.title('Point-wise Error')
        plt.tricontourf(triang, abs(u_W1_optimized.reshape(x.shape[0],y.shape[0]).detach().numpy().flatten()-u_c_optimized.reshape(x.shape[0],y.shape[0]).detach().numpy().flatten()), 100, cmap='jet')
        plt.colorbar()
        plt.xlabel('x'); plt.ylabel('y')

        plt.suptitle('KnW Optimized', fontsize = 20)
        plt.tight_layout()

        plt.savefig(path + '\\figures\\' + model_name + '_animate_' + str(fig_num), dpi = 300)
        plt.close()
        return

def competitive_plot(path, model_name, log, rate, *args):
    PDE = args[0]
    x = args[1]
    y = args[2]
    triang = args[3]

    
    if PDE == 'poisson':
        num = len(log)

        plt.figure()
        plt.rcParams['text.usetex'] = True
        x = log[0][0]; u_c = log[0][1]; u_W1 = log[0][2]
        plt.plot(x,u_W1.detach().numpy(),'r', linewidth = 0.33, alpha = 0.33)
        plt.plot(x,u_c.detach().numpy(),'k--', linewidth = 0.33,alpha = 0.33)

        x = log[-1][0]; u_c = log[-1][1]; u_W1 = log[-1][2]
        plt.plot(x,u_W1.detach().numpy(), 'r', linewidth = 1, alpha = 1, label = 'u(W1,W2)')
        plt.plot(x,u_c.detach().numpy(), 'k--', linewidth = 1, alpha = 1, label = 'u(c)')

        plt.ylabel('u', fontsize=20); plt.xlabel('x', fontsize=20);
        plt.xticks(fontsize=10); plt.yticks(fontsize=10); plt.legend()

        plt.savefig(path + '\\figures\\' + model_name + '_competitivePlot', dpi = 300, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        x = log[-1][0]; u_c = log[-1][1]; u_W1 = log[-1][2]
        plt.figure(figsize = (17.5,5))
        plt.subplot(1, 3, 1)
        plt.title('Prediction')
        plt.tricontourf(triang, u_W1.reshape(x.shape[0],y.shape[0]).detach().numpy().flatten(), 100, cmap='jet')
        plt.colorbar()
        plt.xlabel('x'); plt.ylabel('y')

        plt.subplot(1, 3, 2)
        plt.title('Solution')
        plt.tricontourf(triang, u_c.reshape(x.shape[0],y.shape[0]).detach().numpy().flatten(), 100, cmap='jet')
        plt.colorbar()
        plt.xlabel('x'); plt.ylabel('y')

        plt.subplot(1, 3, 3)
        plt.title('Point-wise Error')
        plt.tricontourf(triang, abs(u_W1.reshape(x.shape[0],y.shape[0]).detach().numpy().flatten()-u_c.reshape(x.shape[0],y.shape[0]).detach().numpy().flatten()), 100, cmap='jet')
        plt.colorbar()
        plt.xlabel('x'); plt.ylabel('y')

        plt.suptitle('KnW Optimized', fontsize = 20)
        plt.tight_layout()
        plt.savefig(path + '\\figures\\' + model_name + '_competitivePlot', dpi = 300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    return

class knw_log():
    def __init__(self):
        self.log = []
    
    def add(self, x, u_c_optimized, u_W1_optimized):
        self.log.append([x, u_c_optimized, u_W1_optimized])
    
    def report(self):
        return self.log
    
def training_results(number_of_cases, x, xb, u_ls, mhpinn_u_pred, basis, body_layers, path, model_name, PDE, *args):
    triang = args[0]
    
    if PDE == 'poisson':
        l2_error_ls = []; norm2_ls = []
        for i in range(number_of_cases):
            pred = mhpinn_u_pred[i].numpy().flatten()
            l2_error_ls.append(np.linalg.norm(u_ls[i]-pred.flatten(), 2)/np.linalg.norm(u_ls[i], 2))
            norm2_ls.append(np.linalg.norm(u_ls[i]-pred.flatten(), 2))
        print('Relative L2 Error (mean):', np.mean(np.array(l2_error_ls)))
        print('Relative L2 Error (std):', np.std(np.array(l2_error_ls)))
        print('Relative L2 Error (median):', np.median(np.array(l2_error_ls)))
        print('Relative L2 Error (min-max):', [np.min(np.array(l2_error_ls)),np.max(np.array(l2_error_ls))])
        print('2-Norm Error (mean):', np.mean(np.array(norm2_ls)))
        print('2-Norm Error (std):', np.std(np.array(norm2_ls)))

        # Plot
        plt.figure(figsize = (15,22.5))
        plt.rcParams['text.usetex'] = True
        plt.subplot(4,1,1)
        for i in range(body_layers[-1]):
            plt.plot(x, basis.detach().numpy()[:,i])

        for i in range(number_of_cases):
            plt.subplot(4,1,2)
            plt.plot(x, mhpinn_u_pred[i].numpy().flatten())
            plt.subplot(4,1,3)
            plt.plot(x, u_ls[i])
            plt.subplot(4,1,4)
            plt.plot(x, abs(u_ls[i]-mhpinn_u_pred[i].numpy().flatten()))

        plt.subplot(4,1,1)
        plt.ylabel('u', fontsize=20); plt.xlabel('x', fontsize=20); plt.title('Learned (body) basis functions', fontsize=20);
        plt.xticks(fontsize=0); plt.yticks(fontsize=15); 
        plt.subplot(4,1,2)
        plt.scatter(xb,xb*0, c ='k', s = 50)
        plt.ylabel('u', fontsize=20); plt.xlabel('x', fontsize=20); plt.title('Prediction', fontsize=20);
        plt.xticks(fontsize=0); plt.yticks(fontsize=15); 
        plt.subplot(4,1,3)
        plt.scatter(xb,xb*0, c ='k', s = 50)
        plt.ylabel('u', fontsize=20); plt.xlabel('x', fontsize=20); plt.title('Solution', fontsize=20);
        plt.xticks(fontsize=0); plt.yticks(fontsize=15);
        plt.subplot(4,1,4)
        plt.scatter(xb,xb*0, c ='k', s = 50)
        plt.ylabel(r'|u_{sol} - u_{pred}|', fontsize=20); plt.xlabel('x', fontsize=20); plt.title('Point-wise Error', fontsize=20);
        plt.xticks(fontsize=15); plt.yticks(fontsize=15);

        plt.savefig(path + '\\figures\\' + model_name + '_modelTraining', dpi = 300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    else:
        for i in range(number_of_cases):
            mhpinn_u_pred[i] = mhpinn_u_pred[i].numpy()
        mhpinn_u_pred = np.array(mhpinn_u_pred)

        l2_error_ls = []; norm2_ls = []
        for i in range(number_of_cases):
            pred = mhpinn_u_pred[i].flatten()
            l2_error_ls.append(np.linalg.norm(u_ls[i].flatten()-pred.flatten(), 2)/np.linalg.norm(u_ls[i].flatten(), 2))
            norm2_ls.append(np.linalg.norm(u_ls[i].flatten()-pred.flatten(), 2))
        print('Relative L2 Error (mean):', np.mean(np.array(l2_error_ls)))
        print('Relative L2 Error (std):', np.std(np.array(l2_error_ls)))
        print('Relative L2 Error (median):', np.median(np.array(l2_error_ls)))
        print('Relative L2 Error (min-max):', [np.min(np.array(l2_error_ls)),np.max(np.array(l2_error_ls))])
        print('2-Norm Error (mean):', np.mean(np.array(norm2_ls)))
        print('2-Norm Error (std):', np.std(np.array(norm2_ls)))

        plt.figure(figsize=(15, 15))
        vmin = basis.detach().numpy().min()
        vmax = basis.detach().numpy().max()
        plot_num = int(np.ceil(np.sqrt(body_layers[-1])))
        for i in range(body_layers[-1]):
            plt.subplot(plot_num, plot_num, i+1)
            plt.title('Basis \#' + str(i+1))
            plt.tricontourf(triang, basis.detach().numpy()[:,i].flatten(), 100, cmap='jet', vmin=vmin, vmax=vmax)
            plt.axis('off')

        plt.suptitle('Learned Network Basis Functions', fontsize = 20)
        plt.savefig(path + '\\figures\\' + model_name + '_modelTraining_Basis', dpi = 300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

        vmin_pred = np.concatenate((u_ls[:number_of_cases,:,:].flatten(),mhpinn_u_pred.flatten())).min()
        vmax_pred = np.concatenate((u_ls[:number_of_cases,:,:].flatten(),mhpinn_u_pred.flatten())).max()
        vmin_err = (abs(u_ls[:number_of_cases,:,:].flatten()-mhpinn_u_pred.flatten())).min()
        vmax_err = (abs(u_ls[:number_of_cases,:,:].flatten()-mhpinn_u_pred.flatten())).max()

        # Plot means over example set
        plt.figure(figsize = (17.5,5))
        plt.subplot(1, 3, 1)
        plt.title('Prediction Mean')
        plt.tricontourf(triang, np.mean(mhpinn_u_pred,0).flatten(), 100, cmap='jet')
        plt.colorbar()
        plt.xlabel('x'); plt.ylabel('y')

        plt.subplot(1, 3, 2)
        plt.title('Solution Mean')
        plt.tricontourf(triang, np.mean(u_ls,0).flatten(), 100, cmap='jet')
        plt.colorbar()
        plt.xlabel('x'); plt.ylabel('y')

        plt.subplot(1, 3, 3)
        plt.title('Point-wise Error Mean')
        plt.tricontourf(triang, np.mean(abs(u_ls[:number_of_cases].reshape(number_of_cases,u_ls.shape[1]*u_ls.shape[2])-mhpinn_u_pred.reshape(number_of_cases,mhpinn_u_pred.shape[1])),0), 100, cmap='jet')
        plt.colorbar()
        plt.xlabel('x'); plt.ylabel('y')

        plt.suptitle('Mean Plots over Examples', fontsize = 20)
        plt.savefig(path + '\\figures\\' + model_name + '_modelTraining_Mean', dpi = 300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()


        # Plot each exmaple
        plt.figure(figsize = (17.5,5*number_of_cases))
        for i in range(number_of_cases):
            plt.subplot(number_of_cases, 3, 1+i*3)
            plt.title('Prediction: Case #' + str(i+1))
            plt.tricontourf(triang, mhpinn_u_pred[i].flatten(), 100, cmap='jet', vmin=vmin_pred, vmax=vmax_pred)
            plt.colorbar()
            plt.xlabel('x'); plt.ylabel('y')

            plt.subplot(number_of_cases, 3, 2+i*3)
            plt.title('Solution: Case #' + str(i+1))
            plt.tricontourf(triang, u_ls[i].flatten(), 100, cmap='jet', vmin=vmin_pred, vmax=vmax_pred)
            plt.colorbar()
            plt.xlabel('x'); plt.ylabel('y')

            plt.subplot(number_of_cases, 3, 3+i*3)
            plt.title('Point-wise Error: Case #' + str(i+1))
            plt.tricontourf(triang, abs(u_ls[i].flatten()-mhpinn_u_pred[i].flatten()), 100, cmap='jet', vmin=vmin_err, vmax=vmax_err)
            plt.colorbar()
            plt.xlabel('x'); plt.ylabel('y')

        plt.suptitle('Individual Example Plots', fontsize = 20)
        plt.savefig(path + '\\figures\\' + model_name + '_modelTraining_Examples', dpi = 300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    
    return l2_error_ls