'''
Kevin Zhang 2023
'''

from utils import MAPE
from model import model_summary

from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, ReduceLROnPlateau
from torch.optim import NAdam
import torch

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import time
import gc
    
    
class Trainer:
    def __init__(self, model, config, train_set, val_set, test_set=None):
        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.config = config
        
        if self.test_set is None:
            self.test_set = self.val_set
        
        if config.device=='gpu' and torch.cuda.is_available(): # take over whatever gpus are on the system
            self.device = torch.cuda.current_device()
            self.model.to(self.device)
            print('using f{self.device}')
        else:
            self.device = 'cpu'
            print('using CPU')
            
        self.opt = NAdam(self.model.parameters(), 
                         lr=config.ini_lr, 
                         betas=config.betas, 
                         eps=config.eps,
                         weight_decay=config.weight_decay,
                         )
        
        model_summary(self.model)
        print('model loaded in cuda:', next(self.model.parameters()).is_cuda)
        
    def train(self):
        model, optimizer, config = self.model, self.opt, self.config
        loss_function = torch.nn.CrossEntropyLoss()
        # scheduler = ExponentialLR(optimizer, 
        #                           gamma=config.gamma,
        #                           )
        # scheduler = LinearLR(optimizer, 
        #                       start_factor=1., 
        #                       end_factor=config.fin_lr/config.ini_lr,
        #                       total_iters=config.max_epochs,
        #                      )
        scheduler = ReduceLROnPlateau(optimizer, 
                                      factor=0.5, 
                                      patience=20,
                                      min_lr=1e-6,
                                      verbose=True
                                      )
        
        loader = DataLoader(self.train_set, shuffle=True, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)
        
        print(vars(config))
        
        lrs, train_losses, val_losses = [], [], []
        
        for epoch in range(config.max_epochs):
            t0 = time.time()
            
            if self.config.progress_bar:
                # set up a progress bar
                pbar = tqdm(enumerate(loader), 
                            total=len(loader), 
                            # bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                            unit="batch",
                            )
            else:
                pbar = enumerate(loader)
            
            sum_loss = 0.
            sum_err = 0.
            for it, (x, y) in pbar:
                x, y = x.to(self.device), y.to(self.device)
                
                model.train()
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_function(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()
                
                sum_loss += loss.item()
                
                train_out = torch.argmax(pred, axis=1) == torch.argmax(y, axis=1)
                train_acc = train_out.sum() / pred.size()[0]
                
                # lr = scheduler.get_last_lr()[0]
                lr = optimizer.param_groups[0]['lr']
                
                if self.config.progress_bar:
                    pbar.set_description(f'epoch {epoch+1} iter {it}: lr {lr:.4e} loss {loss.item():.4e} batch train_acc {train_acc.item():.4f}')
                
            # validation
            model.eval()
            with torch.no_grad():
                val_x = torch.tensor(self.val_set.x).to(self.device)
                val_y = torch.tensor(self.val_set.y).to(self.device)
                val_pred = model(val_x)
                val_loss = loss_function(val_pred, val_y)
                
                pred_out = torch.argmax(val_pred, axis=1) == torch.argmax(val_y, axis=1)
                pred_acc = pred_out.sum() / val_pred.size()[0]
                
            lrs.append(lr)
            train_losses.append(sum_loss/len(loader))
            val_losses.append(val_loss.item())
            
            print(f'End of epoch {epoch+1}: {time.time()-t0:4f}s lr {lr:.4e} train_loss {train_losses[-1]:.4e} train_acc {train_acc.item():.4f} val_loss {val_losses[-1]:.4e} pred_acc {pred_acc:.4f}')
            
            # update lr scheduler
            scheduler.step(val_loss.item())
            
            # save model
            if (config.epoch_save_freq > 0 and epoch > config.max_epochs-500 and epoch % config.epoch_save_freq == 0) \
                or (epoch == config.max_epochs - 1):
                torch.save(model, 
                           config.epoch_save_name+str(epoch+1)+'.pth' # f'_{self.avg_loss:.2f}.pth'
                           )
        
            gc.collect()
            
        print('training finished')
        
        return {'lr': lrs, 'loss': train_losses, 'val_loss': val_losses}
    
    
    def test(self):
        if self.test_set is None:
            print('No allocated test_set')
            return None
        
        test_x = torch.tensor(self.test_set.x).to(self.device)
        test_y = torch.tensor(self.test_set.y).to(self.device)
        loss_function = torch.nn.MSELoss()
        
        self.model.eval()
        with torch.no_grad():
            test_pred = self.model(test_x)
            loss = loss_function(test_pred, test_y)
            test_err = MAPE(test_pred, test_y)
            test_err_np = test_err.cpu().numpy()
            
        pred = np.argmax(test_pred.cpu().numpy(), axis=1)
        label = np.argmax(self.test_set.y, axis=1)
        
        from sklearn.metrics import confusion_matrix
        import seaborn as sn
        import pandas as pd
        
        cm = confusion_matrix(label, pred, normalize='true')
        fig, ax = plt.subplots(figsize=(6,6)) 
        df_cm = pd.DataFrame(cm, index = [i for i in np.arange(10)],
                      columns = [i for i in np.arange(10)])
        sn.heatmap(df_cm, annot=True, cmap="OrRd", fmt=".1%", ax=ax, vmin=0., vmax=1.)
        ax.invert_yaxis()
        ax.set_xlabel('prediction')
        ax.set_ylabel('label')
        fig.show()
        
        return {'loss': loss, 'err': test_err}
        
