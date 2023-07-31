

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
import time
import os 
from metrics import metric
from Read_Data import *

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, seq_len ,pred_len  ):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        return x # [Batch, Output length, Channel]

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class Linear:

    def __init__(self , pred_len , label_len , seq_len , features , lr , use_gpu =True , use_multi_gpu=False , gpu =1 ):
        self.pred_len = pred_len
        self.label_len = label_len
        self.learning_rate = lr
        self.seq_len = seq_len
        self.features = features
        self.use_gpu = use_gpu
        self.use_multi_gpu = use_multi_gpu
        self.gpu =gpu
        self.device = self._acquire_device()
        self.model = Model(self.seq_len , self.pred_len ).to(self.device)

        

    def _acquire_device(self):
        
        if self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.gpu) if not self.use_multi_gpu else self.devices
            device = torch.device('cuda:{}'.format(self.gpu))
            print('Use GPU: cuda:{}'.format(self.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def data_provider(self, data, flag , embed , freq , batch_size , root_path ,data_path , target , num_workers):
        Data = data
        timeenc = 0 if embed != 'timeF' else 1


        if flag == 'test':
            shuffle_flag = False
            drop_last = False
            batch_size = batch_size
            freq = freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = batch_size
            freq = freq

        data_set = Data(
            root_path=root_path,
            data_path=data_path,
            flag=flag,
            size=[self.seq_len, self.label_len, self.pred_len],
            features=self.features,
            target=target,
            timeenc=timeenc,
            freq=freq,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=num_workers,
            drop_last=drop_last)
        return data_set, data_loader

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters() , lr=self.learning_rate)
        return model_optim

    def adjust_learning_rate(self , optimizer, epoch):
        # lr = args.learning_rate * (0.2 ** (epoch // 2))
        lr_adjust = {epoch: self.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

    def visual(self, true, preds=None, name='./pic/test.pdf'):
        """
        Results visualization
        """
        plt.figure()
        plt.plot(true, label='GroundTruth', linewidth=2)
        if preds is not None:
            plt.plot(preds, label='Prediction', linewidth=2)
        plt.legend()
        plt.savefig(name, bbox_inches='tight')
        plt.show()
        print("  ")
    def visual_loss(self , model_name , validation_loss , training_loss , test_loss):
        """
        Results visualization
        """
        plt.figure()
        plt.plot(validation_loss, label='validation_loss', linewidth=2)
        
        plt.plot(training_loss, label='training_loss', linewidth=2)
        plt.plot(test_loss, label='test_loss', linewidth=2)
        
        plt.legend()
        plt.title(model_name)
        plt.savefig("loss_{}".format(model_name), bbox_inches='tight')
        plt.show()
        print("  ")
        

    def train(self, data , embed , freq , batch_size , root_path ,data_path , target , num_workers , model_name):
        path = os.path.join('checkpoints', model_name)
        if not os.path.exists(path):
            os.makedirs(path)


        train_data, train_loader = self.data_provider(data, 'train' , embed , freq , batch_size , root_path ,data_path , target , num_workers )
        vali_data, vali_loader = self.data_provider(data, 'val' , embed , freq , batch_size , root_path ,data_path , target , num_workers)
        test_data, test_loader = self.data_provider(data, 'test' , embed , freq , batch_size , root_path ,data_path , target , num_workers)


        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=3, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_losses = []
        valid_losses = []
        test_losses = []
        for epoch in range(10):
            iter_count = 0
            train_loss = []


            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder

                outputs = self.model(batch_x)

                # print(outputs.shape,batch_y.shape)
                #forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
                f_dim = -1 if self.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                import sys
                old_stdout = sys.stdout

                log_file = open("message.log","w")

                sys.stdout = log_file


                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((10 - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                sys.stdout = old_stdout

                log_file.close()
                loss.backward()
                model_optim.step()
            
    

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            train_losses.append(train_loss)
            valid_losses.append(vali_loss)
            test_losses.append(test_loss)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            self.adjust_learning_rate(model_optim, epoch + 1)
          
        self.visual_loss( model_name , valid_losses , train_losses , test_losses)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder


                outputs = self.model(batch_x)

                f_dim = -1 if self.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, model_name,  data , embed , freq , batch_size , root_path ,data_path , target , num_workers , test=0 ):
        test_data, test_loader = self.data_provider(data, 'test' , embed , freq , batch_size , root_path ,data_path , target , num_workers )

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + model_name, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + model_name + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                outputs = self.model(batch_x)
                
                f_dim = -1 if self.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    self.visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))


        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        folder_path = './results/' + model_name + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(model_name + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'pred.npy', preds)
        return


    def predict(self, model_name,  data,  embed , freq , batch_size , root_path ,data_path , target , num_workers, load=False  ):
        pred_data, pred_loader = self.data_provider(data, 'pred' , embed , freq , batch_size , root_path ,data_path , target , num_workers )

        if load:
            path = os.path.join('checkpoints', model_name)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                outputs = self.model(batch_x)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        

        preds = np.concatenate(preds, axis=0)
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds.reshape(-1, preds.shape[-1]))
            print(pred_data.shape)

        # result save
        folder_path = './results/' + model_name + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return

      