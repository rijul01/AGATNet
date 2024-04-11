from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from icecream import ic
from datetime import datetime
import glob
import os
from tqdm import tqdm

import multiprocessing
from torch.utils.data import DataLoader


import geopy.distance # to compute distances between stations
import scipy.sparse as sp

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric_temporal.nn import STConv
from torch.nn import Sequential, Linear, Sigmoid, Tanh
from torch_scatter import scatter_add#, scatter_sub  # no scatter sub in lastest PyG
from torch.nn import Parameter

import random
import torch.nn.utils as utils
import torch.optim as optims
import torch.multiprocessing as mp
from torch.nn.utils import weight_norm

from torch.utils.data import Dataset, DataLoader

import cProfile
import torch.nn.init as init
from time import time




" fix random seeds for reproducibility "

SEED = 12101989
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def cleans_GPU():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

cleans_GPU()

class MyMemmapDataset(Dataset):
    def __init__(self, input_path, target_path, time_path, keep_indices, device):
        self.input = np.load(input_path, mmap_mode='r')[:, :, :, keep_indices]
        self.time = np.load(time_path, mmap_mode='r')
        self.labels = np.load(target_path)  # Assuming labels fit easily into memory
        self.keep_indices = keep_indices
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_feature = self.input[idx, :, :, :]
        time_feature = self.time[idx, :, :, :]
        # print(input_feature.shape, time_feature.shape)
        feature = np.concatenate([time_feature, input_feature], axis=2)
        feature_tensor = torch.tensor(feature, dtype=torch.float32, device=self.device)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32, device=self.device)
        return feature_tensor, label_tensor

class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx])
        label = torch.tensor(self.labels[idx])
        return feature.to(device), label.to(device)

#######################################################################################################################################

all_indices = list(range(47))
remove_indices = [5, 6, 9, 13, 15, 19, 27, 28, 31, 34, 40]
keep_indices = [i for i in all_indices if i not in remove_indices]

#################################### Training Data  #################################################

print('#### Preparing Training data ####')
train_input_path = '/tng4/users/rdimri/Pearson_graph_conv1d/seoul/train_input_data_25_v4.npy' #/tng4/users/rdimri/Pearson_graph_conv1d/seoul/train_input_data_183_v3.npy'# '/tng4/users/rdimri/PM2.5-Graphs/sk/train_input_data_183.npy'#
train_target_path = '/tng4/users/rdimri/Pearson_graph_conv1d/seoul/train_target_data_25_v4.npy'# ''/tng4/users/rdimri/PM2.5-Graphs/sk/train_target_data_183.npy'#'
train_time_path = '/tng4/users/rdimri/Pearson_graph_conv1d/train_input_time_25_v3.npy'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = MyMemmapDataset(train_input_path, train_target_path, train_time_path, keep_indices, device = device)
############################################################################################################################################

#################################### Validation Data  #################################################
print('#### Preparing Validation data ####')
val_input_path = '/tng4/users/rdimri/Pearson_graph_conv1d/val_input_data_25_v4.npy'    # '/tng4/users/rdimri/Pearson_graph_conv1d/val_input_data_25_v4.npy'
val_target_path = '/tng4/users/rdimri/Pearson_graph_conv1d/val_target_data_25_v4.npy'
val_time_path = '/tng4/users/rdimri/Pearson_graph_conv1d/val_target_time_25_v3.npy'

val_dataset = MyMemmapDataset(val_input_path, val_target_path, val_time_path, keep_indices, device = device)

############################################################################################################################################

#################################### Test Data  #################################################

print('#### Preparing Test data ####')
test_input_path = '/tng4/users/rdimri/Pearson_graph_conv1d/test_input_data_25_v4.npy'# '/tng4/users/rdimri/Pearson_graph_conv1d/test_input_data_25_v4.npy'
test_target_path = '/tng4/users/rdimri/Pearson_graph_conv1d/test_target_data_25_v4.npy'
test_time_path = '/tng4/users/rdimri/Pearson_graph_conv1d/test_target_time_25_v3.npy'

test_dataset = MyMemmapDataset(test_input_path, test_target_path, test_time_path, keep_indices, device = device)


# #################################### Training Data  #################################################

# print('#### Preparing Training data ####')
# train_input_path = '/tng4/users/rdimri/Pearson_graph_conv1d/seoul/train_input_data_183_v3.npy' #/tng4/users/rdimri/Pearson_graph_conv1d/seoul/train_input_data_183_v3.npy'# '/tng4/users/rdimri/PM2.5-Graphs/sk/train_input_data_183.npy'#
# train_target_path = '/tng4/users/rdimri/PM2.5-Graphs/sk/train_target_data_183.npy'# ''/tng4/users/rdimri/PM2.5-Graphs/sk/train_target_data_183.npy'#'
# train_time_path = '/tng4/users/rdimri/PM2.5-Graphs/sk//train_input_time_183_v3.npy'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_dataset = MyMemmapDataset(train_input_path, train_target_path, train_time_path, keep_indices, device = device)

# ############################################################################################################################################


# #################################### Test Data  #################################################

# print('#### Preparing Test data ####')
# # test_input_path = '/tng4/users/rdimri/Pearson_graph_conv1d/test_input_data_25_v4.npy'# '/tng4/users/rdimri/Pearson_graph_conv1d/test_input_data_25_v4.npy'
# # test_target_path = '/tng4/users/rdimri/Pearson_graph_conv1d/test_target_data_25_v4.npy'
# # test_time_path = '/tng4/users/rdimri/Pearson_graph_conv1d/test_target_time_25_v3.npy'

# test_input_path = '/tng4/users/rdimri/PM2.5-Graphs/sk/test_input_data_183.npy'#'/tng4/users/rdimri/Pearson_graph_conv1d/test_input_data_25_v4.npy'# '/tng4/users/rdimri/Pearson_graph_conv1d/test_input_data_25_v4.npy'
# test_target_path = '/tng4/users/rdimri/PM2.5-Graphs/sk/test_target_data_183.npy'#'/tng4/users/rdimri/Pearson_graph_conv1d/test_target_data_25_v4.npy'
# test_time_path = '/tng4/users/rdimri/PM2.5-Graphs/sk/test_target_time_183_v3.npy'#'/tng4/users/rdimri/Pearson_graph_conv1d/test_target_time_25_v3.npy'

# test_dataset = MyMemmapDataset(test_input_path, test_target_path, test_time_path, keep_indices, device = device)

############################################################################################################################################

#################################### Model  #################################################
"""
Adaptive Graph Attention Network
"""
class PAGAT(nn.Module):
    def __init__(self, device, feature_dim, node_size, dropout = 0.1, alpha = 0.1, concat = True):
        super(PAGAT, self).__init__()
        self.dropout = dropout
        self.in_features = feature_dim
        self.out_features = feature_dim
        self.alpha = 0.05
        self.concat = concat
        self.node_size = node_size
        self.device = device

        self.W = nn.Sequential(nn.Linear(self.in_features, self.in_features, bias=False))
        nn.init.xavier_uniform_(self.W[0].weight, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.A = nn.Sequential(nn.Linear(self.node_size, self.node_size * 4, bias=False),
                               nn.Sigmoid(),#ReLU(),
                               nn.Linear(self.node_size * 4, self.node_size, bias=False))
        nn.init.xavier_uniform_(self.A[0].weight, gain=1.414)
        nn.init.xavier_uniform_(self.A[2].weight, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, h):
        batch_size, node_size, feature_size = h.size()
        Wh = self.W(h.to(dtype=torch.float)) 
        e = self._prepare_attentional_mechanism_input(Wh)
        # adj = torch.ones((batch_size, node_size, node_size), dtype=torch.float, device = self.device)
        adj = e + self.A(e) #
        adj = self.sig(adj)
        
        connection_threshold = 0.5
        adj = torch.where(adj > connection_threshold, adj, torch.tensor(0.0))
        
        zero_vec = -9e15*torch.ones_like(e)
        # print(Wh.shape, e.shape, adj.shape)
        attention = torch.where(e > 0, e, zero_vec)
        attention = F.softmax(attention, dim = 2)
        attention = F.dropout(attention, self.dropout, training  = self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):         
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MultiHeadAttention(nn.Module):
    def __init__(self, device, feature_dim, node_size, n_heads, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.device = device
        self.attentions = [PAGAT(device, feature_dim = feature_dim, node_size = node_size, dropout = dropout, alpha = 0.1, concat = True)
                           for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            
        self.out_att = PAGAT(device, feature_dim = feature_dim * n_heads, node_size = node_size, dropout = dropout, alpha = 0.1, concat = False)  
        self.fc = nn.Linear(feature_dim * n_heads, feature_dim, bias=False)

    def forward(self, x):
        x = x.to(self.device)
        multi_head = torch.cat([att(x) for att in self.attentions], dim=-1)
        multi_head = F.elu(self.out_att(multi_head)) # 
        multi_head = self.fc(multi_head)
        return multi_head                                                                                       

########################################################################################   

"""
Temporal Convolution
"""

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Dilated Causal Convolution Block
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout = 0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride = stride, padding = padding, dilation = dilation))
        self.chomp1 = Chomp1d(padding)
        self.sig1 = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
        #                                    stride = stride, padding = padding, dilation = dilation))
        # self.chomp2 = Chomp1d(padding)
        # self.relu1 = nn.Sigmoid()
        # # self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.sig1, self.dropout1)#,
                                #  self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.sig1(x + self.net(x.clone()))# self.relu(self.net(x.clone()))

########################################################################################
    
class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DenseLayer, self).__init__()
        self.fc3 = nn.Linear(in_dim, 264)
        self.fc4 = nn.Linear(264, out_dim)

    def forward(self, x):
        x = self.fc3(x) # F.relu(self.fc3(x))
        return self.fc4(x) # F.relu(self.fc4(x))
    
    
class MRCNN(nn.Module):
    def __init__(self):
        super(MRCNN, self).__init__()
        
        self.dropout = nn.Dropout(0.2)
        self.final_dense = DenseLayer(in_dim = (32 + 24) + 1, out_dim = 1)

    def forward(self, x):                                       # torch.Size([9600, 792])
        x = x.unsqueeze(1)                      #  torch.Size([9600, 32, 1566]) 
        out = x.view(x.size(0), -1)                       # torch.Size([9600, 50112])
        out = self.final_dense(out)  # self.dropout()
        return out


class PastSpatialTemporalBlock(nn.Module):
    def __init__(self, device, node_size, pred_len, hist_len, batch_size, 
                 obs_feature, meteo_cmaq_feature, hidden_dim):
        super(PastSpatialTemporalBlock, self).__init__()
        self.device = device
        self.node_size = node_size
        self.pred_len = pred_len
        self.hist_len = hist_len
        self.batch_size = batch_size
        self.obs_feature = obs_feature
        self.meteo_cmaq_feature = meteo_cmaq_feature
        self.hidden_dim = hidden_dim
        self.kernel_size = 6
        self.num_channels = [38, 38, 38, 38]
        self.num_hidden = len(self.num_channels)
        self.residual = None
        
        
        # self.conv1d = ConvNet(self.obs_feature, self.obs_feature, 3)
        self.tcn_model = []
        for i in range(self.num_hidden):
            dilation_size = 2 ** i
            in_channels = 38 if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            self.tcn_model.append(
                TemporalBlock(in_channels, out_channels, self.kernel_size, stride = 1, dilation = dilation_size,
                              padding = (self.kernel_size - 1) * dilation_size, dropout = 0.1)) # type: ignore
        self.tcn_model_list = nn.ModuleList([*self.tcn_model]).to(self.device)
        self.pgat_layer = nn.ModuleList([
            MultiHeadAttention(self.device, 38, self.node_size, 2) for _ in range(1)
        ])
        # self.pgat = MultiHeadAttention(self.device, 47, node_size, 2).to(self.device)
        self.batch_norm = nn.BatchNorm2d(25)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Sequential(
            nn.Linear(38 * 2 * 24, 256 * 8),
            nn.Sigmoid(), # nn.ReLU(),
            nn.Linear(256 * 8, 72) # 
            )
        

    def forward(self, xf_past):                                  # torch.Size([128, 24, 25, 49])
        "Reshaping to get features * historical length"
        xf_past_re = xf_past.permute(0, 2, 3, 1)  # torch.Size([128, 25, 49, 24])
        "Applying Adaptive Multi-head Graph Attention Block with Residual"
        xf_past_graph = torch.zeros_like(xf_past_re)
        for time in range(xf_past_re.size(3)):
            per_time = xf_past_re[:, :, :, time]
            for pgat in self.pgat_layer:
                per_time = pgat(per_time) + per_time
            xf_past_graph[:, :, :, time] = per_time# + xf_past_re[:, :, :, time]         # self.pgat(xf_past_re[:, :, :, time])
            # xf_past_graph[:, :, :, time] = self.pgat(xf_past_graph[:, :, :, time])
            
        # xf_past_graph = xf_past_graph + self.residual
            
        # xf_past_graph = xf_past_re 
        "Applying Dilated Convolution 1D to Capture Temporal Features"
        xf_past_temporal = torch.zeros_like(xf_past_graph)
        for sta in range(xf_past_graph.size(1)):
            single_sta = xf_past_graph[:, sta, :, :]
            for model in self.tcn_model_list:  
                single_sta = model(single_sta)# + single_sta                           # torch.Size([1600, 94, 24])
            xf_past_temporal[:, sta, :, :] = single_sta# + xf_past_graph[:, sta, :, :]
        
                                                              
        xf_past_graph_temporal = torch.cat([xf_past_graph, xf_past_temporal], dim = 2).to(self.device)                      # torch.Size([64, 25, 24, 24])
        
        xf_past_graph_temporal = xf_past_graph_temporal.reshape(xf_past_graph_temporal.size(0), xf_past_graph_temporal.size(1), 
                                                                xf_past_graph_temporal.size(2) * xf_past_graph_temporal.size(3))
        
        xf_past_graph_temporal = self.linear(xf_past_graph_temporal)          # torch.Size([64, 25, 72])
        return xf_past_graph_temporal
    

class PGATConvModel(nn.Module):
    def __init__(self, device):
        super(PGATConvModel, self).__init__()
        self.device = device
        self.hist_len = 24
        self.layers = nn.ModuleList([])
        channels = np.array([[64, 32, 64],[6, 16, 32]])
        self.node_size = 183
        self.pred_len = 72
        self.hist_len = 24
        self.residual = None
        self.kernel_size = 6
        self.num_channels = [32, 32, 32, 32]
        self.num_hidden = len(self.num_channels)
        
        self.layers = PastSpatialTemporalBlock(self.device, 183, 72, 24, 4, channels[1, 0], 6, channels[0, 0])
        
        self.tcn_model_f = []
        for i in range(self.num_hidden):
            dilation_size = 2 ** i
            in_channels = 32 if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i] 
            self.tcn_model_f.append(
                TemporalBlock(in_channels, out_channels, self.kernel_size, stride = 1, dilation = dilation_size,
                              padding = (self.kernel_size - 1) * dilation_size, dropout = 0.1)) # type: ignore
        self.tcn_model_list_f = nn.ModuleList([*self.tcn_model_f])

        self.graph_layer = nn.ModuleList([
            MultiHeadAttention(self.device, (32 + 24) + 1, self.node_size, 2) for _ in range(1)
        ])
        # self.pgat_f = MultiHeadAttention(self.device, (41 + 144) + 1, self.node_size, 2)
        # self.conv1 = nn.Conv1d(41, 64, 3, stride = 1, padding = 1)
        self.relu = nn.ReLU()
        self.sigu = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(25)
        self.dropout = nn.Dropout(0.1)
        self.mrcnn = MRCNN()

    
    def forward(self, feature):
        feature = feature.float()                                           # (batch_size, seq_length, node, num_features)
        """
        Extracting 72-hrs PM2.5 values using Temporal and Graph Network- Past observation data 
        """
        xf_past = feature[:, :self.hist_len, :, :]                          # torch.Size([batch_size, 24, 25, 33])
        xf_past_graph_temporal = self.layers(xf_past) # self.batch_norm(self.layers(xf_past))                                       # torch.Size([64, 25, 72]) 
        
        """
        Extracting 72-hrs features using Dilated convolution block - Model Input
        """
        xf_forward = feature[:, self.hist_len:, :, :32].permute(0, 2, 3, 1)        # torch.Size([batch_size, node, features, time = 72])
        xf_forward_temporal = torch.zeros_like(xf_forward)
        for sta in range(xf_forward_temporal.size(1)):
            single = xf_forward[:, sta, :, :]
            for model in self.tcn_model_list_f:  
                single = model(single)# + single                            
            xf_forward_temporal[:, sta, :, :] = single# + xf_forward[:, sta, :, :]                # torch.Size([128, 25, 41, 72])
        """
        PM2.5 prediction
        """
        xo_past = feature[:, :self.hist_len, :, -1].permute(0, 2, 1)#, 3)                                # torch.Size([128, 25, 24, 6])
        # xo_past = xo_past.reshape(xo_past.size(0), xo_past.size(1), xo_past.size(2) * xo_past.size(3))   # torch.Size([128, 25, 144])
        pm25 = []
        for time in range(72):
            xf_past_pred = xf_past_graph_temporal[:, :, time]     
            xf = xf_forward_temporal[:, :, :, time] # xf_forward_temporal[:, :, :, time]
            x = torch.cat((xo_past, xf, xf_past_pred[:, :, np.newaxis]), dim = 2) # torch.cat((xo_past, xf, xf_past_pred[:, :, np.newaxis]), dim = 2)            # torch.Size([128, 25, 177])
            xn_gnn = x
            for layer in self.graph_layer:
                xn_gnn = layer(xn_gnn) + xn_gnn # self.dropout(self.pgat_f(x))
            # xn_gnn = xn_gnn + self.residual
            xn = xn_gnn.reshape(xn_gnn.size(0) * xn_gnn.size(1), xn_gnn.size(2))
            xn = self.relu(self.mrcnn(xn))
            xn = xn.reshape(x.size(0), x.size(1), xn.size(1))
            xo_past = torch.cat((xo_past[:, :, 1:], xn), dim = 2)
            pm25.append(xn)
        pm25 = torch.stack(pm25, dim = 1)   # torch.Size([128, 25, 24])
        return torch.squeeze(pm25)
    
############################################################################################################################################

#################################### Training the model and loss function  #################################################

def RMSELoss(yhat, y):
    rmse = torch.sqrt(torch.mean((y - yhat) ** 2))
    # y_range = torch.max(y) - torch.min(y)
    # nrmse = rmse / y_range
    return rmse

class custom_loss(nn.Module):
    def __init__(self):
        super(custom_loss, self).__init__()
    
    def forward(self, predicted, target):
        num = torch.sum((predicted - target)**2)
        den = torch.sum((torch.abs(predicted-torch.mean(target))+torch.abs(target-torch.mean(target)))**2)
        ioa = 1 - num / den
        return -ioa


# best_test_loss = float('inf')
# early_stopping_counter = 0



def train_epoch(epoch, model, train_dataloader, criterion2, optimizer, device):
    model.train()
    train_running_loss = 0.0
    train_acc = 0.0
    hist_len = 24

    #################################### Training Step  ################################################### 
    for i, data in enumerate(train_dataloader):
        feature, pm25 = data
        pm25_label = np.squeeze(pm25[:, hist_len:]).to(device)                # torch.Size([128, 25, 72])
        feature = feature.to(device)
        optimizer.zero_grad()
        outputs = model(feature)    # , xf_past_pred
        loss = criterion2(outputs, pm25_label)# + criterion2(xf_past_pred, pm25_label)
        loss.backward()
        optimizer.step() 
        train_running_loss += loss.item()
        train_acc += -criterion2(outputs, pm25_label).type(torch.float).mean().item()
        
    train_loss = train_running_loss / len(train_dataloader)
    train_accuracy = train_acc / len(train_dataloader)
    return train_loss, train_accuracy


def validate_epoch(epoch, model, val_dataloader, criterion2, device):
    torch.cuda.empty_cache()
    model.eval()
    val_running_loss = 0.0
    val_acc = 0.0
    predictionsv = []
    labelsv = []
    hist_len = 24
    with torch.no_grad():
        for j, data in enumerate(val_dataloader):
            feature, pm25 = data
            pm25_label = np.squeeze(pm25[:, hist_len:]).to(device)
            feature = feature.to(device)
            outputsv = model(feature)              # , xf_past_pred
            loss = criterion2(outputsv, pm25_label)# + criterion2(xf_past_pred, pm25_label)
            val_running_loss += loss.item()
            val_acc += -criterion2(outputsv, pm25_label).type(torch.float).mean().item()
            pred = outputsv.cpu().detach().numpy()
            label = pm25_label.cpu().detach().numpy()
            predictionsv.append(pred)
            labelsv.append(label)
            
    val_loss = val_running_loss / len(val_dataloader)
    val_accuracy = val_acc / len(val_dataloader)
    predictv = np.concatenate(predictionsv, axis=0)
    labelv = np.concatenate(labelsv, axis=0)
    return val_loss, val_accuracy, predictv, labelv
    
def test_epoch(epoch, model, test_dataloader, criterion2, device):
    torch.cuda.empty_cache()
    test_running_loss = 0.0
    test_acc = 0.0
    model.eval()
    predictions = []
    labels = []
    hist_len = 24
    with torch.no_grad():
        for b,data in enumerate(test_dataloader):
            feature, pm25 = data
            feature = feature.to(device)
            pm25_label = pm25
            pm25_label = np.squeeze(pm25[:, hist_len:]).to(device)
            outputs = model(feature)                       # , xf_past_pred
            loss = criterion2(outputs, pm25_label)# + criterion2(xf_past_pred, pm25_label)
            test_running_loss += loss.detach().item()
            test_acc += -criterion2(outputs, pm25_label).type(torch.float).mean().item()
            
            pred = outputs.cpu().detach().numpy()
            label = pm25_label.cpu().detach().numpy()
            predictions.append(pred)
            labels.append(label)
            
    test_loss = test_running_loss / len(test_dataloader)
    test_accuracy = test_acc / len(test_dataloader)   
    predict = np.concatenate(predictions, axis=0)
    label = np.concatenate(labels, axis=0)
    return test_loss, test_accuracy, predict, label


def main(num_epochs, model, train_dataset, test_dataset, criterion2, optimizer, device):
    patience = 5
    best_test_loss = float('inf')
    early_stopping_counter = 0
    num_processes = multiprocessing.cpu_count()
    
    for epoch in range(num_epochs):
        start_time = time()
        train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
        # del train_dataset
        with tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", 
                  unit="batch", bar_format="{desc} |{bar}| {percentage:3.0f}%") as train_tqdm:
            train_loss, train_accuracy = train_epoch(epoch, model, train_tqdm, #dataloader, 
                                                                                 criterion2, optimizer, device)
            elapsed_time = time() - start_time
            train_tqdm.set_postfix({'Time per iteration': f'{elapsed_time / len(train_tqdm):.2f}s',
                                    'Elapsed Time': f'{elapsed_time:.2f}s'})
            train_tqdm.refresh()
            
        # start_time = time()
        # with tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}", 
        #           unit="batch", bar_format="{desc} |{bar}| {percentage:3.0f}%") as val_tqdm:
        #     val_loss, val_accuracy, predv, labelv = validate_epoch(epoch, model, val_tqdm, criterion2, device)
        #     elapsed_time = time() - start_time
        #     val_tqdm.set_postfix({'Time per iteration': f'{elapsed_time / len(val_tqdm):.2f}s',
        #                             'Elapsed Time': f'{elapsed_time:.2f}s'})
        #     val_tqdm.refresh()

        start_time = time()
        test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = False)
        # del test_dataset
        with tqdm(test_dataloader, desc=f"Test Epoch {epoch + 1}/{num_epochs}", 
                  unit="batch", bar_format="{desc} |{bar}| {percentage:3.0f}%") as test_tqdm:
            test_loss, test_accuracy, pred, label = test_epoch(epoch, model, test_tqdm, criterion2, device)
            elapsed_time = time() - start_time
            test_tqdm.set_postfix({'Time per iteration': f'{elapsed_time / len(test_tqdm):.2f}s',
                                    'Elapsed Time': f'{elapsed_time:.2f}s'})
            test_tqdm.refresh()


        print('Epoch: %d | Train Loss: %.4f | Train Accuracy: %.2f | | Test Loss: %.4f | Test Accuracy: %.2f|' \
            %(epoch, train_loss, train_accuracy, test_loss, test_accuracy))
        
        # check for early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), '/tng4/users/rdimri/Pearson_graph_conv1d/pear_graph_temporal_183.pth')
            # np.save('/tng4/users/rdimri/Pearson_graph_conv1d/seoul/attention_25_v2_0.npy', attention_outputs)
            np.save('/tng4/users/rdimri/Pearson_graph_conv1d/test_predict_183_v2_3.npy', pred)
            np.save('/tng4/users/rdimri/Pearson_graph_conv1d/test_label_183_v2_3.npy', label)
            # np.save('/tng4/users/rdimri/Pearson_graph_conv1d/val_predict_25_v2_1.npy', predv)
            # np.save('/tng4/users/rdimri/Pearson_graph_conv1d/val_label_25_v2_1.npy', labelv)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print('Early stopping at epoch %d' % epoch)
                break
if __name__ == '__main__':
    num_epochs = 100
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = PGATConvModel(device).to(device)
    criterion2 = custom_loss()
    criterion1 = RMSELoss
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.00001) # Adam
    main(num_epochs, model, train_dataset, test_dataset, criterion2, optimizer, device)   
    # def __init__(self, device, feature_dim, node_size, dropout = 0.2):
    #     super(PAGAT, self).__init__()
    #     self.feature_dim = feature_dim
    #     self.node_size = node_size
    #     self.dropout = nn.Dropout(dropout)
    #     self.k = 10
        
    #     self.W_h = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
    #     nn.init.xavier_uniform_(self.W_h.weight, gain=1.414)
        
    #     self.beta = Parameter(torch.Tensor(1).uniform_(0, 1), requires_grad=True)
    #     # nn.init.xavier_uniform_(self.beta.data, gain=1.414)
        
        
    #     self.A = nn.Sequential(nn.Linear(self.node_size, self.node_size*2, bias=False),
    #                            nn.ReLU(),
    #                            nn.Linear(self.node_size*2, self.node_size, bias=False))
    #     nn.init.xavier_uniform_(self.A[0].weight, gain=1.414)
    #     nn.init.xavier_uniform_(self.A[2].weight, gain=1.414)
        
    #     self.relu = nn.LeakyReLU()
        

    # def mycorr(self, x):
    #     mean_x = x.mean(dim = -1, keepdim = True)
    #     centered_x = x - mean_x
    #     std_x = x.std(dim = -1, keepdim = True)
    #     bessel_correction = x.shape[-1] - 1
    #     covariance = torch.matmul(centered_x, centered_x.transpose(-1, -2))
    #     p = covariance / bessel_correction / (torch.matmul(std_x, std_x.transpose(-1, -2)))
    #     return p

    # def forward(self, h):
    #     batch_size, node_size, feature_size = h.size()
    #     Wh = self.W_h(h.float())                               # torch.Size([128, 25, 47]) 
    #     corr_p = torch.ones((batch_size, node_size, node_size), dtype=torch.float)# self.relu(Wh)
    #     # corr_p = self.mycorr(Wh).to(device)                                     # torch.Size([64, 25, 25])
    #     "Top k correlation values"
    #     # mask = corr_p > 0.7
    #     # corr_p = corr_p * mask
    #     # values, indices = torch.topk(corr_p, self.k, dim=-1, largest=True)
    #     # mask = torch.zeros_like(corr_p).to(device)
    #     # mask.scatter_(-1, indices, corr_p.gather(-1, indices))
    #     # e = self.beta * torch.absolute(corr_p)                             # torch.Size([64, 25, 25])               
    #     # "Adaptive Graph Structure Learning"
    #     adj = self.A(corr_p.to(device)) # e + 
    #     adj = self.relu(adj)                                                        # torch.Size([64, 25, 25])
    #     zero_vec = -1e12 * torch.ones_like(adj)
    #     # "Construct the connections for the nodes whose values greater than 0."
    #     attention = torch.where(adj > 0, adj, zero_vec)
    #     attention = torch.softmax(adj, dim=-1)
    #     threshold = 0.04
    #     masked_attention_matrix = (attention > threshold).float()
    #     thresholded_attention_matrix = attention * masked_attention_matrix
    #     # attention = attention # self.dropout(attention)
    #     # attention = attention # self.dropout(attention)                                                     # torch.Size([64, 25, 25])
    #     h_prime = torch.matmul(thresholded_attention_matrix, Wh)
    #     return h_prime#, attention