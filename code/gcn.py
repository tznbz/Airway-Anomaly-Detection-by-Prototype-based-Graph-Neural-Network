import os
import numpy as np
import nibabel as nib
import argparse
import torch
import matplotlib.pyplot as plt
from scipy import ndimage, misc
from matplotlib.collections import PatchCollection
from skimage.morphology import erosion, dilation, cube
from graph_tool import *


class GCN(torch.nn.Module):
    def __init__(self,D_in, H, D_out):
        super(GCN, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        #print('D_in, H, D_out',D_in, H, D_out)
        self.conv1 =torch.nn.Conv1d(D_in, H,1)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv1d(H, D_out,1)
        #laplacian
        edge = np.zeros([24,24])
        edge[18,[19,20,21,22,23]]=1
        edge[19,[0,1,2]]=1
        edge[20,[3,4]]=1
        edge[21,[5,6,7,8,9]]=1
        edge[22,[10,11,12,13]]=1
        edge[23,[14,15,16,17]]=1
        Dmatrix = np.zeros([24,24]).astype(np.float32)
        Amatrix = edge.astype(np.float32)
        Dmatrix[18,18] = 5
        Dmatrix[19,19] = 3
        Dmatrix[20,20] = 2
        Dmatrix[21,21] = 5
        Dmatrix[22,22] = 4
        Dmatrix[23,23] = 4
        #for i in range(18):
        #    Dmatrix[i,i] = 1
        lI= np.zeros([24,24]).astype(np.float32)
        for i in range(24):
            lI[i,i] = 1
        #lmatrix = Dmatrix-Amatrix
        #lmatrix = lI# - lmatrix/7
        lmatrix = lI+Dmatrix+Amatrix
        lmatrix = lmatrix/lmatrix.sum(axis=0)
        self.lmatrix = torch.from_numpy(lmatrix).cuda()
    def forward(self, x):
        # Max pooling over a (2, 2) window
        #print(x.dtype,self.lmatrixdtype)
        #x = torch.matmul(x,self.lmatrix)
        #print('x1m',x.shape)
        x = self.sigmoid(self.conv1(x))
        #print('x1',x.shape)
        x = torch.matmul(x,self.lmatrix)
        #print('x2m',x.shape)
        x = self.conv2(x)
        #print('x2',x.shape)
        return x



def input_preprocess(x,xmax=None,xmean = None):
    x = x.transpose([0,2,1]).astype(np.float32)
    if xmax is None:
        xmax = x.max(axis=0)
        xmax[xmax==0] = 1
    x = x / xmax
    if xmean is None:
        xmean = x.mean(axis=0)
    x = x - xmean
    return x,xmax,xmean


m = torch.nn.Dropout(p=0.5)
def augmentation(x):
    f = torch.rand(x.shape).cuda()
    return m(x+f*0.01)

if __name__ == '__main__':
    x = np.load('../data/tr_x.npy')
    y = np.load('../data/tr_y.npy')
    ts_x = np.load('../data/ts_x.npy')
    ts_y = np.load('../data/ts_y.npy')

    x,xmax,xmean = input_preprocess(x)
    ts_x_origin = np.copy(ts_x)
    ts_x,_,_ = input_preprocess(ts_x,xmax,xmean)
    tr_x = x
    tr_y=y.astype(int)
    print('load data y',y.shape,y.sum(),y.sum()/24)


    tr_x = torch.from_numpy(tr_x).cuda()
    tr_y = torch.from_numpy(tr_y).cuda()
    ts_x = torch.from_numpy(ts_x).cuda()
    ts_y = torch.from_numpy(ts_y).cuda()


    N, D_in, H, D_out = tr_x.shape[0], x.shape[1], 8, y.max()+1
    D_out  = 2
    model = GCN(D_in, H, D_out)


    model.cuda()
    model.load_state_dict(torch.load('mlp2_gcnpro.pth'))


    ts_output = model(ts_x)
    ts_pred = torch.argmax(ts_output, dim=1)
    ts_pred, _ = torch.max(ts_pred, dim=1)
    tx = (ts_output).cpu().detach().numpy()
    ts_y = ts_y.cpu().detach().numpy()
    ts_pred = ts_pred.cpu().detach().numpy()
    tss = tx.argmax(axis=1)

    print('test acc', (ts_pred == ts_y).sum(dtype=ts_pred.dtype) / (len(ts_y)), '\n')
    print('test recall', (ts_pred[ts_y==1]).sum(dtype=ts_pred.dtype) / (len(ts_y[ts_y==1])),(ts_pred[ts_y==1]).sum(dtype=ts_pred.dtype) ,'/',len(ts_y[ts_y==1]), '\n')
    print('test sensitivity', (1-ts_pred[ts_y==0]).sum(dtype=ts_pred.dtype) / (len(ts_y[ts_y==0])),(1-ts_pred[ts_y==0]).sum(dtype=ts_pred.dtype),'/',(len(ts_y[ts_y==0])), '\n')


