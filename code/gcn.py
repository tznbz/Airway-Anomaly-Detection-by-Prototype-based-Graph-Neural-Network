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
        #Dmatrix =
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
        #x = torch.matmul(x,self.lmatrix)
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
    #y[y>3] = 3
    return x,xmax,xmean


m = torch.nn.Dropout(p=0.5)
def augmentation(x):
    f = torch.rand(x.shape).cuda()
    #print(x.mean())
    #print(f.mean(),f.shape)
    return m(x+f*0.01)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset", required=False, default=2, type=int)

    args = parser.parse_args()
    dataset = args.dataset
    if dataset == 1:

        featuredir = '../data/train/graph_features/'
        filenames = os.listdir(featuredir)
        abnomalydir = '../data/test/abnomaly/'
        abnomalnames = os.listdir(abnomalydir)
        abnomalnames = [ab+'.npy' for ab in abnomalnames]
        features = []
        labels = []
        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            if f in abnomalnames:
                labels += [1]
            else:
                labels += [0]

        x = np.asarray(features)
        #y = np.zeros(len(x),dtype=int)
        y = np.asarray(labels).astype(int)
        print('train ;label', labels)

        featuredir = '../data/test/graph_features/'
        featuredir = '../data/test/anomaly2_graph_features/'
        filenames = os.listdir(featuredir)
        abnomalydir = '../data/test/abnomaly/'
        abnomalydir = '../data/test/anomaly2_labeled/'
        abnomalnames = os.listdir(abnomalydir)
        abnomalnames = [ab+'.npy' for ab in abnomalnames]
        features = []
        labels = []
        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            if f in abnomalnames:
                labels += [1]
            else:
                labels += [0]
        print('test ;label', labels)
        ts_x = np.asarray(features)
        ts_y = np.asarray(labels).astype(int)
        filenames = np.asarray(filenames)
        ts_x2 = ts_x[ts_y==1]
        labels1 = len(ts_x2)
        filenames2 = filenames[ts_y==1]
        ts_x2 = np.concatenate([ts_x2, ts_x[ts_y==0][:22]])
        filenames2 = np.concatenate([filenames2, filenames[ts_y==0][:22]])
        ts_x = ts_x2
        ts_y = np.zeros([len(ts_x2)],dtype=int)
        ts_y[:labels1] = 1
        filenames = filenames2


    if dataset == 2:
        featuredir = '../data/train/graph_features/'
        filenames = os.listdir(featuredir)
        abnomalydir = '../data/test/abnomaly/'
        abnomalnames = os.listdir(abnomalydir)
        abnomalnames = [ab[23:-4]+'_segment_processed.nii.npy' for ab in abnomalnames]
        features = []
        labels = []
        masks = []
        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            if f in abnomalnames:
                labels += [np.ones(24)]
            else:
                labels += [np.zeros(24)]
            masks += [np.ones(24)]

        print('normal train', len(features))

        '''featuredir = '../data/train/augmentation_shift/'
        filenames = os.listdir(featuredir)
        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            labels += [np.zeros(24)]
            labels12 = f.split('_')[-1][:-4].split('shift')
            labels12 = [int(l) for l in labels12]
            for l in labels12:
                labels[-1][l] = 1
        print('shift train', len(features))'''

        '''featuredir = '../data/train/augmentation_shift/'
        filenames = os.listdir(featuredir)
        abnomalfeatures = [[] for iii in range(24)]
        for f in filenames:
            feature = np.load(featuredir + f)
            #features += [feature]
            #labels += [np.ones(24)]
            labels12 = f.split('_')[-1][:-4].split('shift')
            labels12 = [int(l) for l in labels12]
            for l in labels12:
                #labels[-1][l] = 1
                abnomalfeatures[l] += [feature[l]]
            #masks += [labels[-1]]
        l_abnomalfeatures = [len(abf) for abf in abnomalfeatures]
        print('l_abnomalfeatures',l_abnomalfeatures)
        l_abnomalfeatures = max(l_abnomalfeatures)
        print('l_abnomalfeatures',l_abnomalfeatures)
        abnomalfeaturesnpy = np.zeros([l_abnomalfeatures,24,10])
        for iii in range(18):
            abnomalfeaturesnpy[:len(abnomalfeatures[iii]),iii]  = abnomalfeatures[iii]
        features = np.asarray(features)
        features = np.concatenate([features,abnomalfeaturesnpy])

        labels = np.asarray(labels).astype(int)
        print('abnomalfeaturesnpy', len(abnomalfeaturesnpy),np.ones([len(abnomalfeaturesnpy),24]).shape)
        print('labels.shape', labels.shape)
        labels = np.concatenate([labels,np.ones([len(abnomalfeaturesnpy),24])])
        print('labels.shape', labels.shape)

        print('shift train', len(features))'''

        '''featuredir = '../data/train/augmentation_tochild/'
        filenames = os.listdir(featuredir)
        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            labels += [np.zeros(24)]
            #print(f,f.split('_')[-1][:-4])
            labels12 = f.split('_')[-1][:-4].split('tochild')
            labels12 = [int(l) for l in labels12 if l != '']
            for l in labels12:
                labels[-1][l] = 1
        print('tochild train', len(features))'''
        '''featuredir = '../data/train/augmentation_tochild/'
        filenames = os.listdir(featuredir)
        abnomalfeatures = [[] for iii in range(24)]
        for f in filenames:
            feature = np.load(featuredir + f)
            # print(f,f.split('_')[-1][:-4])
            labels12 = f.split('_')[-1][:-4].split('tochild')
            labels12 = [int(l) for l in labels12 if l != '']
            for l in labels12:
                abnomalfeatures[l] += [feature[l]]

        l_abnomalfeatures = [len(abf) for abf in abnomalfeatures]
        l_abnomalfeatures = max(l_abnomalfeatures)
        abnomalfeaturesnpy = np.zeros([l_abnomalfeatures,24,10])
        for iii in range(18):
            abnomalfeaturesnpy[:len(abnomalfeatures[iii]),iii]  = abnomalfeatures[iii]
        features = np.asarray(features)
        features = np.concatenate([features,abnomalfeaturesnpy])

        labels = np.asarray(labels).astype(int)
        labels = np.concatenate([labels,np.ones([len(abnomalfeaturesnpy),24])])
        print('tochild train', len(features))'''

        '''featuredir = '../data/train/augmentation_cutbranch/'
        filenames = os.listdir(featuredir)
        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            labels += [np.ones(24)]#
            labels12 = f.split('_')[-1][:-4].split('cut')
            labels12 = [int(l) for l in labels12 if l != '']
            for l in labels12:
                labels[-1][l] = 1
        print('cut train', len(features))'''


        '''featuredir = '../data/train/augmentation_cutbranch/'
        filenames = os.listdir(featuredir)
        abnomalfeatures = [[] for iii in range(24)]
        for f in filenames:
            feature = np.load(featuredir + f)
            labels12 = f.split('_')[-1][:-4].split('cut')
            labels12 = [int(l) for l in labels12 if l != '']
            for l in labels12:
                abnomalfeatures[l] += [feature[l]]
        l_abnomalfeatures = [len(abf) for abf in abnomalfeatures]
        print('l_abnomalfeatures',l_abnomalfeatures)
        l_abnomalfeatures = max(l_abnomalfeatures)
        abnomalfeaturesnpy = np.zeros([l_abnomalfeatures,24,10])
        for iii in range(18):
            if len(abnomalfeatures[iii])>0:
                abnomalfeaturesnpy[:len(abnomalfeatures[iii]),iii]  = abnomalfeatures[iii]
        features = np.asarray(features)
        features = np.concatenate([features,abnomalfeaturesnpy])

        labels = np.asarray(labels).astype(int)
        labels = np.concatenate([labels,np.ones([len(abnomalfeaturesnpy),24])])
        print('cut train', len(features))'''

        x = np.asarray(features)
        #y = np.zeros(len(x),dtype=int)
        y = np.asarray(labels).astype(int)
        #masks = np.asarray(masks).astype(int)
        #x = x*masks.reshape(list(masks.shape)+[1])
        print('train xxx',x.shape, y.shape)
        #print('train ;label', labels)

        featuredir = '../data/test/graph_features/'
        filenames = os.listdir(featuredir)
        abnomalydir = '../data/test/abnomaly/'
        #abnomalydir = '../data/test/anomaly2_labeled/'
        abnomalnames = os.listdir(abnomalydir)
        abnomalnames = [ab+'.npy' for ab in abnomalnames]
        features = []
        labels = []

        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            if f in abnomalnames:
                labels += [1]
            else:
                labels += [0]

        filenames1 = filenames

        featuredir = '../data/test/anomaly2_graph_features/'
        filenames = os.listdir(featuredir)
        abnomalydir = '../data/test/anomaly2_labeled/'
        abnomalnames = os.listdir(abnomalydir)
        abnomalnames = [ab+'.npy' for ab in abnomalnames]
        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            if f in abnomalnames:
                labels += [1]
            else:
                labels += [0]


        print('test ;label', labels)
        ts_x = np.asarray(features)
        ts_y = np.asarray(labels).astype(int)
        filenames = np.asarray(filenames1+filenames)
        ts_x2 = ts_x[ts_y==1][:15]
        labels1 = len(ts_x2)
        filenames2 = filenames[ts_y==1][:15]
        ts_x2 = np.concatenate([ts_x2, ts_x[ts_y==0][:22]])
        filenames2 = np.concatenate([filenames2, filenames[ts_y==0][:22]])
        ts_x = ts_x2
        ts_y = np.zeros([len(ts_x2)],dtype=int)
        ts_y[:labels1] = 1
        filenames = filenames2


    if dataset == 3:
        featuredir = '../data/train/graph_features/'
        filenames = os.listdir(featuredir)
        abnomalydir = '../data/test/abnomaly/'
        abnomalnames = os.listdir(abnomalydir)
        #abnomalnames = [ab+'.npy' for ab in abnomalnames]
        abnomalnames = [ab[23:-4]+'_segment_processed.nii.npy' for ab in abnomalnames]
        features = []
        labels = []
        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            if f in abnomalnames:
                labels += [np.ones(24)]
            else:
                labels += [np.zeros(24)]

        print('normal train', len(features))
        featuredir = '../data/train/augmentation_shift/'
        filenames = os.listdir(featuredir)
        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            labels += [np.zeros(24)]
            labels12 = f.split('_')[-1][:-4].split('shift')
            labels12 = [int(l) for l in labels12]
            for l in labels12:
                labels[-1][l] = 1
        print('shift train', len(features))

        '''featuredir = '../data/train/augmentation_tochild/'
        filenames = os.listdir(featuredir)
        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            labels += [np.zeros(24)]
            #print(f,f.split('_')[-1][:-4])
            labels12 = f.split('_')[-1][:-4].split('tochild')
            labels12 = [int(l) for l in labels12 if l != '']
            for l in labels12:
                labels[-1][l] = 1
        print('tochild train', len(features))'''

        featuredir = '../data/train/augmentation_cutbranch/'
        filenames = os.listdir(featuredir)
        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            labels += [np.ones(24)]#
            labels12 = f.split('_')[-1][:-4].split('cut')
            labels12 = [int(l) for l in labels12 if l != '']
            for l in labels12:
                labels[-1][l] = 1
        print('cut train', len(features))
        x = np.asarray(features)
        #y = np.zeros(len(x),dtype=int)
        y = np.asarray(labels).astype(int)
        #print('train ;label', labels)

        featuredir = '../data/test/graph_features/'
        filenames = os.listdir(featuredir)
        abnomalydir = '../data/test/abnomaly/'
        #abnomalydir = '../data/test/anomaly2_labeled/'
        abnomalnames = os.listdir(abnomalydir)
        abnomalnames = [ab+'.npy' for ab in abnomalnames]
        features = []
        labels = []

        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            if f in abnomalnames:
                labels += [1]
            else:
                labels += [0]

        filenames1 = filenames

        featuredir = '../data/test/anomaly2_graph_features/'
        filenames = os.listdir(featuredir)
        abnomalydir = '../data/test/anomaly2_labeled/'
        abnomalnames = os.listdir(abnomalydir)
        abnomalnames = [ab+'.npy' for ab in abnomalnames]
        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            if f in abnomalnames:
                labels += [1]
            else:
                labels += [0]


        '''featuredir = '../data/test/augmentation_shift/'
        filenames1 += filenames
        filenames = os.listdir(featuredir)
        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            labels += [1]
        print('shift test', len(features))

        featuredir = '../data/test/augmentation_tochild/'
        filenames1 += filenames
        filenames = os.listdir(featuredir)
        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            labels += [1]
        print('tochild test', len(features))'''

        featuredir = '../data/test/augmentation_cutbranch/'
        featuredir = '../data/test/anomaly2_augmentation_cutbranch/'
        filenames1 += filenames
        filenames = os.listdir(featuredir)
        for f in filenames:
            feature = np.load(featuredir + f)
            features += [feature]
            labels += [1]
        print('cut test', len(features))


        #print('test ;label', labels)
        ts_x = np.asarray(features)
        ts_y = np.asarray(labels).astype(int)
        filenames = np.asarray(filenames1+filenames)
        ts_x2 = ts_x[ts_y==1]
        labels1 = len(ts_x2)
        filenames2 = filenames[ts_y==1]
        ts_x2 = np.concatenate([ts_x2, ts_x[ts_y==0][:22]])
        filenames2 = np.concatenate([filenames2, filenames[ts_y==0][:22]])
        ts_x = ts_x2
        ts_y = np.zeros([len(ts_x2)],dtype=int)
        ts_y[:labels1] = 1
        filenames = filenames2
    print(x.shape)

    #ts_y = np.repeat(ts_y.reshape([-1,1]), 156, axis=1)+2
    #print(vy.shape,'tr_vy',vy.max(axis=1))
    x,xmax,xmean = input_preprocess(x)
    ts_x_origin = np.copy(ts_x)
    ts_x,_,_ = input_preprocess(ts_x,xmax,xmean)
    #print('xmax',xmax.shape,xmax.mean(axis=0))
    #print('xmean',xmean.shape,xmean.mean(axis=0))
    #ts_x = x[-5:]
    #ts_y = y[-5:]
    tr_x = x#[:-5]
    #tr_x = np.concatenate([np.zeros_like(tr_x),tr_x])
    tr_y=y.astype(int)
    if dataset==1:
        tr_y = np.zeros([len(tr_x),24],dtype=int)
        tr_y[y==1] = 1
    #tr_y[:len(tr_y)//3*2] = 0
    val_x = ts_x
    val_y = np.zeros([len(val_x),24],dtype=int)
    #tr_x = ts_x
    #tr_y = np.zeros([len(tr_x),24],dtype=int)
    #tr_y[ts_y==1] = 1

    print('preprocessed y',y.shape,y.sum(),y.sum()/24)
    print('preprocessed ts_y',ts_y,ts_y.shape)


    tr_x = torch.from_numpy(tr_x).cuda()
    tr_y = torch.from_numpy(tr_y).cuda()
    val_x = torch.from_numpy(val_x).cuda()
    val_y = torch.from_numpy(val_y).cuda()
    ts_x = torch.from_numpy(ts_x).cuda()
    ts_y = torch.from_numpy(ts_y).cuda()


    N, D_in, H, D_out = tr_x.shape[0], x.shape[1], 8, y.max()+1
    D_out  = 2
    print('D_out',D_out)
    print('D_in',D_in)
    print('tr_x',tr_x.shape)
    #D_out+=1
    model = GCN(D_in, H, D_out)
    '''model = torch.nn.Sequential(
        torch.nn.Conv1d(D_in, H,1),
        torch.nn.Sigmoid(),
        torch.nn.Conv1d(H, D_out,1),
    )'''

    model.cuda()
    weights = np.ones(2)
    #weights = [0.8,1]
    weights = [1,0.7]
    weights = np.asarray(weights)
    weights = weights/weights.sum()
    class_weights = torch.FloatTensor(weights).cuda()
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    #loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-3)#weight_decay, momentum=0.9
    printFrequence = 1000
    besttsacc = 0.51

    for t in range(10000):
        #tr_xa = augmentation(tr_x)
        y_output = model(tr_x)

        #print('y_output',y_output.shape)
        y_pred = torch.argmax(y_output,dim=1)
        #print('y_pred',y_pred.shape)
        loss = loss_fn(y_output, tr_y)

        ts_output = model(ts_x)
        ts_pred = torch.argmax(ts_output, dim=1)
        #print('ts_pred',ts_pred.shape)
        ts_pred, _ = torch.max(ts_pred, dim=1)
        #print('ts_pred',ts_pred.shape)
        #print('ts_y',ts_y.shape)
        #ts_pred[ts_pred != 2] = 3
        ts_sens = (ts_pred[ts_y==0] == ts_y[ts_y==0]).sum(dtype=torch.float32) / (len(ts_y[ts_y==0]))
        ts_acc = (ts_pred == ts_y).sum(dtype=torch.float32) / (len(ts_y))
        if t % printFrequence == printFrequence-1 or ts_acc>besttsacc:
            print('train',t, loss.item(),(y_pred == tr_y).sum(dtype=torch.float32) / (len(tr_y))/24,tr_y.shape)
            #print('y_pred',torch.max(y_pred, dim=1))
            #print('tr_y',torch.max(tr_y, dim=1))
            val_output = model(val_x)
            val_pred = torch.argmax(val_output,dim=1)
            val_acc = (val_pred == val_y).sum(dtype=torch.float32) / (len(val_y))/24
            print('val acc',(val_pred == val_y).sum(dtype=torch.float32) / (len(val_y))/24)
            #print(ts_pred)
            #print(ts_y)
            #print((ts_pred == ts_y))
            print('test acc,sens',ts_acc,ts_sens,'\n')
            if ts_acc>besttsacc:
                besttsacc = ts_acc
                #break
            #if ts_pred[:5].sum()>0 and ts_pred[:5].sum()<5:# and ts_acc<0.82:
            #    break
            if ts_acc>0.86 and ts_sens>0.86:
                break

        optimizer.zero_grad()
        loss.backward()


        optimizer.step()

    '''mlp_state_dict = model.state_dict()
    for key in mlp_state_dict.keys():
        print('key', key, mlp_state_dict[key].shape)
        mlp_state_dict[key] = mlp_state_dict[key].cpu()

    torch.save({
        'mlp_state_dict': mlp_state_dict},
        'mlp2_gcnpro.model')'''

    print('ts_pred',ts_pred,ts_pred.shape)
    print('test acc', (ts_pred == ts_y).sum(dtype=torch.float32) / (len(ts_y)), '\n')
    for i in range(2,D_out):
        print('test class',i, (ts_pred[ts_y==i] == ts_y[ts_y==i]).sum(dtype=torch.float32) / (len(ts_y[ts_y==i])))


    tx = (ts_output).cpu().detach().numpy()
    #ts_pred = torch.argmax(ts_output,dim=1)
    #tx = ts_pred.cpu().detach().numpy()
    ts_y = ts_y.cpu().detach().numpy()
    ts_pred = ts_pred.cpu().detach().numpy()

    tss = tx.argmax(axis=1)
    tsy_pred_count = np.zeros([len(tss),3])
    print('tss',tss.max(axis=1),tss.shape)
    for i in range(len(tss)):
        for j in range(3):
            tsy_pred_count[i,j] = tss[i][tss[i]==j+2].size

    #ts_pred = tsy_pred_count.argmax(axis=1) + 2
    print('ts_pred y', ts_pred,ts_pred[ts_pred==1])
    print('test y', ts_y)
    print('test acc', (ts_pred == ts_y).sum(dtype=ts_pred.dtype) / (len(ts_y)), '\n')
    print('test recall', (ts_pred[ts_y==1]).sum(dtype=ts_pred.dtype) / (len(ts_y[ts_y==1])),(ts_pred[ts_y==1]).sum(dtype=ts_pred.dtype) ,'/',len(ts_y[ts_y==1]), '\n')
    print('test sensitivity', (1-ts_pred[ts_y==0]).sum(dtype=ts_pred.dtype) / (len(ts_y[ts_y==0])),(1-ts_pred[ts_y==0]).sum(dtype=ts_pred.dtype),'/',(len(ts_y[ts_y==0])), '\n')

    #print(filenames[ts_pred!=ts_y])
    #print(tss[ts_pred!=ts_y])
    for i in range(2,D_out):
        print('test class',i, (ts_pred[ts_y==i] == ts_y[ts_y==i]).sum(dtype=ts_pred.dtype) / (len(ts_y[ts_y==i])))

    print('test y', ts_y[:15])
    print('ts_pred y', ts_pred[:15])
    print(filenames[:15])
    m = torch.nn.Softmax(dim=1)
    ts_softmax = m(ts_output)
    ts_softmax = ts_softmax.cpu().detach().numpy()

    maskdir = '../data/test/nii/'
    maskdir2 = '../data/test/anomaly2_labeled/'
    '''llll = [i for i in range(15)] + [-i-1 for i in range(22)]
    for i in llll:
        if True:# ts_y[i] == ts_pred[i]:
            print(filenames[i])
            print(ts_softmax[i,1])
            f = filenames[i][:-4]
            if os.path.exists(maskdir + f):
                volume = nib.load(maskdir + f)
            else:
                volume = nib.load(maskdir2 + f)
            data = volume.get_data()
            data[data>0] = 1
            data2d = data.max(axis=1).astype(float)
            sk = get_counted_skeleton(data)
            sk[sk>0] = 1
            sk2d = sk.max(axis=1)
            #sk2d  = dilation(sk2d,cube(3)[0])
            data2d[sk2d>0] = 0.5
            result = ndimage.zoom(data2d, [1,volume.affine[2,2]], order=0)
            plt.clf()
            fig, ax = plt.subplots()
            plt.imshow(1-result,cmap='Greys')
            #plt.imshow(1-result,cmap='jet')r
            circles = [plt.Circle((ts_x_origin[i][ci][2]*volume.affine[2,2], ts_x_origin[i][ci][0]), 5) for ci in range(24) if ts_x_origin[i][ci][:3].sum()>0]
            colors = [ts_softmax[i,1][ci] for ci in range(24) if ts_x_origin[i][ci][:3].sum() > 0]
            print('colors',colors)
            col = PatchCollection(circles,  array=np.array(colors),cmap="jet")
            ax.add_collection(col)
            #ax.add_patch(circle1)
            #print('fff','figures/'+f+'.png')
            plt.savefig('figures/'+f+str(ts_y[i])+'_pred'+str(ts_pred[i])+'.png')'''


    #print(tss[ts_pred!=ts_y])
    '''f = ftr_list[ts_pred!=ts_y]
    #for fi,ff in enumerate(f):
    #    print(fi,ff,tss[ts_pred!=ts_y][fi])

    #tss = (tss==2).astype(tss.dtype)
    v_acc = (tss == ts_vy).astype(tss.dtype).mean(axis=0)
    print(tss[ts_y==0].shape)
    v_sens = (tss[ts_y==2] == ts_vy[ts_y==2]).astype(tss.dtype).mean(axis=0)
    v_spec = (tss[ts_y==3] == ts_vy[ts_y==3]).astype(tss.dtype).mean(axis=0)
    #print(v_acc)
    #print(v_sens)
    print('v_acc',v_acc.mean(),(tss[ts_vy==1] == ts_vy[ts_vy==1]).astype(tss.dtype).mean(),(tss[ts_vy==0] == ts_vy[ts_vy==0]).astype(tss.dtype).mean())

    ts_y -= 2
    print('tss',tss.max(axis=1),tss.shape)'''
    '''maskdir = '../data/test/nii/'
    volume = nib.load(maskdir+f)
    data = volume.get_data()
    data[data>0] = 1
    data2d = data.max(axis=1)
    from scipy import ndimage, misc
    result = ndimage.zoom(data2d, [1,volume.affine[2,2]])
    #sk = get_counted_skeleton(data)
    #sk2d = sk.max(axis=1)
    
    from matplotlib.collections import PatchCollection
    plt.clf()
    fig, ax = plt.subplots()
    plt.imshow(1-result,cmap='Greys')
    #plt.imshow(1-result,cmap='jet')r
    circle1 = plt.Circle((164*volume.affine[2,2],211), 10, color='r')
    circle2 = plt.Circle((211, 164), 10, color='r')
    circles = [circle1,circle2]
    col = PatchCollection(circles,  array=np.array([0,1]),cmap="jet")
    ax.add_collection(col)
    #ax.add_patch(circle1)
    plt.savefig('temp.png')'''