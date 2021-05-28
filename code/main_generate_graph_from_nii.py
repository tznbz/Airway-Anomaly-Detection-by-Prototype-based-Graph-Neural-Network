import nibabel as nib
import numpy as np
import os
from graph_tool import *#get_counted_skeleton,extract_graph_from_skeleton,draw_graph_tomatrix,graph2ete3, remove_noise1,combine_singledge1, render_graph, label_graph
#from ete3 import Tree

def label_graph_by_matrix(root,sk_new):
    """ label the graph based on the matrix.
    Args:
        root (Branch object )
        sk_new (the data matrix )
    Returns:
    """
    nodestack = [root]
    indexstack = [0]
    while nodestack:
        node = nodestack[-1]
        index = indexstack[-1]
        if index==0:
            node.label = getvalue(sk_new,node.position.astype(int))
        if index < len(node.edges):
            nodestack += [node.edges[index].endbracnch]
            indexstack[-1] += 1
            indexstack += [0]
        else:
            nodestack.pop()
            indexstack.pop()


def label_graph_to_matrix(root):
    """ get the nodes to a list per label
    Args:
        root (Branch object )
    Returns:
        list_by_label (the listed nodes )
    """
    list_by_label = [[] for i in range(25)]
    nodestack = [root]
    indexstack = [0]
    while nodestack:
        node = nodestack[-1]
        index = indexstack[-1]
        if index==0:
            #print('node.label',node.label)
            list_by_label[int(node.label)] += [node]
        if index < len(node.edges):
            nodestack += [node.edges[index].endbracnch]
            indexstack[-1] += 1
            indexstack += [0]
        else:
            nodestack.pop()
            indexstack.pop()
    return list_by_label

def branchase_label_transfer(data):
    data2 = np.zeros_like(data)
    data2[data==1] = 19
    data2[data==3] = 20
    data2[data==13] = 1
    data2[data==23] = 2
    data2[data==33] = 3
    data2[data==6] = 21
    data2[data==16] = 4
    data2[data==26] = 5
    data2[data==7] = 22
    data2[data==17] = 6
    data2[data==27] = 7
    data2[data==37] = 8
    data2[data==47] = 9
    data2[data==57] = 10
    data2[data==8] = 23
    data2[data==18] = 23
    data2[data==118] = 11
    data2[data==218] = 12
    data2[data==28] = 23
    data2[data==128] = 13
    data2[data==228] = 14
    data2[data==9] = 24
    data2[data==19] = 15
    data2[data==29] = 18
    data2[data==39] = 17
    data2[data==49] = 16
    data[:] = data2
    print('data2',data.max())
    return data

maskdir = '../data/train/segment_mask_revised_by_Xiaohan_processed_Tianyiprocessed/'
resdir =  '../data/train/graph_features/'
maskdir = '../data/test/nii/'
resdir =  '../data/test/graph_features/'
maskdir = '../data/test/anomaly2_labeled/'
resdir =  '../data/test/anomaly2_graph_features/'

if not os.path.exists(resdir):
    os.mkdir(resdir)
filenames = os.listdir(maskdir)
#filenames = [f for f in filenames if f.startswith('FPFNngs2_th0.3')]
for f in filenames:
    print('processing',f)
    volume = nib.load(maskdir+f)
    data = volume.get_data()
    if data.max()>25:
        data = data%1000
        data = branchase_label_transfer(data)


    print('extracting skeleton')
    sk = get_counted_skeleton(data)
    #data[:] = sk
    #img = nib.Nifti1Image(sk, volume.affine)
    #nib.save(img,resdir+'newsk_'+f)
    print('extracting graph')
    root = extract_graph_from_skeleton(sk)
    #root.print_subtree()
    root.update_bottonuprank()
    remove_noise1(root)
    combine_singledge1(root)
    render_graph(root)
    label_graph_by_matrix(root, data)
    #label_graph(root)
    print('writing new result matrix')
    #sk_new = draw_graph_tomatrix(root,data.shape)
    #print('saving')
    #data[:] = sk_new
    #img = nib.Nifti1Image(sk_new, volume.affine)
    #nib.save(img,resdir+'labeltree_'+f)
    #nib.save(volume,resdir+'labeltree_'+f)
    list_by_label = label_graph_to_matrix(root)
    list_by_label = [sorted(l, key=lambda x: x.rank) if l else [] for l in list_by_label]
    features = np.zeros([24,10])
    for i in range(24):
        if list_by_label[i+1]:
            features[i][:3] = list_by_label[i+1][0].position
            features[i][3:6] = tree_direction(list_by_label[i+1][0])
            features[i][6] = list_by_label[i+1][0].rank
            features[i][7] = list_by_label[i+1][0].bottomuprank
            features[i][8] = list_by_label[i+1][0].get_treelength()
            features[i][9] = len(list_by_label[i+1])
    print(f,features[0])
    np.save(resdir+f,features)

    #t = graph2ete3(root)
    #t = Tree(graph2ete3(root), format=1)
    #with open(resdir+'airwayete3labeltree_'+f[:-4]+'.txt','w') as f:
    #    print(t.get_ascii(),file=f)
