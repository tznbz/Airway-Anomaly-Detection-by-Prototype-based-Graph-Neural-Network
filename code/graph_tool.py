import nibabel as nib
import numpy as np
import time
import datetime

from skimage.morphology import erosion,dilation,square,convex_hull_image,cube
from skimage.morphology import skeletonize_3d

from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.filters import convolve
import skimage.filters.rank as rank
from scipy import ndimage

from branch import Branch
from branchEdge import BranchEdge

def get_counted_skeleton(data):
    """ extract skelton from airway segmentation matrix
        and assigned value on each pixel based on the # of pixels around it
        
    Args:
        data (airway segmentation matrix )
    Returns:
        sk (skeleton matrix  )
    """ 
    #closing
    data = dilation(data,cube(3))
    data = erosion(data,cube(3))
    #fill holes
    data = binary_fill_holes(data)
    #skeletonize
    skeleton = skeletonize_3d(data)
    #sum filter
    cound = convolve(skeleton,cube(3))
    sk = skeleton*cound
    return sk


def get_direction26():
    """ get the constant matrix
    """ 
    direction27 = np.zeros([3,3,3,3])
    for i in range(3):
       for j in range(3):
           for k in range(3):
              direction27[i,j,k] = [i-1,j-1,k-1]
    direction27 = direction27.reshape([27,3])
    return np.delete(direction27, 13, 0).astype(int)
DIRECTION26 = get_direction26()
RIGHTUP=3
RIGHTMID=4
RIGHTDOWN=5
LEFTUP=6
LEFTDOWM=7

def getvalue(arr,position):
    """ get the value of matrix 'arr' at 'position'
    """ 
    return arr[position[0],position[1],position[2]]
def setvalue(arr,position,value):
    """ set the value for matrix 'arr' at 'position' as value
    """ 
    arr[position[0],position[1],position[2]] = value
def setvalues(arr,positions,value):
    """ set the value for matrix 'arr' at 'positions' as value
    """ 
    for position in positions:
        arr[position[0],position[1],position[2]] = value

def findroot(sk):
    """ find the root postion
        
    Args:
        sk (skeleton matrix )
    Returns:
        position (position of root )
    """ 
    #find the first none branch pixel from top
    for i in range(sk.shape[-1]-1,-1,-1):
        if 2 in sk[200:400,200:400,i] :
            #if the first pixel found has value 2, return this position
            position = [xi[0]+200 for xi in np.where(sk[200:400,200:400,i]==2)] + [i]
            return np.asarray(position)
        elif 3 in sk[200:400,200:400,i]:#sometimes pixel with value 3 could be an end (need to check)
            position = [xi[0]+200 for xi in np.where(sk[200:400,200:400,i]==3)] + [i]
            break
    assert position,'no root found'
    #pixel at the position has an edge value 3. Follow the skeleton to find the end. 
    sk_used = np.zeros_like(sk)
    sk_unused = np.copy(sk)
    root_position = position
    #root = Branch(pixels=[root_position],name='root')
    setvalue(sk_used,root_position,1)
    setvalue(sk_unused,root_position,0)
    #extract rood edges
    edgelist,branchlist,endlist = next_pixels(root_position,sk_used,sk_unused)#get next pixels
    if endlist:
        return np.asarray(endlist[0])# next pixel is an end. Checked and return
    if len(edgelist)==1:
        return np.asarray(position)# this pixel is an end. Checked and return
    #This pixel is not the end, search the end along the skeleton
    for edgepoint in edgelist:
        rootedge = BranchEdge([edgepoint])
        while True:
            edgelist1,branchlist,endlist = next_pixels(edgepoint,sk_used,sk_unused)
            if edgelist1:
                assert len(edgelist1)==1, '# of unused pixel arround edge pixel should be 1'
                rootedge.add_pixels(edgelist1)
            else:
                if endlist:
                    return np.asarray(endlist[0])
                elif not branchlist:
                    return np.asarray(rootedge.pixels[-1])
                else:
                    break
    
    assert not branchlist,'no root found'
    #we assume that the first position we got must is or is connected to an end
    return np.asarray(position)


def position_inshape(position,shape):
    """ check if the postion is out of shape 
        
    Args:
        position (current position )
        shape (matrix shape )
    Returns:
        True/False (if inside the shape)
    """ 
    if (position<0).any():
        return False
    if ((position-shape)>=0).any():
        return False
    return True

def next_pixels(position,sk_used,sk_unused):
    """ get next pixels from 26 direction
        
        return all pixels around the position. 
        Pixels have 3 type: edgelist = 3; branchlist >=4; endlist =2.
        
    Args:
        position (current position )
        sk_used (include pixels that are passed )
        sk_unused (include pixels that are not passed )
    Returns:
        edgelist (list of edge pixels)  
        branchlist (list of branch pixels)  
        endlist (list of end pixels)  
    """ 
    edgelist = []
    branchlist = []
    endlist = []
    for di in DIRECTION26:
        nposition = position+di
        if not position_inshape(nposition,sk_used.shape):
            continue
        #print(nposition)
        value = getvalue(sk_unused,nposition)
        setvalue(sk_used,nposition,1)
        setvalue(sk_unused,nposition,0)
        assert value>=0, 'branch number less than 0'
        assert value!=1, 'branch number is 1'
        if value == 2:
            endlist += [nposition]
        elif value == 3:
            edgelist += [nposition]
        elif value >= 4:
            branchlist += [nposition]
    return edgelist,branchlist,endlist



def extract_graph_from_skeleton(sk):
    """ generate graph from skeleton 
        
    Args:
        sk (skeletion matric )
    Returns:
        root (Branch object)  
    """ 
    #used/unsused
    sk_used = np.zeros_like(sk)
    sk_unused = np.copy(sk)
    #root node
    root_position = findroot(sk)
    print('root_position',root_position)
    root = Branch(pixels=[root_position],name='root')
    setvalue(sk_used,root_position,1)
    setvalue(sk_unused,root_position,0)
    #extract rood edge
    edgelist,branchlist,endlist = next_pixels(root_position,sk_used,sk_unused)
    #assert len(edgelist)==1,'root has more than 1 branchedge'################!!!!!!!!
    rootedge = BranchEdge(edgelist[:1])
    while True:
        edgelist,branchlist,endlist = next_pixels(edgelist[0],sk_used,sk_unused)
        if edgelist:
            rootedge.add_pixels(edgelist)
        else:
            break
    assert len(branchlist)>=1,'root has no children'
    #first node(perhaps split LM and RM)
    branch1 = Branch(pixels=branchlist)
    root.add_child(branch1,rootedge)
    branch_startpoint_list = [branch1]##BFS
    edge_startpoint_list = []
    while branch_startpoint_list:
        branch1 = branch_startpoint_list.pop(0)
        edgelist,branchlist,endlist = next_pixels(branch1.pixels[0],sk_used,sk_unused)
        edge_startpoint_list = edgelist
        branch_cumulate_list = branchlist
        while branch_cumulate_list:#cumulate all the branch pixels(>3)
            bposition = branch_cumulate_list.pop(0)
            branch1.add_pixel(bposition)
            edgelist,branchlist,endlist = next_pixels(bposition,sk_used,sk_unused)
            edge_startpoint_list += edgelist
            branch_cumulate_list += branchlist
        #for each connected edge start,trace until next node
        for edge in edge_startpoint_list:
            branchedge1 = BranchEdge([edge])
            edgelist,branchlist,endlist = next_pixels(edge,sk_used,sk_unused)
            while edgelist:#trace until next node
                #print('edgelist',edgelist)
                branchedge1.add_pixels(edgelist)
                edgelist,branchlist,endlist = next_pixels(edgelist[0],sk_used,sk_unused)
            if branchlist:#next branch
                branch2 = Branch(pixels=branchlist)
                ##if branchedge too short, do nothing
                branch1.add_child(branch2,branchedge1)
                branch_startpoint_list.append(branch2)
            elif endlist:#end node
                branch2 = Branch(pixels=endlist)
                ##if branchedge too short, threshold based on rank(todo)
                branch1.add_child(branch2,branchedge1)
            else:#end without endlist (pixel value=3)
                branch2 = Branch(pixels=branchedge1.pixels[-1:])
                ##if branchedge too short, threshold based on rank(todo)
                branch1.add_child(branch2,branchedge1)
        #if this branch has only one edge, merge(may throw assert error)
        if len(branch1.edges) == 1:
            branch1.edges[0].endbracnch.rank-=1
            branch1.parent_edge.endbracnch = branch1.edges[0].endbracnch
            branch1.parent_edge.add_pixels_nocontinious(branch1.pixels)
            branch1.parent_edge.add_pixels(branch1.edges[0].pixels)
            branch1.edges[0].endbracnch.parent_edge = branch1.parent_edge
    return root







def draw_graph_tomatrix(root,shape,showlabel=True):
    """ generate the 3d data matrix based on the graph.

        DFS, set the edge value in matrix based on Branch's label if showlabel=True
        otherwise set as 1; set hte branch value as 2        
    Args:
        root (Branch object )
        shape (shape of the data matrix )
        showlabel (if to set the pixel value based on the label  )
    Returns:
        sk_new (skeleton matrix)  
    """ 
    sk_new = np.zeros(shape) 
    nodestack = [root]
    indexstack = [0]
    while nodestack:
        node = nodestack[-1]
        index = indexstack[-1]
        if index==0:
            setvalues(sk_new,node.pixels,2)#new
            for edge in node.edges:
                if showlabel and edge.endbracnch.label!=0:
                    setvalues(sk_new,edge.pixels,edge.endbracnch.label)#new
                elif showlabel and node.label!=0:
                    setvalues(sk_new,edge.pixels,node.label)#new
                else:
                    setvalues(sk_new,edge.pixels,1)#new
        if index < len(node.edges):
            nodestack += [node.edges[index].endbracnch]
            indexstack[-1] += 1
            indexstack += [0]
        else:
            nodestack.pop()
            indexstack.pop()
    return sk_new





def draw_graph_tomatrix_clean(root,shape,showlabel=True):
    """ generate the 3d data matrix based on the graph
        
        DFS, set the value in matrix based on Branch's label if showlabel=True
        otherwise set as 1. Do not set different value for branch node.
    Args:
        root (Branch object )
        shape (shape of the data matrix )
        showlabel (if to set the pixel value based on the label  )
    Returns:
        sk_new (skeleton matrix)  
    """ 
    sk_new = np.zeros(shape) 
    nodestack = [root]
    indexstack = [0]
    while nodestack:
        node = nodestack[-1]
        index = indexstack[-1]
        if index==0:
            if node.edges:
                if showlabel:
                    if node.label!=0:
                        setvalues(sk_new,node.pixels,node.label)#new
                    else:
                        setvalues(sk_new,node.pixels,1)#new
                else:
                    setvalues(sk_new,node.pixels,1)#new
            for edge in node.edges:
                if showlabel and edge.endbracnch.label!=0:
                    setvalues(sk_new,edge.pixels,edge.endbracnch.label)#new
                elif showlabel and node.label!=0:
                    setvalues(sk_new,edge.pixels,node.label)#new
                else:
                    setvalues(sk_new,edge.pixels,1)#new
        if index < len(node.edges):
            nodestack += [node.edges[index].endbracnch]
            indexstack[-1] += 1
            indexstack += [0]
        else:
            nodestack.pop()
            indexstack.pop()
    return sk_new


def etename(node):
    """ naming function for ete3
        
    Args:
        node (Branch object )
    Returns:
        name (string)  
    """ 
    return node.name+str(node.rank)+'*'+str(node.bottomuprank)

def etename1(node):
    """ naming function for ete3
        
    Args:
        node (Branch object )
    Returns:
        name (string)  
    """ 
    d = {0:node.name, RIGHTUP:'RIGHTUP', RIGHTMID:'RIGHTMID', RIGHTDOWN:'RIGHTDOWN', LEFTUP:'LEFTUP', LEFTDOWM:'LEFTDOWM'}
    return d[node.label]
        
def graph2ete3(root,etename=etename):
    """ generate ete3 type string from the graph DFS
        
    Args:
        root (Branch object )
        etename (naming function [etename, etename1] )
    Returns:
        t (ete3 string)  
    """ 
    t=''
    nodestack = [root]
    indexstack = [0]
    while nodestack:
        node = nodestack[-1]
        index = indexstack[-1]
        if index==0:
            if len(node.edges)>0:
                if (t) and t[-1]!='(':
                    t+=','
                t+='('
        if index < len(node.edges):
            nodestack += [node.edges[index].endbracnch]
            indexstack[-1] += 1
            indexstack += [0]
        else:
            if len(node.edges)>0:
                t+=')'
            else:
                if (t) and t[-1]!='(':
                    t+=','
            t+=etename(node)
            nodestack.pop()
            indexstack.pop()
    return t+';'

def remove_noise1(root):#DFS
    """ remove_noise DFS
        
    Args:
        root (Branch object )
    Returns:
        flag (Bool if some noise removed)  
    """ 
    root.update_bottonuprank()
    nodestack = [root]
    indexstack = [0]
    flag = False
    while nodestack:
            node = nodestack[-1]
            index = indexstack[-1]
            if index==0:
                #2 criteria for noise branch
                if node.rank+node.bottomuprank ==4:
                    print('tl',node.get_treelength(),node.position)
                    print('detect',len(node.edges),node.rank,node.bottomuprank,'_')
                    print('_',len(node.parent_edge.startbracnch.edges),node.parent_edge.startbracnch.rank,node.parent_edge.startbracnch.bottomuprank)

                if node.rank+node.bottomuprank <=3 or (node.rank+node.bottomuprank ==4 and node.get_treelength()<=2):
                    print('delete',len(node.edges),node.rank,node.bottomuprank,'_')
                    print('_',len(node.parent_edge.startbracnch.edges),node.parent_edge.startbracnch.rank,node.parent_edge.startbracnch.bottomuprank)
                    nodestack[-2].delete_child(node.parent_edge)
                    nodestack.pop()
                    indexstack.pop()
                    print('_',len(node.parent_edge.startbracnch.edges),node.parent_edge.startbracnch.rank,node.parent_edge.startbracnch.bottomuprank)
                    
                    flag=True
                    continue
            if index < len(node.edges):
                nodestack += [node.edges[index].endbracnch]
                indexstack[-1] += 1
                indexstack += [0]
            else:
                nodestack.pop()
                indexstack.pop()
    print('flag',flag)
    return flag 

def combine_singledge1(root):#DFS
    """combine single edge
        
    Args:
        root (Branch object )
    """ 
    nodestack = [root]
    indexstack = [0]
    while nodestack:
            node = nodestack[-1]
            index = indexstack[-1]
            #print('nodestack',len(node.edges),node.rank,node.bottomuprank)
            if index==0:
                if len(node.edges) == 1 and node.parent_edge:
                    print('combine',len(node.edges),node.rank,node.bottomuprank,'_',len(node.parent_edge.startbracnch.edges),node.parent_edge.startbracnch.rank,node.parent_edge.startbracnch.bottomuprank)
                    #update rank later
                    #branch1.edges[0].endbracnch.rank-=1
                    node.parent_edge.endbracnch = node.edges[0].endbracnch
                    #node.parent_edge.add_pixels(node.pixels)
                    node.parent_edge.add_pixels_nocontinious(node.pixels)
                    node.parent_edge.add_pixels(node.edges[0].pixels)
                    nodestack[-1] = node.edges[0].endbracnch
                    node.edges[0].endbracnch.parent_edge = node.parent_edge
                    print('combine2', len(node.parent_edge.endbracnch.edges),node.parent_edge.endbracnch.rank,node.parent_edge.endbracnch.bottomuprank ,'_',len(node.parent_edge.startbracnch.edges),node.parent_edge.startbracnch.rank,node.parent_edge.startbracnch.bottomuprank)
                    continue
            if index < len(node.edges):
                nodestack += [node.edges[index].endbracnch]
                indexstack[-1] += 1
                indexstack += [0]
            else:
                nodestack.pop()
                indexstack.pop()
    root.update_rank()



def combine_shortedge(root):#DFS
    """combine shortedges
        
    Args:
        root (Branch object )
    """ 
    nodestack = [root]
    indexstack = [0]
    while nodestack:
            node = nodestack[-1]
            index = indexstack[-1]
            #print('nodestack',len(node.edges),node.rank,node.bottomuprank)
            if index==0:
                i = 0
                while i < len(node.edges):
                    edge = node.edges[i]
                    if edge.n_node<=3 and edge.endbracnch.rank<6 and edge.endbracnch.bottomuprank>1:
                        node.delete_child(edge)
                        node.add_pixels(edge.pixels)
                        node.add_pixels(edge.endbracnch.pixels)
                        for grandedges in edge.endbracnch.edges:
                            node.add_edge(grandedges)
                        continue
                    i += 1
            if index < len(node.edges):
                nodestack += [node.edges[index].endbracnch]
                indexstack[-1] += 1
                indexstack += [0]
            else:
                nodestack.pop()
                indexstack.pop()
    root.update_rank()
    root.update_bottonuprank()




def render_graph(root):
    """update nodes'label based on their ancestors
        
    Args:
        root (Branch object )
    """ 
    root.update_bottonuprank()
    nodestack = [root]
    indexstack = [0]
    acesor_label = [root.label]
    while nodestack:
            node = nodestack[-1]
            index = indexstack[-1]
            if index==0:
                if root.name=='temp':
                    print('aaaa',[[n.label,n.name] for n in nodestack])
                if len(nodestack)>1 and nodestack[-2].name=='temp':
                    print(nodestack[-2].label,len(nodestack[-2].edges))
                if node.label == 0 and len(nodestack)>1:
                    node.label = nodestack[-2].label
                if node.label in acesor_label[:-1] and len(nodestack)>1:
                    node.label = nodestack[-2].label
                if  len(nodestack)>1 and node.label < nodestack[-2].label:
                    node.label = nodestack[-2].label
                if root.name=='temp':
                    print('aaaa',[[n.label,n.name,n.position] for n in nodestack])
            if index < len(node.edges):
                nodestack += [node.edges[index].endbracnch]
                indexstack[-1] += 1
                indexstack += [0]
                acesor_label += [node.edges[index].endbracnch.label]
            else:
                nodestack.pop()
                indexstack.pop()
                acesor_label.pop()

def get_leafnodes(root):
    """get leaf nodes
        
    Args:
        root (Branch object )
    Returns:
        leafnodes_list (list of leaf nodes)  
    """ 
    leafnodes_list = []
    nodestack = [root]
    indexstack = [0]
    while nodestack:
            node = nodestack[-1]
            index = indexstack[-1]
            if index==0:
                if len(node.edges) == 0:
                    leafnodes_list += [node]
            if index < len(node.edges):
                nodestack += [node.edges[index].endbracnch]
                indexstack[-1] += 1
                indexstack += [0]
            else:
                nodestack.pop()
                indexstack.pop()  
    return leafnodes_list

def tree_direction(root):
    """get tree direction
        
    Args:
        root (Branch object )
    Returns:
        tree_direction
    """
    leafnodes_list = get_leafnodes(root)
    leafnodesposition_list = [node.position for node in leafnodes_list]
    return np.mean(leafnodesposition_list,axis=0)-root.position
EDGE=1
BRANCH=2
RIGHTUP=3
RIGHTMID=6
RIGHTDOWN=7
LEFTUP=8
LEFTUPUP=18
LEFTUPDOWN=28
LEFTDOWM=9
def label_graph(root,oritationy = 1):
    """label airway from 5 lobes
        
    Args:
        root (Branch object )
        oritationy (1 or -1 )
    """
    ###
    #divide left or right lung
    ####
    # node list afer root
    rl_lung_branch = root.get_children()[0].get_children()
    assert len(rl_lung_branch)==2,'r, l two lungs'
    ## 1 layer of nodes
    rl_lung_branch.sort(key=lambda x: x.position[0])#right left in order
    ###
    #right lung
    ####
    ###
    #RIGHTUP
    ####
    right_branch = rl_lung_branch[0]#get the right branch
    ## 2 layer of nodes
    branchlist1 = right_branch.get_children()#get children
    branchlist1.sort(key=lambda x: tree_direction(x)[2])#sort the children by z axis (3rd dimention)    \
                                                          #z increases when goes up\
                                                          #main plus right up(s)\/
                                                           # pre-defined only the first branch goes to right MID and DOWN
    assert len(branchlist1)>=2,'right up has to have at least two branches'
    for branch in branchlist1[1:]:#set [1:] RIGHTUP
        branch.label = RIGHTUP
    ## 3 layer of right nodes
    branchlist2 = branchlist1[0].get_children()#get children for  right MID and DOWN
    #assert len(branchlist2)>=2,'right middle has to have at least two branches'
    branchlist2.sort(key=lambda x: tree_direction(x)[2])#main plus right middles (right bottoms)
    branchlist2 = branchlist2[1:]## pre-defined only the first branch goes to right DOWN
    #for b in branchlist2:
    #    print(b.position ,'b', branchlist1[0].position)
    assert oritationy in [-1,1],'oritationy wrong'
    ###
    #RIGHTMID
    ####
    print([b.position for b in branchlist2])
    if oritationy==-1:#make sure the right MID is forward
        branchlist222 = [b for b in branchlist2 if b.position[1] >= branchlist1[0].position[1]]#compare y between layer 2 and 3, biger y is foward
    elif oritationy==1:
        branchlist222 = [b for b in branchlist2 if b.position[1] < branchlist1[0].position[1]]
    backuplist = branchlist2
    if not branchlist222:# when right DOWN appear first
        for branch in branchlist2:
            branch.label = RIGHTDOWN
        #find the next branch
        branchlist1=branchlist1[0].get_children()
        branchlist1.sort(key=lambda x: tree_direction(x)[2])#sort by z. layer2 -> layer 3
        branchlist2 = branchlist1[0].get_children() # layer 4
        branchlist2.sort(key=lambda x: tree_direction(x)[2])#main plus right middles (right bottoms)
        branchlist2 = branchlist2[1:]#-1*min(2,len(branchlist2)-1)
        print('branchlist2',[b.position for b in branchlist2])
        print('branchlist1',[b.position for b in branchlist1])
        if oritationy==-1:#make sure the right MID is forward
            branchlist222 = [b for b in branchlist2 if b.position[1] >= branchlist1[0].position[1]]#compare y between layer 3 and 4, biger y is foward
        elif oritationy==1:
            branchlist222 = [b for b in branchlist2 if b.position[1] < branchlist1[0].position[1]]
            
    #assert branchlist222,'branchlist2 empty oritationy:'+str(oritationy)#raise error when patient has disease that distort the rightMID
    #[TODO if the airway is distort that we can not find right MID, raise warning.]
    if not branchlist222:
        branchlist2 = backuplist
        for branch in backuplist:
            branch.label = RIGHTMID
    else:
        for branch in branchlist222:
            branch.label = RIGHTMID
    ###
    #RIGHTDOWN
    ####
    ## 3 layer of right nodes
    branchlist3 = branchlist1[0].get_children()
    branchlist3 = [b for b in branchlist3 if b not in branchlist2]
    assert branchlist3,'branchlist3 empty'
    for branch in branchlist3:
        branch.label = RIGHTDOWN 
    ###
    #left lung
    ####
    ###
    #LEFTUP
    ####
    left_branch = rl_lung_branch[1]
    ## 2 layer of nodes
    branchlist1 = left_branch.get_children()
    assert len(branchlist1)>=2,'left up has to have two branches'
    branchlist1.sort(key=lambda x: tree_direction(x)[2])#main plus right up(s)
    ## 3 layer of nodes
    branchlist2 = branchlist1[1:]## pre-defined only the first branch goes to left DOWN
    for branch in branchlist2:
        branch.label = LEFTUP 
    #branchlist3 = [b for b in branchlist1 if b.position[2]<=left_branch.position[2]]
    ###
    #LEFTDOWM
    ####
    ## 3 layer of nodes
    branchlist3 = [branchlist1[0]]
    for branch in branchlist3:
        branch.label = LEFTDOWM 

    render_graph(root)
    return 1


def get_lungnode(root,label,ranklim=5):
    """get the list of nodes that their label==label. DFS
        
    Args:
        root (Branch object )
        label (Rarget label )
        ranklim (only look at node after this rank )
    Returns:
        nodes_list (list of Branch object)
    """
    nodes_list = []
    nodestack = [root]
    indexstack = [0]
    while nodestack:
            node = nodestack[-1]
            index = indexstack[-1]
            if index==0:
                if node.label == label:
                    nodes_list += [node]
                    nodestack.pop()
                    indexstack.pop() 
                    continue
            if index < len(node.edges):
                indexstack[-1] += 1
                if node.edges[index].endbracnch.rank<=ranklim:
                    nodestack += [node.edges[index].endbracnch]
                    indexstack += [0]
            else:
                nodestack.pop()
                indexstack.pop()  
    return nodes_list    


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def tree_parent_direction(root):
    """get the tree direction from the parent node
        
    Args:
        root (Branch object )
    Returns:
        tree direction (tree direction)
    """
    leafnodes_list = get_leafnodes(root)
    leafnodesposition_list = [node.position for node in leafnodes_list]
    print('len(leafnodes_list) ',len(leafnodes_list),[ll.position for ll in leafnodes_list],root.position )
    if len(leafnodes_list) == 1 and root in leafnodes_list:
        return np.mean(leafnodesposition_list,axis=0)-root.parent_edge.startbracnch.position
    if root.parent_edge:
        return np.mean(leafnodesposition_list,axis=0)-root.parent_edge.startbracnch.position
    return np.mean(leafnodesposition_list,axis=0)-root.position


def labelback(sub_segments,yoritation,lobelabel,idstart=1,uplevel=10):
    """label branches based on y deriction. Used in bronchus classfication functions below
        
    Args:
        sub_segments (list of Branch object )
        yoritation ( 1 or -1) )
        lobelabel (which lobe to label )
        idstart (label fron this number) 
        uplevel ( 10 or 100) )
    """
    auglesback = [angle_between(tree_parent_direction(ss)[:2],[0,yoritation]) for ss in sub_segments]
    sub_segments = np.asarray(sub_segments)
    sub_segments = sub_segments[np.argsort(auglesback)]
    for si,ss in enumerate(sub_segments):
        ss.label = (idstart+si)*uplevel+lobelabel
        print('labelback',si,ss.label)
        render_graph(ss)

def labelup(sub_segments,lobelabel,zoritation=1,idstart=1,uplevel=10):
    """label branches based on z deriction. Used in bronchus classfication functions below
        
    Args:
        sub_segments (list of Branch object )
        lobelabel (which lobe to label )
        zoritation ( 1 or -1) )
        idstart (label fron this number) 
        uplevel ( 10 or 100) )
    """
    auglesback = [angle_between(tree_parent_direction(ss),[0,0,zoritation]) for ss in sub_segments]
    sub_segments = np.asarray(sub_segments)
    sub_segments = sub_segments[np.argsort(auglesback)]
    for si,ss in enumerate(sub_segments):
        ss.label = (idstart+si)*uplevel+lobelabel
        print('labelup',si,ss.label)
        render_graph(ss)

LABELTRASH=911
n_segnents = {RIGHTUP:3, RIGHTMID:2, RIGHTDOWN:5, LEFTUP:2, LEFTDOWM:4,LEFTUPUP:2,LEFTUPDOWN:2}
def label_LEFTUPDOWN(root,lobelabel=LEFTUPDOWN,ranklim=5,yoritation=1):##############
    """label LEFTUPDOWN lobe.
        
    Args:
        root (Branch object )
        lobelabel (which lobe to label )
        ranklim (only look at node after this rank )
        yoritation ( 1 or -1) )
    """
    #get the mainnode
    mainnode1_list = get_lungnode(root,lobelabel)
    sub_segments = []
    if len(mainnode1_list) > 1:
        mainnode1_list.sort(key=lambda x: x.get_nbranch())
        sub_segments += mainnode1_list[:-1]
        mainnode1 = mainnode1_list[-1]
    else:
        mainnode1 = mainnode1_list[0]
    print(mainnode1.position)
    mainnode = mainnode1
    n_segment = n_segnents[lobelabel]
    #collect sub_segments 
    while len(sub_segments) < n_segment-1:
        children = mainnode.get_children()
        print('len(children)',len(children))
        if not children:
            break
        children.sort(key=lambda x: x.get_nbranch())#nbranch
        assert len(children)>1,'only one child'
        ###############################################
        for c in children[:-1]:
            if not c.edges:
                print('c.parent_edge.n_node',c.parent_edge.n_node,lobelabel)
                if c.parent_edge.n_node<=7:
                    #c.label = LABELTRASH
                    continue
            sub_segments += [c]
        mainnode = children[-1]
    sub_segments+=[mainnode]
    print('LEFTUP sub_segments a',sub_segments,[n.position for n in sub_segments])
    #if collected sub_segments more than need, combine
    while len(sub_segments) > n_segment:
        anglematrix = np.zeros([len(sub_segments),len(sub_segments)])
        for i in range(len(sub_segments)):
            for j in range(len(sub_segments)):
                if j==i:
                    continue
                anglematrix[i,j] = angle_between(tree_parent_direction(sub_segments[i]),tree_parent_direction(sub_segments[j]))
        for i in range(len(sub_segments)):
            anglematrix[i,i] = anglematrix.max()+1
        mergenodesidx = np.unravel_index(anglematrix.argmin(),anglematrix.shape)
        print('mergenodesidx',mergenodesidx,anglematrix)
        node = Branch(name='temp')
        edge1 = BranchEdge()
        node.edges.append(edge1)
        edge1.startbracnch = node
        edge1.endbracnch = sub_segments[mergenodesidx[0]]
        edge2 = BranchEdge()
        node.edges.append(edge2)
        edge2.startbracnch = node
        edge2.endbracnch = sub_segments[mergenodesidx[1]]
        #if those two branch are all leaf node, next iteration will raize error when compute the angle. Refer to label_LEFTUPUP to fix this
        node.position = np.mean([sub_segments[mergenodesidx[0]].position,sub_segments[mergenodesidx[1]].position],axis=0)
        temp0=sub_segments[mergenodesidx[0]]
        temp1=sub_segments[mergenodesidx[1]]
        sub_segments.remove(temp0)
        sub_segments.remove(temp1)
        sub_segments+=[node] 
        print('mergenodesidx',node.edges[0].endbracnch.position,node.edges[1].endbracnch.position,node.position)
    #print('LEFTUP sub_segments a',sub_segments,node)
    labelup(sub_segments,lobelabel=lobelabel,idstart=1,uplevel=100)

def label_LEFTUPUP(root,lobelabel=LEFTUPUP,ranklim=5,yoritation=1):
    """label LEFTUPUP lobe.
        
    Args:
        root (Branch object )
        lobelabel (which lobe to label )
        ranklim (only look at node after this rank 
        yoritation ( 1 or -1) )
    """
    #get the mainnode
    mainnode1_list = get_lungnode(root,lobelabel)
    sub_segments = []
    if len(mainnode1_list) > 1:
        mainnode1_list.sort(key=lambda x: x.get_nbranch())
        sub_segments += mainnode1_list[:-1]
        mainnode1 = mainnode1_list[-1]
    else:
        mainnode1 = mainnode1_list[0]
    print(mainnode1.position)
    mainnode = mainnode1
    n_segment = n_segnents[lobelabel]
    #collect sub_segments 
    while len(sub_segments) < n_segment-1:
        children = mainnode.get_children()
        print('len(children)',len(children))
        if not children:
            break
        children.sort(key=lambda x: x.get_nbranch())#nbranch
        assert len(children)>1,'only one child'
        ###############################################
        for c in children[:-1]:
            if not c.edges:
                print('c.parent_edge.n_node',c.parent_edge.n_node,lobelabel)
                if c.parent_edge.n_node<=7:
                    #c.label = LABELTRASH
                    continue
            sub_segments += [c]
        mainnode = children[-1]
    sub_segments+=[mainnode]
    print('LEFTUP sub_segments a',sub_segments,[n.position for n in sub_segments])

    #if collected sub_segments more than need, combine
    while len(sub_segments) > n_segment:
        anglematrix = np.zeros([len(sub_segments),len(sub_segments)])
        for i in range(len(sub_segments)):
            for j in range(len(sub_segments)):
                if j==i:
                    continue
                anglematrix[i,j] = angle_between(tree_parent_direction(sub_segments[i]),tree_parent_direction(sub_segments[j]))
                print('tree(',i,',',j,')',tree_parent_direction(sub_segments[i]),tree_parent_direction(sub_segments[j]))
        for i in range(len(sub_segments)):
            anglematrix[i,i] = anglematrix.max()+1
        mergenodesidx = np.unravel_index(anglematrix.argmin(),anglematrix.shape)
        print('mergenodesidx',mergenodesidx,anglematrix)
        node = Branch(name='temp')
        edge1 = BranchEdge()
        node.edges.append(edge1)
        edge1.startbracnch = node
        edge1.endbracnch = sub_segments[mergenodesidx[0]]
        edge2 = BranchEdge()
        node.edges.append(edge2)
        edge2.startbracnch = node
        edge2.endbracnch = sub_segments[mergenodesidx[1]]
        if sub_segments[mergenodesidx[0]].parent_edge:
            p1 = sub_segments[mergenodesidx[0]].parent_edge.startbracnch.position
        else:
            p1 = sub_segments[mergenodesidx[0]].position
        if sub_segments[mergenodesidx[1]].parent_edge:
            p2 = sub_segments[mergenodesidx[1]].parent_edge.startbracnch.position
        else:
            p2 = sub_segments[mergenodesidx[1]].position
        node.position = np.mean([p1,p2],axis=0)
        temp0=sub_segments[mergenodesidx[0]]
        temp1=sub_segments[mergenodesidx[1]]
        sub_segments.remove(temp0)
        sub_segments.remove(temp1)
        sub_segments+=[node] 
        print('mergenodesidx',node.edges[0].endbracnch.position,node.edges[1].endbracnch.position,node.position)
    #print('LEFTUP sub_segments a',sub_segments,node)
    labelup(sub_segments,lobelabel=lobelabel,idstart=1,uplevel=100)
  

def label_LEFTDOWM(root,lobelabel=LEFTDOWM,ranklim=5,yoritation=1):
    """label LEFTDOWM lobe.
        
    Args:
        root (Branch object )
        lobelabel (which lobe to label )
        ranklim (only look at node after this rank 
        yoritation ( 1 or -1) )
    """
    #get the mainnode
    mainnode1_list = get_lungnode(root,lobelabel)
    sub_segments = []
    if len(mainnode1_list) > 1:
        mainnode1_list.sort(key=lambda x: x.get_nbranch())
        sub_segments += mainnode1_list[:-1]
        mainnode1 = mainnode1_list[-1]
    else:
        mainnode1 = mainnode1_list[0]
    print(mainnode1.position)
    mainnode = mainnode1
    n_segment = n_segnents[lobelabel]
    ###
    #finding segments
    ####
    while len(sub_segments) < n_segment:#One more segmnet for DOWN
        children = mainnode.get_children()
        if not children:
            break
        children.sort(key=lambda x: x.get_nbranch())#nbranch
        assert len(children)>1,'only one child'
        ###############################################
        for c in children[:-1]:
            if not c.edges:
                print('c.parent_edge.n_node',c.parent_edge.n_node,lobelabel)
                if c.parent_edge.n_node<=7:
                    #c.label = LABELTRASH
                    continue
            sub_segments += [c]
        mainnode = children[-1]
    sub_segments+=[mainnode]
    for ss in sub_segments:
        if angle_between(tree_parent_direction(ss)[:2],[0,yoritation])<1.5:
            break
    ss.label = 10+lobelabel
    sub_segments.remove(ss)
    #if collected sub_segments more than need, combine
    while len(sub_segments) > n_segment-1:
        anglematrix = np.zeros([len(sub_segments),len(sub_segments)])
        for i in range(len(sub_segments)):
            for j in range(len(sub_segments)):
                anglematrix[i,j] = angle_between(tree_parent_direction(sub_segments[i]),tree_parent_direction(sub_segments[j]))
            anglematrix[i,i] = anglematrix.max()
        mergenodesidx = np.unravel_index(anglematrix.argmin(),anglematrix.shape)
        node = Branch(name='temp')
        edge1 = BranchEdge()
        node.edges.append(edge1)
        edge1.startbracnch = node
        edge1.endbracnch = sub_segments[mergenodesidx[0]]
        edge2 = BranchEdge()
        node.edges.append(edge2)
        edge2.startbracnch = node
        edge2.endbracnch = sub_segments[mergenodesidx[1]]
        #if those two branch are all leaf node, next iteration will raize error when compute the angle. Refer to label_LEFTUPUP to fix this
        node.position = np.mean([sub_segments[mergenodesidx[0]].position,sub_segments[mergenodesidx[1]].position],axis=0)
        temp0=sub_segments[mergenodesidx[0]]
        temp1=sub_segments[mergenodesidx[1]]
        sub_segments.remove(temp0)
        sub_segments.remove(temp1)
        sub_segments+=[node] 
    
    labelback(sub_segments,yoritation,lobelabel=lobelabel,idstart=2)



def label_LEFTUP(root,lobelabel=LEFTUP,ranklim=5,yoritation=1):
    """label LEFTUP lobe.
        
    Args:
        root (Branch object )
        lobelabel (which lobe to label )
        ranklim (only look at node after this rank 
        yoritation ( 1 or -1) )
    """
    #get the mainnode
    mainnode1_list = get_lungnode(root,lobelabel)
    sub_segments = []
    if len(mainnode1_list) > 1:
        mainnode1_list.sort(key=lambda x: x.get_nbranch())
        sub_segments += mainnode1_list[:-1]
        mainnode1 = mainnode1_list[-1]
    else:
        mainnode1 = mainnode1_list[0]
    print(mainnode1.position)
    mainnode = mainnode1
    n_segment = n_segnents[lobelabel]
    ###
    #finding segments
    ####
    while len(sub_segments) < n_segment-1:
        children = mainnode.get_children()
        print('len(children)',len(children))
        if not children:
            break
        children.sort(key=lambda x: x.get_nbranch())#nbranch
        assert len(children)>1,'only one child'
        ###############################################
        for c in children[:-1]:
            if not c.edges:
                print('c.parent_edge.n_node',c.parent_edge.n_node,lobelabel)
                if c.parent_edge.n_node<=7:
                    #c.label = LABELTRASH
                    continue
            sub_segments += [c]
        mainnode = children[-1]
    sub_segments+=[mainnode]
    print('LEFTUP sub_segments a',sub_segments,[n.position for n in sub_segments])
    #if collected sub_segments more than need, combine
    while len(sub_segments) > n_segment:
        anglematrix = np.zeros([len(sub_segments),len(sub_segments)])
        for i in range(len(sub_segments)):
            for j in range(len(sub_segments)):
                if j==i:
                    continue
                anglematrix[i,j] = angle_between(tree_parent_direction(sub_segments[i]),tree_parent_direction(sub_segments[j]))#[TODO] angle_between(v1,v2) |v1| or |v2| should > 0 
        for i in range(len(sub_segments)):
            anglematrix[i,i] = anglematrix.max()+1
        mergenodesidx = np.unravel_index(anglematrix.argmin(),anglematrix.shape)
        print('mergenodesidx',mergenodesidx,anglematrix)
        node = Branch(name='temp')
        edge1 = BranchEdge()
        node.edges.append(edge1)
        edge1.startbracnch = node
        edge1.endbracnch = sub_segments[mergenodesidx[0]]
        edge2 = BranchEdge()
        node.edges.append(edge2)
        edge2.startbracnch = node
        edge2.endbracnch = sub_segments[mergenodesidx[1]]
        #if those two branch are all leaf node, next iteration will raize error when compute the angle. Refer to label_LEFTUPUP to fix this
        node.position = np.mean([sub_segments[mergenodesidx[0]].position,sub_segments[mergenodesidx[1]].position],axis=0)
        temp0=sub_segments[mergenodesidx[0]]
        temp1=sub_segments[mergenodesidx[1]]
        sub_segments.remove(temp0)
        sub_segments.remove(temp1)
        sub_segments+=[node] 
        print('mergenodesidx',node.edges[0].endbracnch.position,node.edges[1].endbracnch.position,node.position)
    #print('LEFTUP sub_segments a',sub_segments,node)
    labelup(sub_segments,lobelabel=lobelabel,idstart=1)
  


def label_RIGHTDOWN(root,lobelabel=RIGHTDOWN,ranklim=5,yoritation=1):
    """label RIGHTDOWN lobe.
        
    Args:
        root (Branch object )
        lobelabel (which lobe to label )
        ranklim (only look at node after this rank 
        yoritation ( 1 or -1) )
    """
    #get the mainnode
    mainnode1_list = get_lungnode(root,lobelabel)
    sub_segments = []
    if len(mainnode1_list) > 1:
        mainnode1_list.sort(key=lambda x: x.get_nbranch())
        sub_segments += mainnode1_list[:-1]
        mainnode1 = mainnode1_list[-1]
    else:
        mainnode1 = mainnode1_list[0]
    print(mainnode1.position)
    mainnode = mainnode1
    n_segment = n_segnents[lobelabel]
    ###
    #finding segments
    ####
    while len(sub_segments) < n_segment:#find one more in DOWM specil
        children = mainnode.get_children()
        if not children:
            break
        children.sort(key=lambda x: x.get_nbranch())#nbranch
        assert len(children)>1,'only one child'
        ###############################################
        for c in children[:-1]:
            if not c.edges:
                print('c.parent_edge.n_node',c.parent_edge.n_node,lobelabel)
                if c.parent_edge.n_node<=7:
                    #c.label = LABELTRASH
                    continue
            sub_segments += [c]
        mainnode = children[-1]
    sub_segments+=[mainnode]
    sub_segments.sort(key=lambda x: x.position[2], reverse=True)
    for ss in sub_segments:
        if angle_between(tree_parent_direction(ss)[:2],[0,yoritation])<1.5:
            break
    #label apical first
    ss.label = 10+lobelabel
    sub_segments.remove(ss)
    #if collected sub_segments more than need, combine
    while len(sub_segments) > n_segment-1:
        anglematrix = np.zeros([len(sub_segments),len(sub_segments)])
        for i in range(len(sub_segments)):
            for j in range(len(sub_segments)):
                anglematrix[i,j] = angle_between(tree_parent_direction(sub_segments[i]),tree_parent_direction(sub_segments[j]))
            anglematrix[i,i] = anglematrix.max()
        mergenodesidx = np.unravel_index(anglematrix.argmin(),anglematrix.shape)
        node = Branch(name='temp')
        edge1 = BranchEdge()
        node.edges.append(edge1)
        edge1.startbracnch = node
        edge1.endbracnch = sub_segments[mergenodesidx[0]]
        edge2 = BranchEdge()
        node.edges.append(edge2)
        edge2.startbracnch = node
        edge2.endbracnch = sub_segments[mergenodesidx[1]]
        #if those two branch are all leaf node, next iteration will raize error when compute the angle. Refer to label_LEFTUPUP to fix this
        node.position = np.mean([sub_segments[mergenodesidx[0]].position,sub_segments[mergenodesidx[1]].position],axis=0)
        temp0=sub_segments[mergenodesidx[0]]
        temp1=sub_segments[mergenodesidx[1]]
        sub_segments.remove(temp0)
        sub_segments.remove(temp1)
        sub_segments+=[node] 
    
    labelback(sub_segments,yoritation,lobelabel=lobelabel,idstart=2)

  

def label_RIGHTMID(root,lobelabel=RIGHTMID,ranklim=5,yoritation=1):
    """label RIGHTMID lobe.
        
    Args:
        root (Branch object )
        lobelabel (which lobe to label )
        ranklim (only look at node after this rank 
        yoritation ( 1 or -1) )
    """
    #get the mainnode
    mainnode1_list = get_lungnode(root,lobelabel)# find the start point of Right MID
    ###
    #finding segments
    ####
    sub_segments = []
    if len(mainnode1_list) > 1:
        mainnode1_list.sort(key=lambda x: x.get_nbranch())# sort by # of branch
        sub_segments += mainnode1_list[:-1]
        mainnode1 = mainnode1_list[-1]# choose the node with maximum branches as main
    else:
        mainnode1 = mainnode1_list[0]
    print(mainnode1.position)
    mainnode = mainnode1
    n_segment = n_segnents[lobelabel]# predefined n_segment is constant 2
    while len(sub_segments) < n_segment-1:  #####collect all possible segs after layer 3
        children = mainnode.get_children()
        print('len(children)',len(children))
        if not children:
            break
        children.sort(key=lambda x: x.get_nbranch())#nbranch
        assert len(children)>1,'only one child'
        ###############################################
        for c in children[:-1]:
            if not c.edges:
                print('c.parent_edge.n_node',c.parent_edge.n_node,lobelabel)
                if c.parent_edge.n_node<=7:
                    #c.label = LABELTRASH
                    continue
            sub_segments += [c]
        mainnode = children[-1]
    sub_segments+=[mainnode]  
    print('RIGHTMID sub_segments a',sub_segments,[n.position for n in sub_segments])
    ###
    #conbine segments if find too many sub_segments, we want to conbine some pairs
    ####
    while len(sub_segments) > n_segment:
        anglematrix = np.zeros([len(sub_segments),len(sub_segments)])
        for i in range(len(sub_segments)):
            for j in range(len(sub_segments)):
                if j==i:
                    continue
                anglematrix[i,j] = angle_between(tree_parent_direction(sub_segments[i]),tree_parent_direction(sub_segments[j]))####calculate the angle between two tree
        for i in range(len(sub_segments)):
            anglematrix[i,i] = anglematrix.max()+1
        mergenodesidx = np.unravel_index(anglematrix.argmin(),anglematrix.shape)###combine the two branches that has the minimum angle, (i,j)
        print('mergenodesidx',mergenodesidx,anglematrix)
        ####combining: create a 'temp' node. Set the two nearest nodes as the temp node's children. Replace the two nodes by the temp node in sub_segments;
        ###repeat until the # of sub_segments equals to n_segment(constant 2)
        node = Branch(name='temp')
        edge1 = BranchEdge()
        node.edges.append(edge1)
        edge1.startbracnch = node
        edge1.endbracnch = sub_segments[mergenodesidx[0]]
        edge2 = BranchEdge()
        node.edges.append(edge2)
        edge2.startbracnch = node
        edge2.endbracnch = sub_segments[mergenodesidx[1]]
        #if those two branch are all leaf node, next iteration will raize error when compute the angle. Refer to label_LEFTUPUP to fix this
        node.position = np.mean([sub_segments[mergenodesidx[0]].position,sub_segments[mergenodesidx[1]].position],axis=0)#[TODO] a better way to define this position
        temp0=sub_segments[mergenodesidx[0]]
        temp1=sub_segments[mergenodesidx[1]]
        print('mergenodesidx',sub_segments[mergenodesidx[0]].position,sub_segments[mergenodesidx[1]].position,node.position)
        sub_segments.remove(temp0)
        sub_segments.remove(temp1)
        sub_segments+=[node] 
        print('mergenodesidx',node.edges[0].endbracnch.position,node.edges[1].endbracnch.position,node.position)
    print('RIGHTMID sub_segments a',sub_segments)
    
    ###
    #labal segments based on direction
    ####
    labelback(sub_segments,yoritation,lobelabel=lobelabel,idstart=1)
    

def label_RIGHTUP(root,lobelabel=RIGHTUP,ranklim=5,yoritation=1):
    """label RIGHTUP lobe.
        
    Args:
        root (Branch object )
        lobelabel (which lobe to label )
        ranklim (only look at node after this rank 
        yoritation ( 1 or -1) )
    """
    #get the mainnode
    mainnode1_list = get_lungnode(root,lobelabel)
    sub_segments = []
    if len(mainnode1_list) > 1:
        mainnode1_list.sort(key=lambda x: x.get_nbranch())
        sub_segments += mainnode1_list[:-1]
        mainnode1 = mainnode1_list[-1]
    else:
        mainnode1 = mainnode1_list[0]
    print(mainnode1.position)
    mainnode = mainnode1
    n_segment = n_segnents[lobelabel]
    ###
    #finding segments
    ####
    while len(sub_segments) < n_segment-1:
        children = mainnode.get_children()
        if not children:
            break
        children.sort(key=lambda x: x.get_nbranch())#nbranch
        assert len(children)>1,'only one child'
        ###############################################
        for c in children[:-1]:
            if not c.edges:
                print('c.parent_edge.n_node',c.parent_edge.n_node,lobelabel)
                if c.parent_edge.n_node<=7:
                    #c.label = LABELTRASH
                    continue
            sub_segments += [c]
        mainnode = children[-1]
    sub_segments+=[mainnode]
    #if collected sub_segments more than need, combine
    while len(sub_segments) > n_segment:
        anglematrix = np.zeros([len(sub_segments),len(sub_segments)])
        for i in range(len(sub_segments)):
            for j in range(len(sub_segments)):
                anglematrix[i,j] = angle_between(tree_parent_direction(sub_segments[i]),tree_parent_direction(sub_segments[j]))
            anglematrix[i,i] = anglematrix.max()
        mergenodesidx = np.unravel_index(anglematrix.argmin(),anglematrix.shape)
        node = Branch(name='temp')
        edge1 = BranchEdge()
        node.edges.append(edge1)
        edge1.startbracnch = node
        edge1.endbracnch = sub_segments[mergenodesidx[0]]
        edge2 = BranchEdge()
        node.edges.append(edge2)
        edge2.startbracnch = node
        edge2.endbracnch = sub_segments[mergenodesidx[1]]
        #if those two branch are all leaf node, next iteration will raize error when compute the angle. Refer to label_LEFTUPUP to fix this
        node.position = np.mean([sub_segments[mergenodesidx[0]].position,sub_segments[mergenodesidx[1]].position],axis=0)
        print('sub_segments',node,sub_segments)
        temp0=sub_segments[mergenodesidx[0]]
        temp1=sub_segments[mergenodesidx[1]]
        sub_segments.remove(temp0)
        sub_segments.remove(temp1)
        sub_segments+=[node] 
        print('sub_segments',node,sub_segments)
    
    print('sub_segments a',sub_segments)
    #label the apical first
    auglesup = [angle_between(tree_parent_direction(ss),[0,0,1]) for ss in sub_segments]
    sub_segments[np.argmin(auglesup)].label = 10+lobelabel
    render_graph(sub_segments[np.argmin(auglesup)])
    #label post/ant
    sub_segments.remove(sub_segments[np.argmin(auglesup)])
    labelback(sub_segments,yoritation,lobelabel=lobelabel,idstart=2)



def label_sublobe(root,lobelabel,ranklim=5):
    """Older version  brochus classification.
        
    Args:
        root (Branch object )
        lobelabel (which lobe to label )
        ranklim (only look at node after this rank  )
    """
    #get the mainnode
    mainnode1_list = get_lungnode(root,lobelabel)
    sub_segments = []
    if len(mainnode1_list) > 1:
        mainnode1_list.sort(key=lambda x: x.get_nbranch())
        sub_segments += mainnode1_list[:-1]
        mainnode1 = mainnode1_list[-1]
    else:
        mainnode1 = mainnode1_list[0]
    print(mainnode1.position)
    mainnode = mainnode1
    n_segment = n_segnents[lobelabel]
    ###
    #finding segments
    ####
    while len(sub_segments) < n_segment-1:
        children = mainnode.get_children()
        if not children:
            print('break')
            break
        children.sort(key=lambda x: x.get_nbranch())#nbranch
        assert len(children)>1,'only one child'
        ###############################################
        for c in children[:-1]:
            if not c.edges:
                print('c.parent_edge.n_node',c.parent_edge.n_node,lobelabel)
                if c.parent_edge.n_node<=7:
                    #c.label = LABELTRASH
                    continue
            sub_segments += [c]
                
        #sub_segments += children[:-1]
        mainnode = children[-1]
    sub_segments+=[mainnode]
    ###
    #labeling segments
    ####
    for si,ss in enumerate(sub_segments):
        ss.label = 10*(1+si)+lobelabel
        print(si,ss.position,n_segment)
    render_graph(mainnode1)
    
def load_labeltree_fromnii(skd):
    """Get graph from skeleton saved in nii.
        
    Args:
        skd (skeleton matrix loaded from nii )
    """
    sk = np.zeros_like(skd)
    sk[skd>0] = 3
    sk[skd==2] = 4
    root_position = findroot(skd)
    setvalue(sk,root_position,2)
    root = extract_graph_from_skeleton(sk) 
    nodestack = [root]
    indexstack = [0]
    while nodestack:
            node = nodestack[-1]
            index = indexstack[-1]
            if index==0 and node.parent_edge:
                pp = np.array(node.parent_edge.pixels).transpose()
                #print('pp',pp.shape)
                label = skd[pp.tolist()]
                #print('label',label.shape)
                label = label.max()
                #print('label',label)
                if label > 1:
                    node.label=label
            if index < len(node.edges):
                indexstack[-1] += 1
                nodestack += [node.edges[index].endbracnch]
                indexstack += [0]
            else:
                nodestack.pop()
                indexstack.pop()  
    return root
       
def skeleton_rander_seg(sk,seg):
    """Older version of lobe_airwayseg.
        
    Args:
        sk (skeletion matrix )
        seg (airway segmentation matrixk =  )
    Returns:
        result (rendered matrix)
    """
    result = np.zeros_like(seg)
    for i in range(10):    
        print('sk',i,sk.sum(),result.max())
        sk = ndimage.maximum_filter(sk, size=3)
        k = sk*seg
        result[result==0] = k[result==0]
    for i in range(2):    
        print('sk',i+10,sk.sum(),result.max())
        sk = ndimage.maximum_filter(sk, size=11)
        k = sk*seg
        result[result==0] = k[result==0]
        
    return result


LABELTRASH=911
def cut_tail(root,labelthresh=14):
    """Cut tail that leak into lung.
        
    Args:
        root (Branch object )
        labelthresh (detect nodes after this rank   )
    """
    nodestack = [root]
    indexstack = [0]
    while nodestack:
            node = nodestack[-1]
            index = indexstack[-1]
            if index==0:
                #3 criteria
                if (node.edges) and node.rank >= labelthresh and node.get_nbranch()> 10 and root.get_treenpixels()/node.get_nbranch()<=3:
                    node.label = LABELTRASH
            if index < len(node.edges):
                indexstack[-1] += 1
                nodestack += [node.edges[index].endbracnch]
                indexstack += [0]
            else:
                nodestack.pop()
                indexstack.pop()  
    render_graph(root) 



def lobe_airwayseg(airwayseg,skeleton):
    """For each pixel inside airway, find the nearest pixel in skeleton.
        
    Args:
        airwayseg (airway segmentation mask  )
        skeleton (lung skeleton mask  )
    Returns:
        airwayseg (rendered airwayseg)
    """
    assert airwayseg.shape == skeleton.shape,'airwayseg and skeleton file has different size'
    assert skeleton.max()>1,'skeleton not labeled'
    airwayposi = np.asarray(np.where(skeleton>0))
    airwayshape = np.asarray(np.where(airwayseg>0)).transpose()
    count = 0
    for [i,j,k] in airwayshape:
        mydis =  np.linalg.norm(airwayposi-[[i],[j],[k]],axis=0)#list of distanse from [i,j,k] to all pixels in skeleton
        nnp = airwayposi[:,np.argmin(mydis)]#nearest pixel in skeleton
        setvalue(airwayseg,[i,j,k],getvalue(skeleton,nnp))
        count += 1
        if count%10000==0:
            print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),i,j,k,airwayseg.shape,nnp,getvalue(skeleton,nnp),airwayseg[i,j,k],skeleton[skeleton>0].shape,airwayseg[airwayseg>0].shape)   
      
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),'lobe_airwayseg finished')
    return airwayseg



labeldic = {13:'RS1',23:'RS2',33:'RS3',16:'RS4',26:'RS5',17:'RS6',27:'RS7',37:'RS10',47:'RS9',57:'RS8',
    18:'LS123',28:'LS45',118:'LS1',218:'LS3',128:'LS4',228:'LS5',19:'LS6',29:'LS10',39:'LS9',49:'LS8'}
def label_nodule(skeleton,position,affine):
    """For the pixel atposition, find the nearest pixel in skeleton.
        
    Args:
        skeleton (lung skeleton mask  )
        position ([x,y,z] )
    Returns:
        label (the value at the nearest pixel in skeleton)
    """
    assert skeleton.max()>1,'skeleton not labeled'
    airwayposi = np.asarray(np.where(np.logical_and(skeleton>10,skeleton>10)))
    airwayposiphisic = np.dot(affine[:3,:3],airwayposi) + affine[:3,3:4]
    [i,j,k] = nib.affines.apply_affine(affine,position)
    mydis =  np.linalg.norm(airwayposiphisic-[[i],[j],[k]],axis=0)#list of distanse from [i,j,k] to all pixels in skeleton
    #print(mydis[np.argsort(mydis)][:30])
    #print(airwayposi[:,np.argsort(mydis)][:,:30])
    nnp = airwayposi[:,np.argmin(mydis)]#nearest pixel in skeleton
    #print('nnp',nnp,)
    label = getvalue(skeleton,nnp)
    print('nnp',nnp,label)
    return labeldic[label]



def lobe_render(lungseg,airway):####the whole lung based on labeled airway 
    """For each pixel inside lung, find the nearest pixel in airway.
        
    Args:
        lungseg (lung segmentation mask  )
        airway (airway segmentation mask with bronchus labeled  )
    Returns:
        lungseg (rendered lungseg)
    """
    assert lungseg.shape == airway.shape,'lung mask and airway file has different size'
    assert airway.max()>1,'airway not labeled'
    airwayposi = np.asarray(np.where(airway>10))
    lungshape = np.asarray(np.where(lungseg>0)).transpose()
    count = 0
    for [i,j,k] in lungshape:
        #if lungseg[i,j,k] == 0:
        #    continue           
        if airway[i,j,k] > 0:
            continue
        mydis =  np.linalg.norm(airwayposi-[[i],[j],[k]],axis=0)
        nnp = airwayposi[:,np.argmin(mydis)]# position of the nearest pixel in airway
        setvalue(lungseg,[i,j,k],getvalue(airway,nnp))
        count += 1
        if count%100000==0:
            print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),i,j,k,lungseg.shape,nnp,getvalue(airway,nnp),lungseg[i,j,k],nnp)
            #break   
    lungseg[airway>0] = airway[airway>0]
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),'lobe_render finished')
      
    return lungseg






