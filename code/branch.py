import numpy as np
from branchEdge import BranchEdge
class Branch:
    """Branch"""
    #subtreelength = 0
    #n_children = 0
    #n_leafnodes = 0
    def __init__(self, pixels=[],name = 'node'):
        """Initial the edge by the pixels
        Args:
            pixels (list[ numpy vector(3)]): 
        """
        self.pixels = []
        self.position = None
        self.n_node =0
        #self.children = []
        self.edges = []
        self.parent_edge = None
        #self.parent = None
        self.rank = 0
        self.name = name
        self.bottomuprank=0
        self.label=0
        if pixels:
            self.pixels = pixels
            self.position = np.mean(pixels,axis=0)
            self.n_node = len(pixels)
    def add_pixels(self,newpixels):
        """Add a list of pixels
        
        Add a list of pixels and update corresponding attrs(n_node, position) 
        Args:
            newpixels (list[ numpy vector(3)]): 
        """
        if len(newpixels) <= 0:
            return
        for npl in newpixels:
            assert npl.size==3,"pixel must contine 3 coordinates(x,y,z)"
        self.n_node += len(newpixels)
        self.pixels += newpixels
        self.position = np.mean(self.pixels,axis=0)
    def add_pixel(self,newpixel):
        """Add a pixel

        Add a pixel and update corresponding attrs(n_node, position) 
        Args:
            newpixel (numpy vector(3)): 
        """
        assert newpixel.size==3,"pixel must contine 3 coordinates(x,y,z)"
        self.n_node += 1
        self.pixels += [newpixel]
        self.position = np.mean(self.pixels,axis=0)
    def add_edge(self,edge):
        """Add a edge

        Add a edge without visiting the child node 
        Args:
            edge (BranchEdge): 
        """
        assert type(edge) is BranchEdge,'child edge is not BranchEdge type'
        self.edges.append(edge)
        edge.startbracnch = self
    def add_child(self,child,edge):
        """Add a child

        Add a child and update corresponding attrs(rank) 
        Args:
            child (Branch): 
            edge (BranchEdge): 
        """
        assert type(child) is Branch,'child is not Branch type'
        assert type(edge) is BranchEdge,'child edge is not BranchEdge type'
        #self.subtreelength = 0
        #self.n_children = 0
        #self.n_leafnodes = 0
        #self.children.append(child)
        self.edges.append(edge)
        edge.startbracnch = self
        edge.endbracnch = child
        child.parent_edge = edge
        #child.parent = self
        child.rank = self.rank+1
    def delete_child(self,edge):
        """delete a child

        delete a child without updating the bottomuprank
        Args:
            edge (BranchEdge): 
        """
        self.edges.remove(edge)
    def get_children(self):
        """return children
        """
        return [e.endbracnch for e in self.edges]
    def update_rank(self):
        """Update_ranks for the whole subtree whose root is self
        """
        print('update_rank')
        grandchildernqueue = [e.endbracnch for e in self.edges]
        current_rank = self.rank 
        while grandchildernqueue:
            current_rank += 1
            childernqueue = grandchildernqueue
            grandchildernqueue = []
            for child in childernqueue:
                child.rank = current_rank
                grandchildernqueue += [e.endbracnch for e in child.edges]
    def update_bottonuprank(self):
        """Update_bottonuprank for the whole subtree whose root is self
        """
        #print('update_bottonuprank')
        #print('print_subtree')
        nodestack = [self]
        indexstack = [0]
        burankstack = [0]
        while nodestack:
            node = nodestack[-1]
            index = indexstack[-1]
            #if index==0:
            #    print(' '*(node.rank-self.rank)+node.name+str(node.position.astype(int))+'br'+str(len(node.edges))+('l'+str(node.parent_edge.n_node if node.parent_edge else '')))
            if index < len(node.edges):
                nodestack += [node.edges[index].endbracnch]
                indexstack[-1] += 1
                indexstack += [0]
                burankstack += [0]
            else:
                node.bottomuprank = burankstack.pop()
                nodestack.pop()
                indexstack.pop()
                if burankstack:
                    burankstack[-1] = max(burankstack[-1],node.bottomuprank+1 )
                
    def print_subtree(self):
        """print the subtree whose root is self
        """
        #print('print_subtree')
        nodestack = [self]
        indexstack = [0]
        while nodestack:
            node = nodestack[-1]
            index = indexstack[-1]
            if index==0:
                print(' '*(node.rank-self.rank)+node.name)
                #print(' '*(node.rank-self.rank)+node.name+str(node.position.astype(int))+'br'+str(len(node.edges))+('l'+str(node.parent_edge.n_node if node.parent_edge else '')))
            if index < len(node.edges):
                nodestack += [node.edges[index].endbracnch]
                indexstack[-1] += 1
                indexstack += [0]
            else:
                nodestack.pop()
                indexstack.pop()
    def get_nbranch(self):
        """count branck
        """
        nbranch = 0
        nodestack = [self]
        indexstack = [0]
        while nodestack:
            node = nodestack[-1]
            index = indexstack[-1]
            if index==0:
                nbranch += len(node.edges)
            if index < len(node.edges):
                nodestack += [node.edges[index].endbracnch]
                indexstack[-1] += 1
                indexstack += [0]
            else:
                nodestack.pop()
                indexstack.pop()
        return nbranch
    def get_treelength(self):
        """count branck
        """
        nbranch = 0
        nodestack = [self]
        indexstack = [0]
        treelengthstack = [0]
        treelength = 0
        start = 0
        while nodestack:
            node = nodestack[-1]
            index = indexstack[-1]
            if index==0:
                if node.parent_edge and start==0:
                    start=1
                    treelengthstack[-1] += node.parent_edge.n_node
            if index < len(node.edges):
                nodestack += [node.edges[index].endbracnch]
                indexstack[-1] += 1
                indexstack += [0]
                treelengthstack += [treelengthstack[-1]]
            else:
                nodestack.pop()
                indexstack.pop()
                treelength = max(treelength,treelengthstack.pop())
        return treelength
    def get_treenpixels(self):
        """count # pixel in the subtree
        """
        nodestack = [self]
        indexstack = [0]
        treenpixels = 0
        while nodestack:
            node = nodestack[-1]
            index = indexstack[-1]
            if index==0:
                if node.edges:
                    for e in node.edges:
                        treenpixels += e.n_node
            if index < len(node.edges):
                nodestack += [node.edges[index].endbracnch]
                indexstack[-1] += 1
                indexstack += [0]
            else:
                nodestack.pop()
                indexstack.pop()
        return treenpixels
        
'''            
            
import numpy as np
from branchEdge import BranchEdge
from branch import Branch
if __name__ == '__main__':
root = Branch(name='root')
a = Branch(name='a')
b = BranchEdge()
root.add_child(a,b)
root.print_subtree()
a2 = Branch(name='a2')
root.add_child(a2,b)
b2 = Branch(name='b2')
a2.add_child(b2,b)
root.print_subtree()
b1 = Branch(name='b1')
a.add_child(b1,b)
root.print_subtree()'''
