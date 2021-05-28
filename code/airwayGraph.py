import numpy as np
from branch import Branch
from branchEdge import BranchEdge
class AirwayGraph:
    def __init__(self, root=None):
        self.root = root
    def update_rank(self):
        self.root.rank=0
        self.root.update_rank()
    def get_tree_length(self):
        return 0
    def get_max_n_nodes_inapath(self):
        return 0
    def get_max_rank(self):
        return 0
    def get_path_length(self,node1,node2):
        return 0
    def get_path_n_node(self,node1,node2):
        return 0
