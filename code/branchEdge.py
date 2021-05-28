import numpy as np
class BranchEdge:
    """BranchEdge"""
    def __init__(self, pixels=[]):
        """Initial the edge by the pixels
        Args:
            pixels (list[ numpy vector(3)]): 
        """
        self.pixels = []
        self.n_node =0
        self.length =0
        self.add_pixels(pixels)
        self.startbracnch=None
        self.endbracnch=None
    def update_length(self):
        """Update the length by self.pixels
        """
        if len(self.pixels) <= 0:
            self.length = 0
            return
        self.length = np.linalg.norm(self.pixels[0]-self.pixels[-1])
    def neighbor_check(self,pixel1,pixel2):
        """Check if two pixels are neighbors (always return True because the graph bulding algorithm can not satify this check)
        Args:
            pixel1 (numpy vector(3))
            pixel2 (numpy vector(3))
        Returns:
            bool: The return value. True for are neighbors, False otherwise.
        """
        if np.linalg.norm(pixel1-pixel2)>2:
            return False
        return True
    def add_pixels(self,newpixels):
        """Add a list of pixels
        
        Add a list of pixels and update corresponding attrs(n_node, length) 
        Args:
            newpixels (list[ numpy vector(3)]): 
        """
        if len(newpixels) <= 0:
            return
        assert newpixels[0].size==3,"pixel must contine 3 coordinates(x,y,z)"
        for ni in range(len(newpixels)-1):
            assert newpixels[ni+1].size==3,"pixel must contine 3 coordinates(x,y,z)"
            #assert self.neighbor_check(newpixels[ni],newpixels[ni+1]),"pixels not continuous"
        self.n_node += len(newpixels)
        self.pixels += newpixels
        self.update_length()
    def add_pixel(self,newpixel):
        """Add a pixel

        Add a list of pixel and update corresponding attrs(n_node, length) 
        Args:
            newpixel (numpy vector(3)): 
        """
        assert newpixel.size==3,"pixel must contine 3 coordinates(x,y,z)"
        #assert (not self.pixels) or self.neighbor_check(self.pixels[-1],newpixel),"pixels not continuous"
        self.n_node += 1
        self.pixels += [newpixel]
        self.update_length()

    def add_pixels_nocontinious(self,newpixels):
        """Add a list of pixels(sort the pixels to be continuouis)
        
        Add a list of pixels and update corresponding attrs(n_node, length) 
        Args:
            newpixels (list[ numpy vector(3)]): 
        """
        if len(newpixels) <= 0:
            return
        assert newpixels[0].size==3,"pixel must contine 3 coordinates(x,y,z)"
        for ni in range(len(newpixels)-1):
            assert newpixels[ni+1].size==3,"pixel must contine 3 coordinates(x,y,z)"
        self.n_node += len(newpixels)
        self.pixels += newpixels
        self.update_length()
        '''if len(newpixels) <= 0:
            return
        newpixels_conti = []
        lastpixel = self.pixels[-1]
        while newpixels:
            found_flag = False
            for p in newpixels:
                if self.neighbor_check(lastpixel,p):
                    found_flag = True
                    newpixels_conti.append(p)
                    newpixels.remove(p)
                    lastpixel = p
                    break
            if not found_flag:
                break
        
        assert newpixels_conti[0].size==3,"pixel must contine 3 coordinates(x,y,z)"
        for ni in range(len(newpixels_conti)-1):
            assert newpixels_conti[ni+1].size==3,"pixel must contine 3 coordinates(x,y,z)"
        self.n_node += len(newpixels_conti)
        self.pixels += newpixels_conti
        self.update_length()'''



if __name__ == '__main__':
    a = BranchEdge([np.zeros(3),np.ones(3)])
    print(a,a.n_node,a.length,a.pixels)
    a.add_pixel(np.ones(3)*2)
    print(a,a.n_node,a.length,a.pixels)
    a.add_pixels([np.ones(3)*3,np.ones(3)*4])
    print(a,a.n_node,a.length,a.pixels)
    #a.add_pixel(np.ones(3)*2)
    #print(a,a.n_node,a.length,a.pixels)
    print('type check',type(a),type(a) is BranchEdge)

