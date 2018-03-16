import numpy as np
import matplotlib.pyplot as plt
import warnings
#import DielectricConstants
"""
V20130306 JCP
This is a module contain the Multilayer class, that defines objects of the type
multilayer and the various thinks youcan dow with or to them
"""

class MultiLayer:
    """
    A class for the properties of multilayers it has 2 properties
    eps - the dielectric constant which should be a complex number
    d - the thickness of the layer whch sould be real the two outer layers 
    be  of thickness np.inf
    
    Example defining a multilayer:
    ML=MultiLayer(np.array([1.51+0j,1.46+0j,1.0+0j,epsg[0],1.51+0j]),
                        np.array([np.inf,1e-7,1e-7,2e-7,np.inf]))

    """
    def __init__(self,DielectricConstants, Thicknesses):
        self.eps = DielectricConstants
        self.d = Thicknesses
    def reverse(self):
        "reverses the array just for testing"
        self.eps = self.eps[::-1]
        self.d = self.d[::-1]
    def split(self,number,depth):
        "Splits layer 'number' in two at a distance 'depth' from the bottom"
        original_thickness=self.d[number]
        if number>len(self.d):
            print('This is not a valid layer number')
        if depth>original_thickness:
            print('This is deeper then the layer is')
            depth=self.d(number)-1e-9
        new_thickness_1=depth
        new_thickness_2=original_thickness-depth
        self.eps = np.insert(self.eps,number,self.eps[number])
        self.d[number]=new_thickness_1
        self.d = np.insert(self.d,number+1,new_thickness_2)
    def subset(self,start,stop):
        "Takes a section out of the multilayer"
        self.eps=self.eps[start:stop+1]
        self.d=self.d[start:stop+1]
    def splitUD(self,N):
        """
        returns two parts of the Multilayer, the upper and the lower,
        the lower is reversed 
        """
        MLA=MultiLayer(np.flipud(self.eps[0:N+1]),np.flipud(self.d[0:N+1]))    
        MLB=MultiLayer(self.eps[N:len(self.eps)],self.d[N:len(self.eps)])    
        return MLA,MLB
    def display(self):
        "Displays the current multilayer"
        zheight=self.d
        zheight[0]=1e-7
        zheight[len(zheight)-1]=1e-7
        zheight=np.insert(zheight,[0],[0])
        zheight=np.cumsum(zheight)*1e6
        dielectric_value=np.transpose(np.tile(np.real(self.eps),(1,1)))
        print('zheigth=',zheight)
        print('dielectric constant',dielectric_value)
        xaxis=np.linspace(0,1,2)

        fig = plt.figure(num=None)
        fig.add_subplot(111, frameon=False, xticks=[], yticks=(zheight))
        plt.pcolormesh(xaxis,zheight,dielectric_value,cmap=plt.cm.autumn)
        plt.suptitle('The multilayer stack')
        plt.ylabel('Multilayer crossection (in um)')
        cb=plt.colorbar()
        cb.set_label('Real part of dielectric constant')
        plt.show()
 
 
def test():
    ML=MultiLayer(np.array([1.51+0j,1.46+0j,1.0+0j,3.0+0j,1.51+0j]),
                            np.array([np.inf,1e-7,1e-7,2e-7,np.inf]))
    
    MultiLayer.display(ML)
        
    ML_up,ML_down=ML.splitUD(2)
    
    MultiLayer.display(ML_up)
    MultiLayer.display(ML_down)
    return
    
test()  
 