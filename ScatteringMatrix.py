import numpy as np
import matplotlib.pyplot as plt
import warnings
"""
This is a module for doing scattering and transefer matrix matrix calculations.
with the main aim of obtaining reflection and transmission coefficients.
It is based on several sources
[1] Principles of Nano Optics by L.Novotny and B.Hecht, 2nd edition
[2] Theory of the radiation of dipoles within a multilayer system, Polerecjy,
    harmle and Maccraith Applied Optics Vol 39 page 3968 (2000)
[3] Formulation and comparison of two recursive matrix algorithms for modeling
    layered diffraction gratings, Lifeng Li, J. opt. Soc. Am A 13 1024 (1996)
Note that [2] and [3] have different definitions for the transfer matrix and 
the order of propagationa and interface transmission!

The implementation of the transfer matrix method is historical, I started with 
it unknowing that it is not a stable algorithm for large parallel k-vectors. 
"""

class MultiLayer:
    """
    V 20130306 JCP
    A class for the properties of multilayers. each layer has 2 properties
    eps - the dielectric constant which should be a complex number
    d - the thickness of the layer which should be real the two outer layers 
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
        epsA=self.eps[0:N+1]
        dA=self.d[0:N+1]
        epsB=self.eps[N:len(self.eps)]
        dB=self.d[N:len(self.eps)]
        MLA=MultiLayer(np.flipud(epsA.copy()),np.flipud(dA.copy()))    
        MLB=MultiLayer(epsB.copy(),dB.copy())    
        return MLA,MLB
    def findLayer(self,distance):
        """
        returns the layer number you are in whene you are at distance from
        the first interface and the distance to the lower (d1) and upper (d2)
        surface
        """
        if distance<=0:
            raise ValueError('The distance value provided is below the layer stack, please use a value > 0')
        D=self.d[0:-1]
        D[0]=0
        d_cum=np.cumsum(D)
        if d_cum[-1]<distance:
            raise ValueError('The distance value provided is above the layer stack, please use a value beteen 0 and '+ str(d_cum[-1]*1e9)+' nm')
        
        layer_nr=np.argwhere((d_cum<distance))[-1]+1
        d2=d_cum[layer_nr]-distance
        d1=D[layer_nr]-d2
        return layer_nr,d1,d2 
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
    

def TransferInterfaceF(k_rho,k0,e1,e2):
    """
    Forward transfer matrices for s and p polarization for one interface, this is 
    the forward matrix T: A(n+1)=T*A(n)
    input:
        k_rho 2D arrays (created by np.meshgrid f.i.)
        k0 the vacuum wave vector
        e1 and e2 the dielectric constants of the first and second medium both are complex
    output:
        Mp and Ms the transfer matrices of the interface for S and P polarisation
        the output has the form of an 4 dimensional array (2D array of 2x2 arrays)
    WARNING; stacking this method is unstable for evanescent waves
    """
    # some input checks (not complete)
    if not (isinstance(e1,complex) or isinstance(e2,complex)):
        raise ValueError('e1 and e2 should be complex numbers')
    elif not isinstance(k_rho,np.ndarray):
        raise ValueError('TransferIndex:k_rho .. is not of type array')
    elif isinstance(k_rho,np.ndarray):
        if k_rho.ndim==1:
            warnings.warn('TransferIndex:k_rho is not 2D, this is not a problem')
            k_rho=k_rho[None,:]#np.array([k_rho])
        elif k_rho.ndim!=2:
            raise ValueError('TransferInterface:k_rho Illegal number of',
                             'dimensions of input argument k_rho')
    
    k1=np.sqrt(e1)*k0
    k2=np.sqrt(e2)*k0 
    kz1=np.sqrt((e1*k0**2-k_rho**2))
    kz2=np.sqrt((e2*k0**2-k_rho**2))
    ts=2*kz1/(kz1+kz2)
    rs=(kz1-kz2)/(kz1+kz2)
    tp=2*k1*k2*kz1/(k2**2*kz1+k1**2*kz2) 
    rp=(k2**2*kz1-k1**2*kz2)/(k2**2*kz1+k1**2*kz2)
    # there is a divide by zero risk here  np.finfo(double).tiny
    # this might be made faster by directly inserting r and t in
    # the matrix M might give different behaviour for kz=0 ???        
    #Ms=np.dstack(((ts/(1-rs**2))*Id,(ts/(1-rs**2))*-rs,(ts/(1-rs**2))*-rs,(ts/(1-rs**2))*Id)).reshape(k_rho.shape +(2,2))
#    Id=np.ones(k_rho.shape)    
#    Ms=np.transpose((ts/(1-rs**2))*np.array([[Id,-rs],[-rs,Id]]),(2,3,0,1))# nice oneliner but array() is very slow!
#    Mp=np.transpose((tp/(1-rp**2))*np.array([[Id,-rp],[-rp,Id]]),(2,3,0,1))
    #print(' Ms ',Ms.shape)
    #print(' c ',c.shape)
    #print(c-Ms)
    Ms=np.empty(k_rho.shape+(2,2),dtype=complex)
    Mp=np.empty(k_rho.shape+(2,2),dtype=complex)
    Ms[:,:,0,0]=(ts/(1-rs**2))
    Ms[:,:,0,1]=(ts/(1-rs**2))*-rs
    Ms[:,:,1,0]=(ts/(1-rs**2))*-rs
    Ms[:,:,1,1]=(ts/(1-rs**2))
    Mp[:,:,0,0]=(tp/(1-rp**2))
    Mp[:,:,0,1]=(tp/(1-rp**2))*-rp
    Mp[:,:,1,0]=(tp/(1-rp**2))*-rp
    Mp[:,:,1,1]=(tp/(1-rp**2))
    
    return Mp,Ms

def SMatrixStack(k_rho,k0,MultiL):
    """
    Iterative procedure for calculating the S-matrix of a multilayer:
    
    S_p, S_s=SMatrixStack(k_rho,k0,MultiL)    
    where
    r12p=Sp[:,:,1,0]
    r21p=Sp[:,:,0,1]
    t12p=Sp[:,:,0,0]
    t21p=Sp[:,:,1,1]

    S_p=np.transpose(np.array([[Tuu_p,Rud_p],[Rdu_p,Tdd_p]]),(2,3,0,1))    
    S_s=np.transpose(np.array([[Tuu_s,Rud_s],[Rdu_s,Tdd_s]]),(2,3,0,1))    
    
    following: 
    JOSA A 13 1024 1996
    
    |u(n+1)| = |Tuu Rud|*| u(0) | 
    | d(0) | = |Rdu Tdd| |d(n=1)|
    
    """
    N=len(MultiL.d)
    # initailizing output S-matrices for s and p polarization
    One=np.ones(k_rho.shape)
    Zero=np.zeros(k_rho.shape)

    Tuu_p=One
    Tdd_p=One
    Rdu_p=Zero
    Rud_p=Zero

    Tuu_s=One
    Tdd_s=One
    Rdu_s=Zero
    Rud_s=Zero
    
    # start the loop over all interfaces and layers
    for n in range(0,N-1):
        e1=MultiL.eps[n]
        e2=MultiL.eps[n+1]
        if n==0:
            d=0
        else:
            d=MultiL.d[n]
        T_p,T_s=TransferInterfaceF(k_rho,k0,e1,e2)
        t11_p=T_p[:,:,0,0]
        t12_p=T_p[:,:,0,1]
        t21_p=T_p[:,:,1,0]
        t22_p=T_p[:,:,1,1]
        t11_s=T_s[:,:,0,0]
        t12_s=T_s[:,:,0,1]
        t21_s=T_s[:,:,1,0]
        t22_s=T_s[:,:,1,1]

        kz=np.sqrt(k0**2*e1-k_rho**2)
        Phi=np.exp(1j*kz*d)
        
        Theta_p=Phi*Rud_p*Phi
        Tdd_p=Tdd_p*Phi/(t22_p+t21_p*Theta_p)
        Rud_p=(t12_p+t11_p*Theta_p)/(t22_p+t21_p*Theta_p)
        Rdu_p=(Rdu_p-Tdd_p*t21_p*Phi*Tuu_p)
        Tuu_p=(t11_p-Rud_p*t21_p)*Phi*Tuu_p
        
        Theta_s=Phi*Rud_s*Phi
        Tdd_s=Tdd_s*Phi/(t22_s+t21_s*Theta_s)
        Rud_s=(t12_s+t11_s*Theta_s)/(t22_s+t21_s*Theta_s)
        Rdu_s=(Rdu_s-Tdd_s*t21_s*Phi*Tuu_s)
        Tuu_s=(t11_s-Rud_s*t21_s)*Phi*Tuu_s

#    the nice oneliners below are very slow due to the array()
#    S_p=np.transpose(np.array([[Tuu_p,Rud_p],[Rdu_p,Tdd_p]]),(2,3,0,1))    
#    S_s=np.transpose(np.array([[Tuu_s,Rud_s],[Rdu_s,Tdd_s]]),(2,3,0,1))
    S_p=np.empty(Tuu_p.shape+(2,2),dtype=complex)
    S_s=np.empty(Tuu_p.shape+(2,2),dtype=complex)
    S_p[:,:,0,0]=Tuu_p
    S_p[:,:,0,1]=Rud_p
    S_p[:,:,1,0]=Rdu_p
    S_p[:,:,1,1]=Tdd_p
    S_s[:,:,0,0]=Tuu_s
    S_s[:,:,0,1]=Rud_s
    S_s[:,:,1,0]=Rdu_s
    S_s[:,:,1,1]=Tdd_s
    return S_p, S_s

def TestFunction():
    """
    test of the formalism verus the fresnel equations:
    The sign definitions are the same for both !!!
    And follow those in Principles of Nano-optics
    """
    
    ## create k-space mesh
    m=1000
    k=2*np.pi/580e-9
    kx=np.linspace(0,20*k,m)
    #
    #K_rho=np.array([kx])
    #
    # create multilayer
    e1=(1.0+0j)**2
    e2=(1.4+0j)**2
    k1=k*np.sqrt(e1)
    k2=k*np.sqrt(e2)
    ML=MultiLayer(np.array([e1,e1,e1,e2,e2]),
                          np.array([np.inf,10e-9,10e-9,10e-9,np.inf]))
    
    ## reflection via new S matrix 
    Sp,Ss=SMatrixStack(kx,k,ML)
    Trans_s=Ss[:,:,0,0]
    Refl_s=Ss[:,:,1,0]
    Trans_p=Sp[:,:,0,0]
    Refl_p=Sp[:,:,1,0]
    
    kz1=np.sqrt(k1**2-kx**2)
    kz2=np.sqrt(k2**2-kx**2)
    tp=(2*e2*kz1/(e2*kz1+e1*kz2))*(np.sqrt(e1/e2))
    ts=2*kz1/(kz1+kz2)
    rp=(e2*kz1-e1*kz2)/(e2*kz1+e1*kz2)
    rs=(kz1-kz2)/(kz1+kz2)
    
#    ts=2*kz1/(kz1+kz2)
#    rs=(kz1-kz2)/(kz1+kz2)
#    tp=2*k1*k2*kz1/(k2**2*kz1+k1**2*kz2) 
#    rp=(k2**2*kz1-k1**2*kz2)/(k2**2*kz1+k1**2*kz2)
    #
    plt.figure()
    plt.title('S-matrix')
    plt.subplot(2,2,1)
    plt.plot(np.real(kx/k),np.squeeze(np.real(Refl_p)),
             np.real(kx/k),np.squeeze(np.real(Refl_s)),'o',
             np.real(kx/k),np.squeeze(np.real(Trans_p)),
             np.real(kx/k),np.squeeze(np.real(Trans_s)),'x')
    plt.subplot(2,2,2)
    plt.plot(np.real(kx/k),np.squeeze(np.real(rp)-np.real(Refl_p)),
             np.real(kx/k),np.squeeze(np.real(rs)-np.real(Refl_s)),'o',
             np.real(kx/k),np.squeeze(np.real(tp)-np.real(Trans_p)),
             np.real(kx/k),np.squeeze(np.real(ts)-np.real(Trans_s)),'x')
    plt.subplot(2,2,3)
    plt.plot(np.real(kx/k),np.squeeze(np.imag(Refl_p)),
             np.real(kx/k),np.squeeze(np.imag(Refl_s)),'o',
             np.real(kx/k),np.squeeze(np.imag(Trans_p)),
             np.real(kx/k),np.squeeze(np.imag(Trans_s)),'x')
    plt.subplot(2,2,4)
    plt.plot(np.real(kx/k),np.squeeze(np.imag(rp)-np.imag(Refl_p)),
             np.real(kx/k),np.squeeze(np.imag(rs)-np.imag(Refl_s)),'o',
             np.real(kx/k),np.squeeze(np.imag(tp)-np.imag(Trans_p)),
             np.real(kx/k),np.squeeze(np.imag(ts)-np.imag(Trans_s)),'x')
    #plt.axis([0,10,-2,4])
    plt.xlabel('Normalised k-vector')
    plt.ylabel('abs(transmission and reflection amplitude)')
    plt.legend(('Rp','Rs','Tp','Ts'))
    plt.show()
    
       
    
    
#TestFunction()   
    
    #print(ML.findLayer(1.2e-7))
    
    #MLA,MLB=ML.splitUD(ML.findLayer(101e-9))
    #print MLA.eps
    #print MLB.eps
    #
    ##import cProfile
    ##cProfile.run('SMatrixStack(K_rho,k,ML)')
    ##cProfile.run('TransferInterfaceF(K_rho,k,1.0+0j,2.2+0j)')
