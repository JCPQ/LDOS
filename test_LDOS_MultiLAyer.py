"""
20130311 JCP

This script is the main scrip for the calculation of the LDOS in an arbitrary
multilayer. 

For a 3 or 2 layer system consider using simpler, faster scripts this code
is up to 5 times slower then a didcated3 layer code.

"""
# standrad modules
import numpy as np
import matplotlib.pyplot as plt
# importing my own modules
import ScatteringMatrix as SM
import GreenFunctions as GF
#from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})

# wavelength for the calculation in meter
lambda0 = 488e-9
# angular frequency    
w = 2*np.pi*3e8/lambda0

# some dielectrc constants at 488 nm
gold = -2.5460519958986936+4.0716231013182584j
glass = (1.52+0j)**2
air = (1.0+0j)**2
Cr = -3.6565363558577548+17.565887662743119j
water = 1.78#dc.H2O(lambda0*1e9)
Si = 19.069947966758999+0.69100529085872497j#dc.Si(lambda0*1e9)
# Generate a multilayer, use dielectric constants and not refractive indices.
# Layer 0: glass, infinately thick
# Layer 1: Cr, 4 nm
# Layer 2: Au, 100 nm
# Layer 3: water, 300 nm, in this layer the dipole will be put
# Layer 4: water, infinately thick
ML = SM.MultiLayer(np.array([glass,Cr,gold,water,water]),
                    np.array([np.inf,4e-9,100e-9,300e-9,np.inf]))

# The layer number in which the dipole is located. (Note: it starts from 0!)
LayerNumber = 3

# A loop through all the positions in the layer
N = 100                           # number of positions
LDOSper = np.empty(N)             # generate LDOS arrays
LDOSpar = np.empty(N)
d = np.linspace(1e-9,ML.d[LayerNumber]-1e-9,N) # generate z-positions in layer
for n in range(0,len(d)): 
    print(n, ' of ', len(d))
    A,B = GF.LDOS(ML,LayerNumber,d[n],w)
    LDOSper[n] = A
    LDOSpar[n] = B


"""
making a graph for Atto488
"""
lifetime=4.0e-9
QE=0.92 # quantum efficiency
fig,ax=plt.subplots()
plt.plot(d*1e9,1e9*lifetime*(1/(1-QE+LDOSpar*QE)),linewidth=3,color=(79.0/256,129.0/256,189.0/256))
plt.plot(d*1e9,1e9*lifetime*(1/(1-QE+LDOSper*QE)),linewidth=3,color=(200.0/256,10.0/256,0.0/256))
plt.plot(d*1e9,1e9*lifetime*(1/(1-QE+((1/3)*LDOSper+(2/3)*LDOSpar)*QE)),linewidth=3,color=(100.0/256,10.0/256,0.0/256))

plt.xlabel('Distance (nm)',fontsize=20)
plt.xticks(np.array([0,50,100,150,200,250,300]),fontsize=16)
plt.yticks(fontsize=16)
plt.legend(('Parallel','Perpendicular','Isotropic'),loc=4)
plt.ylabel('Lifetime in ns',fontsize=20)
plt.title('Atto 488 lifetime Glass, Cr, Au, water ')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
for line in ax.get_xticklines() + ax.get_yticklines():
    line.set_markeredgewidth(2)
    line.set_markersize(5)