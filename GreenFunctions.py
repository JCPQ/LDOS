""" Module containing functions related to Green functions for a multilayer

Version: 20130311 JCP

The aim is to put all complete green functions in here and efficient ways to
calculate them. For the moment it only has:
    G(r,r)xx in k-space (or G(r,r)yy)
    G(r,r)zz in k-space
    LDOS (the integral of the above)

TODO: Speed up the code, either by integrating the functions further
or by using Cython, F2Py etc. Some profiling is also needed
"""


import numpy as np
import ScatteringMatrix as SM
import quadgk as gk
import scipy.constants


def G0_krho_xx(k_rho,k):
    """
    The xx element of the angle integrated green function in free space 
    see Novotny&Hecht 2nd Ed 10.25
    Input:
        k_rho - the radial component of the wavevector
        k - the wave vector in the medium
    
    One might think that the upward Green function differs from the downward
    due to all the (z<z0 and z>z0 dependent) signs that are present in e.g.
    eq 10.6 but the diagonal elements dont have this and the propagation 
    upwards or downwards both have a positive phase 
    accumulation (exp(1j*kz*|z-z0|))
    """ 
    kz=np.sqrt(k**2-k_rho**2)
    Gxx_s=1j/(8*np.pi*k**2)*k_rho*k**2/kz
    Gxx_p=-1j/(8*np.pi*k**2)*k_rho*kz
    return Gxx_p,Gxx_s

def G0_krho_zz(k_rho,k):
    """
    The zz element of the angle integrated green function in free space 
    see Novotny&Hecht 2nd Ed 10.25
    Input:
        k_rho - the radial component of the wavevector
        k - the wave vector in the medium
    
    One might think that the upward Green function differs from the downward
    due to all the (z<z0 and z>z0 dependent) signs that are present in f.i.
    eq 10.6 but the diagonal elements dont have this and the propagation 
    upwards or downwards both have a positive phase 
    accumulation (exp(1j*kz*|z-z0|))
    """
    kz=np.sqrt(k**2-k_rho**2)
    Gzz_p=1j/(8*np.pi*k**2)*(k_rho/kz)*(2*k_rho**2) 
    return Gzz_p

def Gref_krho_xx(k_rho,k0,MultiL_up,MultiL_down,d_up,d_down):
    """
    Calculates the reflected green function xx in k-space (the xx component 
    of the Green tensor). Inputs are:
        k_rho - the radial k vector component
        k - the vacuum wavevector
        MultiL_up - Multilayer above the dipole
        MultiL_down - Multilayer below the dipole
        d_up - the distance to the upper first interface
        d_down - the distance to the lower first interface

    One tricky thing is that the reflection coefficients as retrieved from
    the transfer matrix for S and P polarization have a sign difference for 
    k_rho=0 (perpendicular incidence). For this reason, following Hecht and
    Novotny the the equations for P differ by a minus sign from those for S 
    See Novotny & Hecht 2nd ed. eq. 10.16 ...
    """
    k=k0*np.sqrt(MultiL_up.eps[0])           #Changed wave number in embedding medium
    Gxx_p,Gxx_s=G0_krho_xx(k_rho,k) # the green function in the medium
    kz=np.sqrt(k**2-k_rho**2)       # z component of the k-vector
    
    # calculate the reflection coefficients at the upper side
    Sp,Ss=SM.SMatrixStack(k_rho,k0,MultiL_up)
    r12s_up=Ss[:,:,1,0]
    r12p_up=Sp[:,:,1,0]
    p_up=np.exp(1j*kz*d_up)     #upward propagation phase
    
    # calculate the reflection coefficients at the lower side
    Sp,Ss=SM.SMatrixStack(k_rho,k0,MultiL_down)
    r12s_down=Ss[:,:,1,0]
    r12p_down=Sp[:,:,1,0]
    p_down=np.exp(1j*kz*d_down) #downward propagation phase
    
    # below A and B (for both p and s polarisation are derived, by writing
    # out the geometric series for the total refelcted fields at the z-plane
    # where the dipole is located
    Ap=p_up**2*r12p_up
    Bp=p_down**2*r12p_down
 
    As=p_up**2*r12s_up
    Bs=p_down**2*r12s_down

    Rp=(Ap+Bp+2*Ap*Bp)/(1-Bp*Ap)
    Rs=(As+Bs+2*As*Bs)/(1-Bs*As)

    Gref_xx_p=Rp*Gxx_p
    Gref_xx_s=Rs*Gxx_s
    return np.squeeze(Gref_xx_p+Gref_xx_s)

def Gref_krho_zz(k_rho,k0,MultiL_up,MultiL_down,d_up,d_down):
    """
    Calculates the reflected green function zz in k-space (the zz component 
    of the Green tensor). Inputs are:
        k_rho - the radial k vector component
        k - the vacuum wavevector
        MultiL_up - Multilayer above the dipole
        MultiL_down - Multilayer below the dipole
        d_up - the distance to the upper first interface
        d_down - the distance to the lower first interface
    
    See also Novotny & Hecht 2nd ed. eq. 10.16
    """
    k=k0*np.sqrt(MultiL_up.eps[0])       # wave number in embedding medium
    Gzz_p=G0_krho_zz(k_rho,k)   # the G0zz in the medium
    kz=np.sqrt(k**2-k_rho**2)   # z component of the wavevector
    
    # calculate the refelection on the upper side
    Sp,Ss=SM.SMatrixStack(k_rho,k0,MultiL_up)
    r12p_up=Sp[:,:,1,0]
    p_up=np.exp(1j*kz*d_up)     # uward propagation
    
    Sp,Ss=SM.SMatrixStack(k_rho,k0,MultiL_down)
    r12p_down=Sp[:,:,1,0]
    p_down=np.exp(1j*kz*d_down) # downward propagation

    Ap=p_up**2*r12p_up
    Bp=p_down**2*r12p_down
 
    Rp=(Ap+Bp+2*Ap*Bp)/(1-Bp*Ap)
    
    Gref_zz_p=Rp*Gzz_p
    return np.squeeze(Gref_zz_p)

def LDOS(MultiL,LayerN,d_up,w):
    """
    Calculate the Normalised LDOS for a multilayer
    
    Input
    --------
    MultiL - is a multilayer class object
    LayerN - is the layer number of the dipole in the medium 
    d_up   - is its distance from the upper surface
    w      - is the angular frequency for which the calculation is done
    
    Output
    -------
    LDOSper: float
       Photonic LDOS for dipole perpendicular to the layers
    LDOSpar: float
       Photonic LDOS for dipole parallel to the layers
    """
    # input test
    if d_up>MultiL.d[LayerN]:
        raise ValueError('GreenFunction:G_krho_xx, Value of height dipole in medium is larger then layer thickness')

    # fundamental constants
    mu0=scipy.constants.mu_0 
    mu1=1.0
    c=scipy.constants.c
    eps0=scipy.constants.epsilon_0
    
    # other constants
    DipMomV=1               # value of the dipole moment (set to 1)
    k0=w/c                  # vacuum wavevector
    e1=MultiL.eps[LayerN]   # dielectric constant embedding (host) medium
    k=k0*np.sqrt(e1)        # embedding medium wave vector
    d_down=MultiL.d[LayerN]-d_up #downward distance
    #split the multilayer in upper and lower
    MultiL_up,MultiL_down=MultiL.splitUD(LayerN)
    #MultiL_up.d[0]=1e-20
    #MultiL_down.d[0]=1e-20
    
    # the integral over k_tho is split in two parts. the first is evading
    # the poles by moving through the negative complex plane    
    GAxx=gk.quadgk(Gref_krho_xx,0,2*k.real,k0,MultiL_up,MultiL_down,d_up,d_down,Waypoints=np.array([k-0.1j*k]))
    GBxx=gk.quadgk(Gref_krho_xx,2*k.real,np.Inf,k0,MultiL_up,MultiL_down,d_up,d_down)
    Gxx=GAxx[0]+GBxx[0]
    errx=(GAxx[1]+GBxx[1])/Gxx
    GAzz=gk.quadgk(Gref_krho_zz,0,2*k.real,k0,MultiL_up,MultiL_down,d_up,d_down,Waypoints=np.array([k-0.1j*k]))
    GBzz=gk.quadgk(Gref_krho_zz,2*k.real,np.Inf,k0,MultiL_up,MultiL_down,d_up,d_down)
    Gzz=GAzz[0]+GBzz[0] 
    errz=(GAzz[1]+GBzz[1])/Gzz
    ####### end testing impl
    #print(errz)
    # the scattered (reflected fields) at the dipole position    
    Es_xx=w**2*mu0*mu1*Gxx*DipMomV
    Es_zz=w**2*mu0*mu1*Gzz*DipMomV
    
    # calculate power of the scttered field
    Psxx=(w/2)*np.imag(np.conj(DipMomV)*Es_xx) # 8.74
    Pszz=(w/2)*np.imag(np.conj(DipMomV)*Es_zz) # 8.74
    
    # the power in the host medium homgeneous space
    P0=DipMomV**2*w*(w*np.sqrt(e1)/c)**3/(12*np.pi*eps0*e1)
    #print(P0,Pszz)
    # normalised LDOS to isotropic host medium
    LDOSper=np.real((Pszz+P0)/P0)
    LDOSpar=np.real((Psxx+P0)/P0)
    return LDOSper, LDOSpar

def LDOScol(MultiL,LayerN,d_up,w,NA):
    """
    Calculate the Normalised collection efficiency for a multilayer,
    assuming the LDOS integral until k_NA
    MultiL - is a multilayer class object
    LayerN - is the layer number of the dipole in the medium 
    d_up   - is its distance from the upper surface
    w      - is the angular frequency for which the calculation is done
    """
    # input test
    if d_up>MultiL.d[LayerN]:
        raise ValueError('GreenFunction:G_krho_xx, Value of height dipole in medium is larger then layer thickness')

    # fundamental constants
    mu0=scipy.constants.mu_0 
    mu1=1.0
    c=scipy.constants.c
    eps0=scipy.constants.epsilon_0
    
    # other constants
    DipMomV=1               # value of the dipole moment (set to 1)
    k0=w/c                  # vacuum wavevector
    e1=MultiL.eps[LayerN]   # dielectric constant embedding (host) medium
    k=k0*np.sqrt(e1)        # embedding medium wave vector
    d_down=MultiL.d[LayerN]-d_up #downward distance
    #split the multilayer in upper and lower
    MultiL_up,MultiL_down=MultiL.splitUD(LayerN)
    #MultiL_up.d[0]=1e-20
    #MultiL_down.d[0]=1e-20
    
    # the integral over k_tho is split in two parts. the first is evading
    # the poles by moving through the negative complex plane    
    GAxx=gk.quadgk(Gref_krho_xx,0,NA*k.real/np.real(np.sqrt(e1)),k0,MultiL_up,MultiL_down,d_up,d_down,Waypoints=np.array([k-0.1j*k]))
    #GBxx=gk.quadgk(Gref_krho_xx,2*k.real,np.Inf,k0,MultiL_up,MultiL_down,d_up,d_down)
    Gxx=GAxx[0]#+GBxx[0]
    errx=(GAxx[1])/Gxx
    GAzz=gk.quadgk(Gref_krho_zz,0,NA*k.real/np.real(np.sqrt(e1)),k0,MultiL_up,MultiL_down,d_up,d_down,Waypoints=np.array([k-0.1j*k]))
    #GBzz=gk.quadgk(Gref_krho_zz,2*k.real,np.Inf,k0,MultiL_up,MultiL_down,d_up,d_down)
    Gzz=GAzz[0]#+GBzz[0] 
    errz=(GAzz[1])/Gzz
    ####### end testing impl
    #print(errz)
    # the scattered (reflected fields) at the dipole position    
    Es_xx=w**2*mu0*mu1*Gxx*DipMomV
    Es_zz=w**2*mu0*mu1*Gzz*DipMomV
    
    # calculate power of the scttered field
    Psxx=(w/2)*np.imag(np.conj(DipMomV)*Es_xx) # 8.74
    Pszz=(w/2)*np.imag(np.conj(DipMomV)*Es_zz) # 8.74
    
    # the power in the host medium homgeneous space
    P0=DipMomV**2*w*(w*np.sqrt(e1)/c)**3/(12*np.pi*eps0*e1)
    #print(P0,Pszz)
    # normalised LDOS to isotropic host medium
    Fracper=np.real((Pszz+P0)/P0)
    Fracpar=np.real((Psxx+P0)/P0)
    return Fracper, Fracpar

def Gtrans_krho_zz(k_rho,k0,MultiL_up,MultiL_down,d_up,d_down):
    """
    Calculates the transmitted green function zz in k-space (the zz component 
    of the Green tensor). Inputs are:
        k_rho - the radial k vector component
        k0 - the vacuum wavevector
        MultiL_up - Multilayer above the dipole
        MultiL_down - Multilayer below the dipole
        d_up - the distance to the upper first interface
        d_down - the distance to the lower first interface
    
    See also Novotny & Hecht 2nd ed. eq. 10.18
    """
    # fundamental constants
    mu0=scipy.constants.mu_0 
    mu1=1.0
    c=scipy.constants.c
    eps0=scipy.constants.epsilon_0
    
    w=c*k0/(2*np.pi)
    
    eps_up=MultiL_up.eps[len(MultiL_up.eps)-1]
    k=k0*MultiL_up.eps[0]       # wave number in embedding medium
    kz=np.sqrt(k**2-k_rho**2)   # z component of the wavevector
    Gzz_p=1j/(8*np.pi**2)*(k_rho**2/(kz*k**2))

    # calculate the refelection on the upper side
    Sp,Ss=SM.SMatrixStack(k_rho,k0,MultiL_up)
    r12p_up=Sp[:,:,1,0]
    t12p_up=Sp[:,:,0,0]
    p_up=np.exp(1j*kz*d_up)     # uward propagation
    
    
    Sp,Ss=SM.SMatrixStack(k_rho,k0,MultiL_down)
    r12p_down=Sp[:,:,1,0]
    t12p_down=Sp[:,:,0,0]
    p_down=np.exp(1j*kz*d_down) # downward propagation
    
    Gtrans_up_zz=Gzz_p*t12p_up*p_up*(1+p_down**2*r12p_down)/(1-r12p_down*r12p_up*p_up**2*p_down**2)
    Gtrans_down_zz=Gzz_p*t12p_down*p_down*(1+p_up**2*r12p_up)/(1-r12p_up*r12p_down*p_down**2*p_up**2)
    
#    print(np.abs(Gtrans_up_zz))
#    print(np.abs(Gzz_p))
    
    DipMomV=1   
    Es=w**2*mu0*mu1*Gtrans_up_zz*DipMomV
    S=0.5*np.sqrt(eps0*eps_up/(mu0*mu1))*np.abs(Es)**2*k_rho*2*np.pi
    return np.abs(np.squeeze(S))# np.squeeze(Gtrans_down_zz)#,Gtrans_down_zz

def G0_krho_phi_zz(k_rho,k):
    """
    The zz element of the NOT angle integrated green function in free space 
    see Novotny&Hecht 2nd Ed 10.15
    Input:
        k_rho - the radial component of the wavevector
        k - the wave vector in the medium
    
    One might think that the upward Green function differs from the downward
    due to all the (z<z0 and z>z0 dependent) signs that are present in f.i.
    eq 10.6 but the diagonal elements dont have this and the propagation 
    upwards or downwards both have a positive phase 
    accumulation (exp(1j*kz*|z-z0|))
    """

    kz=np.sqrt(k**2-k_rho**2)
    Gzz_p=1j/(8*np.pi**2)*(k_rho**2/(kz*k**2))
    return Gzz_p

