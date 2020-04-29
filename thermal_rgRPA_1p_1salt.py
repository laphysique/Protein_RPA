# Apr 27, 2020
# - Upload to GitHub
# - FH function becomes an item in HP dictionary

# Dec 9, 2019
# - Add intMax, phi_min_calc as tunable parameters 

# Feb 9, 2019
# - eh, es are naive FH interaction parameters. 
#   Add a chi = eh/T + es to total free energy without
#   considering its influence on w2 in RPA free energy

# Dec 25, 2018
# - Beyond-RPA: 2-field renormalization
#   following Fredrickson's field theory
# - Summing all (i,j) terms in correlation function
#   with the same |i-j| in advance
# - Smooth short-range potential in free energy 
#   but not in effective Kuhn length
# - Different salt setting

import numpy as np
import scipy.optimize as sco
import scipy.integrate as sci
from numpy import pi

useleff = True
intMax = 200
phi_min_calc = 1e-12

eh, es = 0,0

#------------------------ Entropy auxiliary functions -------------------------

def s_calc(x):
    return (x > phi_min_calc)*x*np.log(x + (x < phi_min_calc)) + 1e5*(x<0)
   
def ds_calc(x):
    return (np.log(x + 1e5*(x<=0)) + 1)*(x>0) 
  
def dds_calc(x):
    return (x>0)/(x + (x==0))

#-------------------------- Heteropolymer parameters --------------------------

# FH_funs = (f, df, ddf)
# f = f(phi,u) is the customized short-range interaction energy 
# df = df/dphi, ddf = ddf/ddphi
def Heteropolymer(sigma, zs=1, zc=1, w2=4*pi/3, wd=1/6, FH_funs=None):
    sig = np.array(sigma)
    N   = sig.shape[0]
    pc  = np.abs( np.sum(sig)/N ) 
    Q  = np.sum( sig*sig )/N 
    IN = np.eye(N)

    mel = np.kron(sig, sig).reshape((N, N))
    Tel = np.array([ np.sum(mel.diagonal(n) + mel.diagonal(-n)) \
                    for n in range(N)])
    Tel[0] /= 2
    Tex = 2*np.arange(N,0,-1)
    Tex[0] /= 2        
    mlx = np.kron(sig, np.ones(N)).reshape((N, N))
    Tlx = np.array([ np.sum(mlx.diagonal(n) + mlx.diagonal(-n)) \
                    for n in range(N)])  
    Tlx[0] /= 2

    L = np.arange(N)
    L2 = L*L


    HP =   {'sig': sig,    \
            'zs' : zs,     \
            'zc' : zc,     \
            'w2' : w2,     \
            'wd' : wd,     \
            'N'  : N,      \
            'pc' : pc,     \
            'Q'  : Q,      \
            'IN' : IN,     \
            'L'  : L,      \
            'L2' : L2,     \
            'Tel': Tel,    \
            'Tex': Tex,    \
            'Tlx': Tlx   
           }    

  
    # Default is a Flory-Huggins model
    if FH_funs is None:
        HP['FH']   = lambda phi, u: (w2/2 - eh*u - es)*phi*phi
        HP['dFH']  = lambda phi, u: (w2 - 2*eh*u - 2*es)*phi
        HP['ddFH'] = lambda phi, u: (w2 - 2*eh*u - 2*es)
    else: 
        HP['FH']   = lambda phi, u: FH_fun[0](phi,u) + w2/2*phi*phi
        HP['dFH']  = lambda phi, u: FH_fun[1](phi,u) + w2*phi
        HP['ddFH'] = lambda phi, u: FH_fun[2](phi,u) + w2

    return HP

#---------------------------------- Entropy -----------------------------------

# Entropy
def enp(HP, phi, phis):
    phic = (HP['pc']*phi + HP['zs']*phis)/HP['zc']
    return s_calc(phi)/HP['N']  \
               + s_calc(phic)   \
               + s_calc(phis)   \
               + s_calc( 1-phi-phic-phis ) 


# Derivatives of entropy
def denp_p(HP, phi, phis):
    phic = (HP['pc']*phi + HP['zs']*phis)/HP['zc']
    return ( ds_calc(phi)/HP['N'] + HP['pc']/HP['zc']*ds_calc(phic) \
             - (1+HP['pc']/HP['zc'])*ds_calc(1-phi-phic-phis) )*(phi>0) 

def denp_s(HP, phi, phis):
    phic = (HP['pc']*phi + HP['zs']*phis)/HP['zc']
    return ( HP['zs']/HP['zc']*ds_calc(phic) + ds_calc(phis) \
             - (1+HP['zs']/HP['zc'])*ds_calc(1-phi-phic-phis)  )*(phis>0)

# 2nd derivatives of entropy
def ddenp_pp(HP, phi, phis):
    phic = (HP['pc']*phi + HP['zs']*phis)/HP['zc']
    return ( dds_calc(phi)/HP['N'] \
             + HP['pc']*HP['pc']/HP['zc']/HP['zc']*dds_calc(phic) \
             + (1+HP['pc']/HP['zc'])*(1+HP['pc']/HP['zc'])*dds_calc(1-phi-phic-phis) \
           )*(phi>0)

def ddenp_ss(HP, phi, phis):
    phic = (HP['pc']*phi + HP['zs']*phis)/HP['zc']
    return ( HP['zs']*HP['zs']/HP['zc']/HP['zc']*dds_calc(phic) \
             + dds_calc(phis) \
             + (1+HP['zs']/HP['zc'])*(1+HP['zs']/HP['zc'])*dds_calc(1-phi-phic-phis) \
           )*(phis>0)
  
def ddenp_ps(HP, phi, phis):
    phic = (HP['pc']*phi + HP['zs']*phis)/HP['zc']
    return ( HP['pc']*HP['zs']/HP['zc']/HP['zc']*dds_calc(phic) \
             + (1+HP['pc']/HP['zc'])*(1+HP['zs']/HP['zc'])*dds_calc(1-phi-phic-phis) \
           )*(phi>0)*(phis>0)

#------------------------ Screening potential of ions -------------------------

def fscr(HP,phi,phis, u):
    phiscr = HP['zs']*(HP['zs']+HP['zc'])*phis + HP['zc']*HP['pc']*phi 
    kappa = np.sqrt(4*pi*u*phiscr)
    return -(np.log(1+kappa)-kappa+kappa*kappa/2)/4/pi 

# 1st derivatives of screening potential of ions
def dfscr(HP,phi,phis,u):
    Zs = HP['zs']*(HP['zs']+HP['zc'])
    Zp = HP['zc']*HP['pc']
    phiscr = Zs*phis + Zp*phi 
    kappa = np.sqrt(4*pi*u*phiscr)

    temp =  -kappa/2/(1+kappa)*u
    return temp*Zp*(phi>0), temp*Zs*(phis>0)

# 2nd derivatives of screening potential of ions
def ddfscr(HP, phi, phis, u):
    Zs = HP['zs']*(HP['zs']+HP['zc'])
    Zp = HP['zc']*HP['pc']
    phiscr = Zs*phis + Zp*phi 
    kappa = np.sqrt(4*pi*u*phiscr)

    tp = Zp/(1+kappa)*(phi>0)
    ts = Zs/(1+kappa)*(phis>0)
    temp = -pi*u*u/( kappa + (kappa==0) )*(kappa>0)

    return temp*tp*tp, temp*ts*ts, temp*tp*ts

#------------------- Polymer structure, many-body potential -------------------

# Structure factors
def G11(HP, k, x):
    return np.mean( HP['Tex']*np.exp(-k*k*x*HP['L']/6) )

def G1c(HP, k, x):
    return np.mean( HP['Tlx']*np.exp(-k*k*x*HP['L']/6) )    

def Gcc(HP, k, x):
    return np.mean( HP['Tel']*np.exp(-k*k*x*HP['L']/6) ) 

def L2G11(HP, k, x):
    return np.sum( HP['Tex']*HP['L2']*np.exp(-k*k*x*HP['L']/6) )
    
def L2G1c(HP, k, x):
    return np.sum( HP['Tlx']*HP['L2']*np.exp(-k*k*x*HP['L']/6) ) 

def L2Gcc(HP, k, x):
    return np.sum( HP['Tel']*HP['L2']*np.exp(-k*k*x*HP['L']/6) ) 


# 1st derivatives of structure factors 
def dG11(HP, k, x):
    return -k*k/6*np.mean( HP['Tex']*HP['L']*np.exp(-k*k*x*HP['L']/6) )

def dG1c(HP, k, x):
    return -k*k/6*np.mean( HP['Tlx']*HP['L']*np.exp(-k*k*x*HP['L']/6) )    

def dGcc(HP, k, x):
    return -k*k/6*np.mean( HP['Tel']*HP['L']*np.exp(-k*k*x*HP['L']/6) ) 

def dL2G11(HP, k, x):
    return -k*k/6*np.sum( HP['Tex']*HP['L']*HP['L2']*np.exp(-k*k*x*HP['L']/6) )
    
def dL2G1c(HP, k, x):
    return -k*k/6*np.sum( HP['Tlx']*HP['L']*HP['L2']*np.exp(-k*k*x*HP['L']/6) ) 

def dL2Gcc(HP, k, x):
    return -k*k/6*np.sum( HP['Tel']*HP['L']*HP['L2']*np.exp(-k*k*x*HP['L']/6) ) 


# 2nd derivatives of structure factors 
def ddG11(HP, k, x):
    return k*k*k*k/36*np.mean( HP['Tex']*HP['L2']*np.exp(-k*k*x*HP['L']/6) )

def ddG1c(HP, k, x):
    return k*k*k*k/36*np.mean( HP['Tlx']*HP['L2']*np.exp(-k*k*x*HP['L']/6) )    

def ddGcc(HP, k, x):
    return k*k*k*k/36*np.mean( HP['Tel']*HP['L2']*np.exp(-k*k*x*HP['L']/6) ) 

def ddL2G11(HP, k, x):
    return k*k*k*k/36*np.sum( HP['Tex']*HP['L2']*HP['L2']*np.exp(-k*k*x*HP['L']/6) )
    
def ddL2G1c(HP, k, x):
    return k*k*k*k/36*np.sum( HP['Tlx']*HP['L2']*HP['L2']*np.exp(-k*k*x*HP['L']/6) ) 

def ddL2Gcc(HP, k, x):
    return k*k*k*k/36*np.sum( HP['Tel']*HP['L2']*HP['L2']*np.exp(-k*k*x*HP['L']/6) ) 

#-------------------------------------

# nu in potential 
def nu(HP,k,phi,phis,u):
    Zs = HP['zs']*(HP['zs']+HP['zc'])
    Zp = HP['zc']*HP['pc']
    return k*k/(4*pi*u) + Zs*phis + Zp*phi

#-------------------------------------

# The v*det(Delta^-1)
# For calculating x:    for_f=False
# For calculating frpa: for_f=True

def vDel(HP, k, phi, phis, u, x, for_f=False):
    g, xi, zeta = G11(HP, k, x), Gcc(HP, k, x), G1c(HP, k, x) 
    nuu, v = nu(HP,k,phi,phis,u), HP['w2']*np.exp(-for_f*HP['wd']*k*k)

    return nuu + phi*(g*nuu*v + xi) + v*phi*phi*(g*xi-zeta*zeta)


#------------------------- Effective Kuhn length -------------------------

# Calculate effective Kuhn length x
def x_eff(HP, phi, phis, u):
    return sco.brenth(x_to_solve, 1/HP['N']/10, 100*HP['N'], \
                      args=(HP,phi,phis,u))

def x_to_solve(x, HP, phi, phis, u):
    return 1 - 1/x - sci.quad(x_RG_to_int, 0, np.inf, \
                              args=(HP, phi, phis, u, x), limit=intMax)[0] \
                              /(18*(HP['N']-1))
    
def x_RG_to_int(k, HP, phi, phis, u, x): 
    g, xi, zeta = G11(HP, k, x), Gcc(HP, k, x), G1c(HP, k, x) 
    A11, A1c, Acc = L2G11(HP,k,x), L2G1c(HP,k,x), L2Gcc(HP,k,x)
    nuu, v = nu(HP,k,phi,phis,u), HP['w2'] 

    I = Acc + nuu*v*A11 + phi*v*(g*Acc - 2*zeta*A1c + xi*A11)
    D = vDel(HP,k,phi,phis,u,x)
    return k*k/(2*pi*pi)*k*k*I/D


# 1st derivatives of x -------------------------------------
def dx_eff(HP, phi, phis, u, x=None):
    x = x_eff(HP,phi,phis,u) if x==None else x
   
    beta = 18*(HP['N']-1)
    Dd = sci.quad(Dd_int, 0, np.inf, args=(HP, phi, phis, u, x), limit=intMax)[0] 
    Np = sci.quad(Np_int, 0, np.inf, args=(HP, phi, phis, u, x), limit=intMax)[0]
    Ns = sci.quad(Ns_int, 0, np.inf, args=(HP, phi, phis, u, x), limit=intMax)[0]

    dxp = Np/(beta/x/x-Dd)
    dxs = Ns/(beta/x/x-Dd)

    return dxp, dxs


def Dd_int(k, HP, phi, phis, u, x):
    g, xi, zeta      = G11(HP, k, x), Gcc(HP, k, x), G1c(HP, k, x) 
    dg, dxi, dzeta   = dG11(HP, k, x), dGcc(HP, k, x), dG1c(HP, k, x) 
    A11, A1c, Acc    = L2G11(HP,k,x), L2G1c(HP,k,x), L2Gcc(HP,k,x)
    dA11, dA1c, dAcc = dL2G11(HP,k,x), dL2G1c(HP,k,x), dL2Gcc(HP,k,x) 
    nuu, v = nu(HP,k,phi,phis,u), HP['w2']

    N1 = dA11*nuu*v + dAcc + \
         phi*v*(A11*dxi - 2*A1c*dzeta + Acc*dg + dA11*xi - 2*dA1c*zeta + dAcc*g)
    N2 = ( phi*phi*v*(dg*xi + dxi*g - 2*dzeta*zeta) + phi*(dg*nuu*v + dxi)) \
         *(A11*nuu*v + Acc + phi*v*(A11*xi - 2*A1c*zeta + Acc*g))
    D  = vDel(HP,k,phi,phis,u,x)

    return k*k/(2*pi*pi)*k*k*( N1*D - N2 )/D/D

def Np_int(k, HP, phi, phis, u, x):
    g,  xi,  zeta    = G11(HP, k, x), Gcc(HP, k, x), G1c(HP, k, x) 
    A11, A1c, Acc    = L2G11(HP,k,x), L2G1c(HP,k,x), L2Gcc(HP,k,x)
    nuu, v, Zp = nu(HP,k,phi,phis,u), HP['w2'], HP['zc']*HP['pc']

    N1 = A11*Zp*v + v*(A11*xi - 2*A1c*zeta + Acc*g) 
    N2 =  (A11*nuu*v + Acc + phi*v*(A11*xi - 2*A1c*zeta + Acc*g)) \
          *( g*nuu*v + g*Zp*phi*v + Zp + 2*phi*v*(g*xi - zeta*zeta) + xi)
    D  = vDel(HP,k,phi,phis,u,x)

    return k*k/(2*pi*pi)*k*k*( N1*D - N2 )/D/D    


def Ns_int(k, HP, phi, phis, u, x):
    g,  xi,  zeta    = G11(HP, k, x), Gcc(HP, k, x), G1c(HP, k, x) 
    A11, A1c, Acc    = L2G11(HP,k,x), L2G1c(HP,k,x), L2Gcc(HP,k,x) 
    nuu, v, Zs = nu(HP,k,phi,phis,u), HP['w2'], HP['zs']*(HP['zs']+HP['zc'])

    N1 = A11*Zs*v
    N2 = (A11*nuu*v + Acc + phi*v*(A11*xi - 2*A1c*zeta + Acc*g)) \
         *( g*Zs*phi*v + Zs )
    D  = vDel(HP,k,phi,phis,u,x)

    return k*k/(2*pi*pi)*k*k*( N1*D - N2 )/D/D


# 2nd derivatives of x -------------------------------------
def ddx_eff(HP, phi, phis, u, x=None, dx_all=None):
    x = x_eff(HP,phi,phis,u) if x==None else x
    dxp, dxs = dx_eff(HP, phi, phis, u, x) if dx_all==None else dx_all

    beta = 18*(HP['N']-1)
    Ddd = sci.quad(Ddd_int, 0, np.inf, args=(HP, phi, phis, u, x), limit=intMax)[0]
    Npp = 0 if phi==0  else sci.quad(Npp_int, 0, np.inf, \
                                     args=(HP, phi, phis, u, x, dxp), limit=intMax)[0]
    Nss = 0 if phis==0 else sci.quad(Nss_int, 0, np.inf, \
                                     args=(HP, phi, phis, u, x, dxs), limit=intMax)[0] 
    Nps = 0 if phi*phis==0 else \
          sci.quad(Nps_int, 0, np.inf, args=(HP, phi, phis, u, x, dxp, dxs), \
                   limit=intMax)[0]

    ddxpp = ( 2*beta*dxp*dxp/x/x/x + Npp )/( beta/x/x - Ddd )*(phi>0)
    ddxss = ( 2*beta*dxs*dxs/x/x/x + Nss )/( beta/x/x - Ddd )*(phis>0)
    ddxps = ( 2*beta*dxp*dxs/x/x/x + Nps )/( beta/x/x - Ddd )*(phi>0)*(phis>0)    

    return ddxpp, ddxss, ddxps

def Ddd_int(k, HP, phi, phis, u, x):
    g, xi, zeta      = G11(HP, k, x), Gcc(HP, k, x), G1c(HP, k, x) 
    dg, dxi, dzeta   = dG11(HP, k, x), dGcc(HP, k, x), dG1c(HP, k, x) 
    A11, A1c, Acc    = L2G11(HP,k,x), L2G1c(HP,k,x), L2Gcc(HP,k,x)
    dA11, dA1c, dAcc = dL2G11(HP,k,x), dL2G1c(HP,k,x), dL2Gcc(HP,k,x) 
    nuu, v = nu(HP,k,phi,phis,u), HP['w2']

    N1 = dA11*nuu*v + dAcc \
         + phi*v*(A11*dxi - 2*A1c*dzeta + Acc*dg + dA11*xi + dAcc*g - 2*dA1c*zeta)
    N2 = (phi*phi*v*(dg*xi + dxi*g - 2*dzeta*zeta) + phi*(dg*nuu*v + dxi)) \
         *(A11*nuu*v + Acc + phi*v*(A11*xi - 2*A1c*zeta + Acc*g))
    D  = vDel(HP,k,phi,phis,u,x)

    return k*k/(2*pi*pi)*k*k*( N1*D - N2 )/D/D
           

def Npp_int(k, HP, phi, phis, u, x, dxp):
    g, xi, zeta      = G11(HP, k, x), Gcc(HP, k, x), G1c(HP, k, x) 
    dg, dxi, dzeta   = dG11(HP, k, x), dGcc(HP, k, x), dG1c(HP, k, x) 
    ddg, ddxi, ddzeta= ddG11(HP, k, x), ddGcc(HP, k, x), ddG1c(HP, k, x) 
    A11, A1c, Acc    = L2G11(HP,k,x), L2G1c(HP,k,x), L2Gcc(HP,k,x)
    dA11, dA1c, dAcc = dL2G11(HP,k,x), dL2G1c(HP,k,x), dL2Gcc(HP,k,x) 
    ddA11, ddA1c, ddAcc = ddL2G11(HP,k,x), ddL2G1c(HP,k,x), ddL2Gcc(HP,k,x) 
    nuu, v, Zp = nu(HP,k,phi,phis,u), HP['w2'], HP['zc']*HP['pc']


    I   = A11*nuu*v + Acc + phi*v*(A11*xi - 2*A1c*zeta + Acc*g) 
    dI  = (dxi*A11+xi*dA11)-2*(dzeta*A1c+zeta*dA1c)+(g*dAcc+dg*Acc)
    ddI = ddxi*A11+2*dxi*dA11+xi*ddA11 \
          - 2*(ddzeta*A1c+2*dzeta*dA1c+zeta*ddA1c) \
          + g*ddAcc+2*dg*dAcc+ddg*Acc 
    Ip  = v*Zp*A11 + (v*nuu*dA11+dAcc)*dxp + v*(xi*A11-2*zeta*A1c+g*Acc) + dxp*v*phi*dI
    D   = vDel(HP,k,phi,phis,u,x)   
    Dp  = Zp + (g*nuu*v+xi) + phi*(dxp*(dg*nuu*v+dxi)+g*Zp*v) \
          + 2*phi*v*(g*xi-zeta*zeta) \
          + v*phi*phi*(dg*xi + g*dxi - 2*dzeta*zeta)*dxp

    N1 = 2*v*Zp*dA11*dxp + (v*nuu*ddA11+ddAcc)*dxp*dxp \
         + 2*v*dxp*dI + dxp*dxp*v*phi*ddI
    N2 = 2*(v*Zp*g+(dg*nuu*v+dxi)*dxp) \
         + phi*( (ddg*nuu*v+ddxi)*dxp*dxp + 2*v*Zp*dg*dxp ) \
         + 2*v*(g*xi-zeta*zeta) + 4*v*phi*(g*dxi+dg*xi-2*zeta*dzeta)*dxp \
         + v*phi*phi*( g*ddxi+2*dg*dxi+ddg*xi-2*(dzeta*dzeta+zeta*ddzeta) )*dxp*dxp 
    N3 = 2*Ip*Dp
    N4 = 2*I*Dp*Dp
 
    return k*k/(2*pi*pi)*k*k*( N1*D*D - (I*N2+N3)*D + N4 )/D/D/D
               
def Nss_int(k, HP, phi, phis, u, x, dxs):
    g, xi, zeta      = G11(HP, k, x), Gcc(HP, k, x), G1c(HP, k, x) 
    dg, dxi, dzeta   = dG11(HP, k, x), dGcc(HP, k, x), dG1c(HP, k, x) 
    ddg, ddxi, ddzeta= ddG11(HP, k, x), ddGcc(HP, k, x), ddG1c(HP, k, x) 
    A11, A1c, Acc    = L2G11(HP,k,x), L2G1c(HP,k,x), L2Gcc(HP,k,x)
    dA11, dA1c, dAcc = dL2G11(HP,k,x), dL2G1c(HP,k,x), dL2Gcc(HP,k,x) 
    ddA11, ddA1c, ddAcc = ddL2G11(HP,k,x), ddL2G1c(HP,k,x), ddL2Gcc(HP,k,x) 
    nuu, v, Zs = nu(HP,k,phi,phis,u), HP['w2'], HP['zs']*(HP['zs']+HP['zc'])

    I   = A11*nuu*v + Acc + phi*v*(A11*xi - 2*A1c*zeta + Acc*g) 
    dI  = (dxi*A11+xi*dA11)-2*(dzeta*A1c+zeta*dA1c)+(g*dAcc+dg*Acc)
    ddI = ddxi*A11+2*dxi*dA11+xi*ddA11 \
            - 2*(ddzeta*A1c+2*dzeta*dA1c+zeta*ddA1c) \
            + g*ddAcc+2*dg*dAcc+ddg*Acc 
    Is = v*Zs*A11 + (v*nuu*dA11+dAcc)*dxs + dxs*v*phi*dI
    D  = vDel(HP,k,phi,phis,u,x)
    Ds = Zs + phi*(g*Zs*v+(dg*nuu*v+dxi)*dxs) \
         + v*phi*phi*(dg*xi + g*dxi - 2*dzeta*zeta)*dxs
 
    N1 = 2*v*Zs*dA11*dxs + (v*nuu*ddA11+ddAcc)*dxs*dxs + dxs*dxs*v*phi*ddI
    N2 = phi*( (ddg*nuu*v+ddxi)*dxs*dxs + 2*v*Zs*dg*dxs ) \
         + v*phi*phi*( g*ddxi+2*dg*dxi+ddg*xi-2*(dzeta*dzeta+zeta*ddzeta) )*dxs*dxs
    N3 = 2*Is*Ds
    N4 = 2*I*Ds*Ds

    return k*k/(2*pi*pi)*k*k*( N1*D*D - (I*N2+N3)*D + N4 )/D/D/D

               
def Nps_int(k, HP, phi, phis, u, x, dxp, dxs):
    g, xi, zeta      = G11(HP, k, x), Gcc(HP, k, x), G1c(HP, k, x) 
    dg, dxi, dzeta   = dG11(HP, k, x), dGcc(HP, k, x), dG1c(HP, k, x) 
    ddg, ddxi, ddzeta= ddG11(HP, k, x), ddGcc(HP, k, x), ddG1c(HP, k, x) 
    A11, A1c, Acc    = L2G11(HP,k,x), L2G1c(HP,k,x), L2Gcc(HP,k,x)
    dA11, dA1c, dAcc = dL2G11(HP,k,x), dL2G1c(HP,k,x), dL2Gcc(HP,k,x) 
    ddA11, ddA1c, ddAcc = ddL2G11(HP,k,x), ddL2G1c(HP,k,x), ddL2Gcc(HP,k,x) 
    nuu, v, Zp, Zs = nu(HP,k,phi,phis,u), HP['w2'], HP['zc']*HP['pc'], HP['zs']*(HP['zs']+HP['zc'])

    I   = A11*nuu*v + Acc + phi*v*(A11*xi - 2*A1c*zeta + Acc*g) 
    dI  = (dxi*A11+xi*dA11)-2*(dzeta*A1c+zeta*dA1c)+(g*dAcc+dg*Acc)
    ddI = ddxi*A11+2*dxi*dA11+xi*ddA11 \
            - 2*(ddzeta*A1c+2*dzeta*dA1c+zeta*ddA1c) \
            + g*ddAcc+2*dg*dAcc+ddg*Acc 
    Ip  = v*Zp*A11 + (v*nuu*dA11+dAcc)*dxp + v*(xi*A11-2*zeta*A1c+g*Acc) + dxp*v*phi*dI
    Is  = v*Zs*A11 + (v*nuu*dA11+dAcc)*dxs + dxs*v*phi*dI
    D   = vDel(HP,k,phi,phis,u,x)
    Dp  = Zp + (g*nuu*v+xi) + phi*(dxp*(dg*nuu*v+dxi)+g*Zp*v) \
            + 2*phi*v*(g*xi-zeta*zeta) \
            + v*phi*phi*(dg*xi + g*dxi - 2*dzeta*zeta)*dxp
    Ds  = Zs + phi*(g*Zs*v+(dg*nuu*v+dxi)*dxs) \
            + v*phi*phi*(dg*xi + g*dxi - 2*dzeta*zeta)*dxs

    N1 = v*dA11*(Zs*dxp+Zp*dxs) + (v*nuu*ddA11+ddAcc)*dxp*dxs \
         + dxs*v*dI + dxs*dxp*v*phi*ddI
    N2 = g*Zs*v + (dg*nuu*v+dxi)*dxs \
         + phi*( v*dg*(Zs*dxp+Zp*dxs) + (ddg*nuu*v+ddxi)*dxp*dxs ) \
         + 2*v*phi*(dg*xi+g*dxi-2*zeta*dzeta)*dxs \
         + v*phi*phi*( g*ddxi+2*dg*dxi+ddg*xi-2*(dzeta*dzeta+zeta*ddzeta) )*dxp*dxs
    N3 = Is*Dp + Ip*Ds
    N4 = 2*I*Dp*Ds  

    return k*k/(2*pi*pi)*k*k*( N1*D*D - (I*N2+N3)*D + N4 )/D/D/D


#------------------------- Polymer free energy -------------------------

# polymer free energy
def frpa(HP,phi,phis,u,x=None):
    if useleff:
        x=x_eff(HP, phi, phis, u) if x==None else x
    else:
        x=1

    Qp = HP['Q']*phi
    Pp = HP['zc']*HP['pc']*phi + HP['zs']*(HP['zs']+HP['zc'])*phis
    f1 = sci.quad( frpa_int, 0, np.inf, args=(HP,phi,phis,u,x), limit=intMax)[0]
    f2 = -(2*np.sqrt(pi)/3)*( ((Qp+Pp)*u)**(3/2) - (Pp*u)**(3/2) )
    return f1 + f2

def frpa_int(k,HP,phi,phis,u,x):
    nuu = nu(HP, k, phi, phis, u)
    A = vDel(HP, k, phi, phis, u, x, for_f=True)/nuu
    B = 1+HP['Q']*phi/nuu
    return k*k/(4*pi*pi)*( np.log(A/B) )#-np.log(B) ) 

# 1st derivatives of polymer free energy -------------------------------------
def dfrpa(HP,phi,phis,u, x=None, dx_all=None):
    if useleff: 
        x = x_eff(HP, phi, phis, u) if x==None else x
        dxp, dxs = dx_eff(HP, phi,phis,u,x) if dx_all==None else dx_all
    else:    
        x, dxp, dxs = 1,0,0

    dfp = 0 if phi==0  else sci.quad(dfrpa_p_int, 0, np.inf,\
                                     args=(HP,phi,phis,u,x,dxp), limit=intMax)[0] 
    dfs = 0 if phis==0 else sci.quad(dfrpa_s_int, 0, np.inf,\
                                     args=(HP,phi,phis,u,x,dxs), limit=intMax)[0] 

    return dfp, dfs        

def dfrpa_p_int(k,HP,phi,phis,u,x,dxp):
    g, xi, zeta      = G11(HP, k, x), Gcc(HP, k, x), G1c(HP, k, x) 
    dg, dxi, dzeta   = dG11(HP, k, x), dGcc(HP, k, x), dG1c(HP, k, x) 
    nuu, Zp = nu(HP,k,phi,phis,u), HP['zc']*HP['pc']
    v = HP['w2']*np.exp(-HP['wd']*k*k)
  
    D   = vDel(HP,k,phi,phis,u,x, for_f=True)
    Dp  = Zp + (g*nuu*v+xi) + phi*(dxp*(dg*nuu*v+dxi)+g*Zp*v) \
             + 2*phi*v*(g*xi-zeta*zeta) \
             + v*phi*phi*(dg*xi + g*dxi - 2*dzeta*zeta)*dxp

    return 1/(4*pi*pi)*( k*k*(Dp/D - Zp/nuu) - 4*pi*u*HP['Q'] )  

def dfrpa_s_int(k,HP,phi,phis,u,x,dxs):
    g, xi, zeta      = G11(HP, k, x), Gcc(HP, k, x), G1c(HP, k, x) 
    dg, dxi, dzeta   = dG11(HP, k, x), dGcc(HP, k, x), dG1c(HP, k, x) 
    nuu, Zs = nu(HP,k,phi,phis,u), HP['zs']*(HP['zs']+HP['zc'])
    v = HP['w2']*np.exp(-HP['wd']*k*k)
    
    D   = vDel(HP,k,phi,phis,u,x, for_f=True)
    Ds  = Zs + phi*(g*Zs*v+(dg*nuu*v+dxi)*dxs) \
            + v*phi*phi*(dg*xi + g*dxi - 2*dzeta*zeta)*dxs

    return 1/(4*pi*pi)*k*k*( Ds/D - Zs/nuu )


# 2nd derivatives of polymer free energy -------------------------------------

def ddfrpa(HP,phi,phis,u, x=None, dx_all=None, ddx_all=None):    
    if useleff: 
        x = x_eff(HP, phi, phis, u) if x==None else x
        dxp, dxs = dx_eff(HP, phi,phis,u,x) if dx_all==None else dx_all          
        ddxpp, ddxss, ddxps = ddx_eff(HP, phi,phis,u,x, (dxp,dxs)) \
                              if ddx_all==None else ddx_all
    else:
        x, dxp, dxs, ddxpp, ddxss, ddxps = 1,0,0,0,0,0

    ddfpp = 0 if phi==0  else sci.quad(ddfrpa_pp_int, 0, np.inf,\
                                       args=(HP,phi,phis,u,x,dxp,ddxpp), limit=intMax)[0] 
    ddfss = 0 if phis==0 else sci.quad(ddfrpa_ss_int, 0, np.inf,\
                                       args=(HP,phi,phis,u,x,dxs,ddxss), limit=intMax)[0] 
    ddfps = 0 if phi*phis==0 else \
            sci.quad(ddfrpa_ps_int, 0, np.inf,\
                     args=(HP,phi,phis,u,x,dxp, dxs,ddxps), limit=intMax)[0] 
 
    return ddfpp, ddfss, ddfps


def ddfrpa_pp_int(k,HP,phi,phis,u,x,dxp,ddxpp):
    g, xi, zeta      = G11(HP, k, x), Gcc(HP, k, x), G1c(HP, k, x) 
    dg, dxi, dzeta   = dG11(HP, k, x), dGcc(HP, k, x), dG1c(HP, k, x) 
    ddg, ddxi, ddzeta= ddG11(HP, k, x), ddGcc(HP, k, x), ddG1c(HP, k, x) 
    nuu, Zp = nu(HP,k,phi,phis,u), HP['zc']*HP['pc']
    v = HP['w2']*np.exp(-HP['wd']*k*k)

    D   = vDel(HP,k,phi,phis,u,x, for_f=True)
    Dp  = Zp + (g*nuu*v+xi) + phi*(dxp*(dg*nuu*v+dxi)+g*Zp*v) \
            + 2*phi*v*(g*xi-zeta*zeta) \
            + v*phi*phi*(dg*xi + g*dxi - 2*dzeta*zeta)*dxp

    Dpp = 2*(v*Zp*g+(dg*nuu*v+dxi)*dxp) \
          + phi*( (ddg*nuu*v+ddxi)*dxp*dxp + 2*v*Zp*dg*dxp + (dg*nuu*v+dxi)*ddxpp) \
          + 2*v*(g*xi-zeta*zeta) + 4*v*phi*(g*dxi+dg*xi-2*zeta*dzeta)*dxp \
          + v*phi*phi*( g*ddxi+2*dg*dxi+ddg*xi-2*(dzeta*dzeta+zeta*ddzeta) )*dxp*dxp \
          + v*phi*phi*( g*dxi+dg*xi-2*zeta*dzeta )*ddxpp

    return k*k/(4*pi*pi)*( (Dpp*D-Dp*Dp)/D/D + Zp*Zp/nuu/nuu)

def ddfrpa_ss_int(k,HP,phi,phis,u,x,dxs,ddxss):
    g, xi, zeta      = G11(HP, k, x), Gcc(HP, k, x), G1c(HP, k, x) 
    dg, dxi, dzeta   = dG11(HP, k, x), dGcc(HP, k, x), dG1c(HP, k, x) 
    ddg, ddxi, ddzeta= ddG11(HP, k, x), ddGcc(HP, k, x), ddG1c(HP, k, x) 
    nuu, Zs = nu(HP,k,phi,phis,u), HP['zs']*(HP['zs']+HP['zc'])
    v = HP['w2']*np.exp(-HP['wd']*k*k)
    
    D   = vDel(HP,k,phi,phis,u,x, for_f=True)
    Ds  = Zs + phi*(g*Zs*v+(dg*nuu*v+dxi)*dxs) \
            + v*phi*phi*(dg*xi + g*dxi - 2*dzeta*zeta)*dxs
    Dss = phi*( (ddg*nuu*v+ddxi)*dxs*dxs + 2*v*Zs*dg*dxs + (dg*nuu*v+dxi)*ddxss ) \
          + v*phi*phi*( g*ddxi+2*dg*dxi+ddg*xi-2*(dzeta*dzeta+zeta*ddzeta) )*dxs*dxs \
          + v*phi*phi*( g*dxi+dg*xi-2*zeta*dzeta )*ddxss

    return k*k/(4*pi*pi)*( (Dss*D-Ds*Ds)/D/D + Zs*Zs/nuu/nuu)

def ddfrpa_ps_int(k,HP,phi,phis,u,x,dxp,dxs,ddxps):
    g, xi, zeta      = G11(HP, k, x), Gcc(HP, k, x), G1c(HP, k, x) 
    dg, dxi, dzeta   = dG11(HP, k, x), dGcc(HP, k, x), dG1c(HP, k, x) 
    ddg, ddxi, ddzeta= ddG11(HP, k, x), ddGcc(HP, k, x), ddG1c(HP, k, x) 
    nuu, Zp, Zs = nu(HP,k,phi,phis,u), HP['zc']*HP['pc'], HP['zs']*(HP['zs']+HP['zc'])
    v = HP['w2']*np.exp(-HP['wd']*k*k)

    D   = vDel(HP,k,phi,phis,u,x, for_f=True)
    Dp  = Zp + (g*nuu*v+xi) + phi*(dxp*(dg*nuu*v+dxi)+g*Zp*v) \
          + 2*phi*v*(g*xi-zeta*zeta) \
          + v*phi*phi*(dg*xi + g*dxi - 2*dzeta*zeta)*dxp
    Ds  = Zs + phi*(g*Zs*v+(dg*nuu*v+dxi)*dxs) \
            + v*phi*phi*(dg*xi + g*dxi - 2*dzeta*zeta)*dxs
    Dps = g*Zs*v + (dg*nuu*v+dxi)*dxs \
          + phi*( v*dg*(Zs*dxp+Zp*dxs) + (ddg*nuu*v+ddxi)*dxp*dxs \
                    + (dg*nuu*v+dxi)*ddxps ) \
          + 2*v*phi*(dg*xi+g*dxi-2*zeta*dzeta)*dxs \
          + v*phi*phi*( g*ddxi+2*dg*dxi+ddg*xi-2*(dzeta*dzeta+zeta*ddzeta) )*dxp*dxs \
          + v*phi*phi*( g*dxi+dg*xi-2*zeta*dzeta )*ddxps
    
    return k*k/(4*pi*pi)*( (Dps*D-Dp*Ds)/D/D + Zp*Zs/nuu/nuu )



#------------------------- System overall free energy -------------------------

# System free energy
def f_eng(HP, phi, phis, u, x=None):      
    if phi > 1 or phi < 0 or phis > 1 or phis < 0:
        return np.nan

    return enp(HP,phi,phis) + fscr(HP,phi,phis,u) + frpa(HP,phi,phis,u,x) \
           + HP['FH'](phi,u)

# 1st derivatives of system free energy
def df_eng(HP, phi, phis, u, x=None, dx=None):
    if phi > 1 or phi < 0 or phis > 1 or phis < 0:
        return np.nan
    
    dfsc_p, dfsc_s = dfscr(HP,phi,phis,u) 
    df2l_p, df2l_s = dfrpa(HP,phi,phis,u,x,dx)

    dfp = denp_p(HP,phi,phis) + dfsc_p + df2l_p + HP['dFH'](phi,u)
    dfs = denp_s(HP,phi,phis) + dfsc_s + df2l_s 

    return dfp, dfs
 
# 2nd derivatives of free energy
def ddf_eng(HP, phi, phis, u, x=None, dx=None, ddx=None):
    ddfsc_pp, ddfsc_ss, ddfsc_ps = ddfscr(HP,phi,phis,u)
    ddf2l_pp, ddf2l_ss, ddf2l_ps = ddfrpa(HP,phi,phis,u,x,dx,ddx)

    ddfpp = ddenp_pp(HP,phi,phis) + ddfsc_pp + ddf2l_pp + HP['ddFH'](phi,u)
    ddfss = ddenp_ss(HP,phi,phis) + ddfsc_ss + ddf2l_ss
    ddfps = ddenp_ps(HP,phi,phis) + ddfsc_ps + ddf2l_ps
 
    return ddfpp, ddfss, ddfps, ddfpp*ddfss-ddfps*ddfps




