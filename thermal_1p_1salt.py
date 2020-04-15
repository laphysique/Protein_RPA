# RPA+FH model for single charge sequence
# Short-range interaction contributes to only the k=0 FH term
# The FH term ehs parameters follow the definition in the PRL paper

# ver Git.1 Apr 14, 2020
# Upload to github

# ver2e: Oct 8, 2019
# - Include linear phi-dependent permittivity
# - Can switch between with and without self energy (NoSelfEnergy = True as default)
# - c_smear = 0 as default for Gaussian smearing width
# - Gamma can be any value equal or greater than 0;
#   however, Gamma != 0 when (NoSelfEnergy=False and c_smear=0)

# ver2c: Sep 18, 2019:
# - Gaussian smearing + simple Coulomb (Gamma=0), no self energy term

# ver2: Aug 27, 2019:
# - Allow concentration-dependent permittivity
# - Rewrite in terms of dict instead of class; simplify the structure.

# ver1b: Aug 22, 2019:
# - Use the new salt and counterion definition as in rG-RPA

# ver1: May 20, 2019: 
# - Linear summation in calculating structure factor S(k)

# ver0: May 27, 2018

import numpy as np
import scipy.integrate as sci
import global_vars as gv

NoSelfEnergy = True # Ture if electrostatic self energy is subtracted
 
Gamma = 0    # Short-range cutoff factor
intlim= 200

# function for calculating entropy
def s_calc(x):
    return (x > gv.phi_min_sys )*x*np.log(x+(x<gv.phi_min_sys)) 

#========================= Thermodynamic functions =========================

# RPAFH returns a dict HP, which includes basic information of the protein
# and the model parameters
#
# epsfun=False   : constant permittivity
# epsfun=True    : linear phi-dependent permittivity, eps_a, eps_b are used 
# eps_modify_ehs : Ture if using eps_r=eps0 to rescale ehs 
#                  (assuming the input ehs is of eps_r=1) 
def RPAFH(sig, ehs=[0,0], epsfun=False, eps_modify_ehs=False, 
          eps_a = 18.931087269965023,
          eps_b = 84.51003476887941  ):
    #sequence parameters
    sig = np.array(sig) # charge pattern
    N   = sig.shape[0]  # sequence length
    pc  = np.abs(np.sum(sig))/N # prefactor for counterions
    Q   = np.sum(sig*sig)/N     # fraction of charged residues (sig=+/-1) 

    # linear summation for S(k)
    mel = np.kron(sig, sig).reshape((N, N))
    Tel = np.array([ np.sum(mel.diagonal(n) + mel.diagonal(-n)) for n in range(N)])
    Tel[0] /= 2
    L = np.arange(N)  

    HP = {  'sig': sig, \
            'N'  : N,   \
            'pc' : pc,  \
            'Q'  : Q,   \
            'L'  : L,   \
            'Tel': Tel, \
            'ehs': ehs  \
           }  
  
    if epsfun:
        a, b = eps_a, eps_b
        HP['eps0'] = b
        flinear = lambda x: a*x + b*(1-x)
        HP['epsx']   = lambda x: b/flinear(x)
        HP['depsx']  = lambda x: -b*(a-b)/(flinear(x))**2
        HP['ddepsx'] = lambda x: 2*b*(a-b)*(a-b)/(flinear(x))**3
    else:
        HP['eps0']   = 1
        HP['epsx']   = lambda x: 1*(x==x)
        HP['depsx']  = lambda x: 0*x
        HP['ddepsx'] = lambda x: 0*x   
    
    # using eps_r=eps0 to rescale ehs (assuming the input ehs is of eps_r=1) 
    if eps_modify_ehs:
        ehs[0] *= HP['eps0']
                   
    return HP

# entropy
def Enp(HP, phi, phis): 
    phic = HP['pc']*phi + phis
    return 1/HP['N']*s_calc(phi) + s_calc(phic) + s_calc(phis) \
           + s_calc(1-phi*gv.r_res-phic*gv.r_con-phis*gv.r_sal) 
                  
def dEnp(HP, phi, phis):
    phic = HP['pc']*phi + phis
    return ( 1 + np.log(phi) )/HP['N'] + HP['pc']*(1 + np.log(phic+(HP['pc']==0)) ) \
             - (gv.r_res + gv.r_con*HP['pc']) \
                *( 1 + np.log(1-phi*gv.r_res-phic*gv.r_con-phis*gv.r_sal) )

def ddEnp(HP, phi, phis):
    phic = HP['pc']*phi + phis
    return 1/HP['N']/phi + HP['pc']*HP['pc']/(phic + (HP['pc']==0))*(np.abs(HP['pc'])>0) \
           + (gv.r_res + gv.r_con*HP['pc'])*(gv.r_res + gv.r_con*HP['pc']) \
              /(1-phi*gv.r_res-phic*gv.r_con-phis*gv.r_sal)

# f_el functions
c_smear = 0

def Uel(k,u):
    return 4*np.pi*u/(k*k*(1+Gamma*k*k))*np.exp(-c_smear*k*k)

def Sk(HP,k):
    return np.mean( HP['Tel']*np.exp(-k*k*HP['L']/6) )

def fel(HP,phi,phis,u):
    f1 = sci.quad(fel_toint1, 0, np.inf, args=(HP,phi,phis,u), limit=intlim)[0]
    f2 = sci.quad(fel_toint2, 0, np.inf, args=(HP,phi,phis,u), limit=intlim)[0]
    return f1+f2

def fel_toint1(k, HP, phi, phis, u):
    sk = Sk(HP,k)
    lk  = Uel(k,u)
    epx = HP['epsx'](phi*gv.r_res)

    G1 = lk*epx*( 2*phis + phi*( HP['pc'] + sk )  )

    return 1/(4*np.pi*np.pi)*k*k*( 1/gv.eta*np.log(1+gv.eta*G1) - G1 + G1*G1/2 )

def fel_toint2(k, HP, phi, phis, u):
    sk = Sk(HP,k)
    lk  = Uel(k,u)
    epx = HP['epsx'](phi*gv.r_res)

    G1 = lk*epx*( 2*phis + phi*( HP['pc'] + sk )  )
    G2 = lk*epx*( 2*phis + phi*( HP['pc'] + HP['Q'] ) )

    return 1/(4*np.pi*np.pi)*k*k*( G1-G1*G1/2-NoSelfEnergy*G2)


def dfel(HP,phi,phis,u):
    return sci.quad(dfel_toint, 0, np.inf, args=(HP,phi,phis,u), limit=intlim )[0]

def dfel_toint(k, HP, phi, phis, u):
    sk   = Sk(HP,k)
    lk   = Uel(k,u)
    epx  = HP['epsx'](phi*gv.r_res)
    depx = gv.r_res*HP['depsx'](phi*gv.r_res)

    SSk = HP['pc'] + sk
    G1 = lk*epx*( 2*phis + phi*SSk  )
 
    A1 = ( depx*(2*phis + phi*SSk ) + epx*SSk )/(1+gv.eta*G1)
    A2 = depx*( 2*phis + phi*( HP['pc'] + HP['Q'] ) ) + epx*( HP['pc'] + HP['Q'] )   

    return 1/(4*np.pi*np.pi)*k*k*lk*( A1- NoSelfEnergy*A2)    


def ddfel(HP,phi,phis,u):
    return sci.quad(ddfel_toint, 0, np.inf, args=(HP,phi,phis,u),limit=intlim )[0]

def ddfel_toint(k, HP, phi, phis, u):
    sk   = Sk(HP,k)
    lk   = Uel(k,u)
    epx  = HP['epsx'](phi*gv.r_res)
    depx = gv.r_res*HP['depsx'](phi*gv.r_res)
    ddepx = gv.r_res*gv.r_res*HP['ddepsx'](phi*gv.r_res)

    SSk  = HP['pc'] + sk
    G1   = lk*epx*( 2*phis + phi*SSk  )
    dG1  = lk*( depx*(2*phis + phi*SSk ) + epx*SSk )
    ddG1 = lk*( ddepx*(2*phis+phi*SSk) + 2*depx*SSk) 

    A11 = dG1/(1+gv.eta*G1)
    A12 = ddG1/(1+gv.eta*G1)

    A2  = ddepx*( 2*phis + phi*( HP['pc'] + HP['Q'] ) ) \
           + 2*depx*( HP['pc'] + HP['Q'] )  

    return 1/(4*np.pi*np.pi)*k*k*( -gv.eta*A11*A11 + A12 - NoSelfEnergy*lk*A2 )  



#f_FH functions
def chi_calc(HP,u):
    ehs = HP['ehs']
    return gv.r_res*gv.r_res*( ehs[0]*u + ehs[1] )

    
# free energy functions
def feng(HP, phi, phis, u):
    return Enp(HP, phi, phis) + fel(HP, phi,phis,u) - chi_calc(HP, u)*phi*phi

def dfeng(HP, phi, phis, u):
    return dEnp(HP, phi, phis) + dfel(HP, phi,phis,u) - 2*chi_calc(HP,u)*phi

def ddfeng(HP, phi, phis, u):
    return ddEnp(HP, phi, phis) + ddfel(HP, phi,phis,u) - 2*chi_calc(HP, u)













