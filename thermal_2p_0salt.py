# RPA model (no FH) for 2 overall neutral charge sequences   
# No salt. No conterions.
# Constant permittivity 
# solute sizes are not tunable

# ver Git.1 Apr 14, 2020
# Upload to github
# Rewrite data structure: from class to dict
# Rewrtie the code for calculating S(k): from matrix product to linear summation

# ver 2 May 14, 2017


import numpy as np
import scipy
import time

import scipy.integrate as sci

from numpy import exp
from numpy import log

phi_min_sys = 1e-12
Gamma = 1    # Short-range cutoff factor, 1 for Olvera de la Cruz's electric potential
c_smear = 0  # Short-range Gaussian smearing, 0.5 for Wang/Fredrickson's smearing 

intlim = 200

# twoProteins: 
# Returns a dict HP, which includes basic information of the proteins and the model parameters 
def twoProteins(sigma1, sigma2):
    sig1, sig2 = np.array(sigma1), np.array(sigma2)
    N1,   N2   = sig1.shape[0], sig2.shape[0] 

    # linear summation for S1(k)
    mel1 = np.kron(sig1, sig1).reshape((N1, N1))
    Tel1 = np.array([ np.sum(mel1.diagonal(n) + mel1.diagonal(-n)) for n in range(N1)])
    Tel1[0] /= 2
    L1 = np.arange(N1)

    # linear summation for S2(k)
    mel2 = np.kron(sig2, sig2).reshape((N2, N2))
    Tel2 = np.array([ np.sum(mel2.diagonal(n) + mel2.diagonal(-n)) for n in range(N2)])
    Tel2[0] /= 2
    L2 = np.arange(N2)

    HP = { 'sig1': sig1,  \
           'sig2': sig2,  \
           'N1'  : N1,    \
           'N2'  : N2,    \
           'T1'  : Tel1,  \
           'T2'  : Tel2,  \
           'L1'  : L1,    \
           'L2'  : L2     \
         }

    return HP

#----------------------------------- Entropy -----------------------------------

def s_calc(x):
    return (x > phi_min_sys )*x*np.log(x+(x<phi_min_sys)) 

def Enp(HP, phi1, phi2):
    return s_calc(phi1)/HP['N1'] + s_calc(phi2)/HP['N2'] + s_calc(1-phi1-phi2)

def d_Enp_1(HP, phi1, phi2):
    return -1 + 1/HP['N1'] + log(phi1)/HP['N1'] - log(1-phi1-phi2) 

def d_Enp_2(HP, phi1, phi2):
    return -1 + 1/HP['N2'] + log(phi2)/HP['N2'] - log(1-phi1-phi2) 

def dd_Enp_11(HP, phi1, phi2):
    return 1/phi1/HP['N1'] + 1/(1-phi1-phi2) 

def dd_Enp_22(HP, phi1, phi2):
    return 1/phi2/HP['N2'] + 1/(1-phi1-phi2) 

def dd_Enp_12(HP, phi1, phi2):
    return 1/(1-phi1-phi2) 

#------------------------------ RPA f_el function ------------------------------
def Uel(k,u):
    return 4*np.pi*u/(k*k*(1+Gamma*k*k))*np.exp(-c_smear*k*k)   

def Sk(Hp,k):
    return np.mean( HP['T1']*np.exp(-k*k*HP['L1']/6) ), \
           np.mean( HP['T2']*np.exp(-k*k*HP['L2']/6) )

def Fel(HP, phi1, phi2, u):
    f1 = sci.quad(Fel_toint1, 0, np.inf, args=(HP,phi,phis,u), limit=intlim)[0]  
    f2 = sci.quad(Fel_toint2, 0, np.inf, args=(HP,phi,phis,u), limit=intlim)[0]  


        return quad(self._Fel_to_int, 0, np.inf, \
                    args=(self, phi1, phi2, u), \
                    epsabs=err_abs, epsrel=err_rel,limit = N_int_max)[0]

# f_el
def Fel_toint1(k, HP, phi1, phi2, u):
    sk1, sk2 = Sk(HP, k)
    uk       = Uel(k,u) 
    G = lk*( phi1*sk1 + phi2*sk2 )

    return  1/(4*np.pi*np.pi)*k*k*( log(1 + G) - G + G*G/2  )

 def Fel_toint2(k, HP, phi1,phi2, u): 
    sk1, sk2 = Sk(HP, k)
    uk       = Uel(k,u) 
    G = lk*( phi1*sk1 + phi2*sk2 )
    
    return 1/(4*np.pi*np.pi)*k*k*(  G - G*G/2  ) 

# 1st-phi1 derivative of f_el
def dFel(HP, phi1, phi2, u ):
    df1 = sci.quad(dFel_1_toint, 0, np.inf, args=(HP,phi,phis,u), limit=intlim )[0]
    df2 = sci.quad(dFel_2_toint, 0, np.inf, args=(HP,phi,phis,u), limit=intlim )[0]
    return df1, df2 

def dFel_1_toint(k, HP, phi1, phi2, u):
    sk1, sk2 = Sk(HP, k)
    uk       = Uel(k,u) 
    G = lk*( phi1*sk1 + phi2*sk2 )
  
    return 1/(4*np.pi*np.pi)*k*k*lk*sk1/( 1 + G )

def dFel_2_toint(k, HP, phi1, phi2, u):
    sk1, sk2 = Sk(HP, k)
    uk       = Uel(k,u) 
    G = lk*( phi1*sk1 + phi2*sk2 )
  
    return 1/(4*np.pi*np.pi)*k*k*lk*sk2/( 1 + G )


# 2nd-phi1^2 derivative of f_el
def ddFel(HP, phi1, phi2, u ):
    ddf11 = sci.quad(ddFel_11_toint, 0, np.inf, args=(HP,phi,phis,u),limit=intlim )[0]
    ddf22 = sci.quad(ddFel_22_toint, 0, np.inf, args=(HP,phi,phis,u),limit=intlim )[0]
    ddf12 = sci.quad(ddFel_12_toint, 0, np.inf, args=(HP,phi,phis,u),limit=intlim )[0]
    return ddf11, ddf22, ddf12  

def ddFel_11_toint(k, HP, phi1, phi2, u):
    sk1, sk2 = Sk(HP, k)
    uk       = Uel(k,u) 
    G = lk*( phi1*sk1 + phi2*sk2 )

    A = lk*sk1/(1+G)

    return  -1/(4*np.pi*np.pi)*k*k*A*A


def ddFel_11_toint(k, HP, phi1, phi2, u):
    sk1, sk2 = Sk(HP, k)
    uk       = Uel(k,u) 
    G = lk*( phi1*sk1 + phi2*sk2 )

    B = lk*sk2/(1+G)

    return  -1/(4*np.pi*np.pi)*k*k*B*B


def ddFel_12_toint(k, HP, phi1, phi2, u):
    sk1, sk2 = Sk(HP, k)
    uk       = Uel(k,u) 
    G = lk*( phi1*sk1 + phi2*sk2 )

    A, B = lk*sk1/(1+G) , lk*sk2/(1+G)

    return  -1/(4*np.pi*np.pi)*k*k*A*B


#---------------------------- free energy functions ----------------------------

def feng(HP, phi1, phi2, u):      
    return Enp(HP, phi1, phi2) + Fel(HP,phi1,phi2,u)

# 1st derivatives 
def dfeng(HP, phi1, phi2, u):
    dFel_1, dFel_2 = dFel(HP, phi1,phi2,u)  
    df1 = d_Enp_1(HP, phi1, phi2) + dFel_1
    df2 = d_Enp_2(HP, phi1, phi2) + dFel_2 

    return df1, df2

# 2nd derivatives
def ddfeng(P, phi1, phi2, u):
    ddFel_11, ddFel_22, dFel_12 = ddFel(HP, phi1,phi2,u)
    ddf11 = dd_Enp_11(HP, phi1, phi2) + ddFel_11
    ddf22 = dd_Enp_22(HP, phi1, phi2) + ddFel_22
    ddf12 = dd_Enp_12(HP, phi1, phi2) + ddFel_12

    return ddf11, ddf22, ddf12, ddf11*ddf22-ddf12*ddf12
    

    return s_calc(phi1)/Ns[0] + s_calc(phi2)/Ns[1] + s_calc(1-phi1-phi2)


