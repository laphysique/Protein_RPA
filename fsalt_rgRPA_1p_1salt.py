# Apr 27, 2020
# - Upload to GitHub
# - thermal_funcs.py is renamed as thermal_rgRPA_1p_1salt.py

# Dec 25, 2018
# - Solve the phase diagram of the 3-component system
#   of protein + salt + water 
# - Use the beyond/modified-RPA theory
# - Different salt setting

import numpy as np
import scipy.optimize as sco
import thermal_rgRPA_1p_1salt as tt

invT = 1e4

# All functions here relies on the HeteroPolymer dict in thermal_func.py
# Genetate HP = tt.HeteroPolymer(sigma) before using the functions 

phi_min_calc = 1e-12

#-------------------- Solve protein-salt coexistence boundary --------------------

# total free energy
def f_total_v( phiPS_a_v, HP, u, phiPS_ori, f0 ): 
    phiori  = phiPS_ori[0] 
    phisori = phiPS_ori[1]  

    phia  = phiPS_a_v[0]
    phisa = phiPS_a_v[1]
    v     = phiPS_a_v[2]
    phib  = (phiori-v*phia)/(1-v)
    phisb = (phisori-v*phisa)/(1-v)
 
    fa = tt.f_eng(HP, phia, phisa, u)
    fb = tt.f_eng(HP, phib, phisb, u)
    
    #print(phia ,phisa, phib,phisb, v)
    return invT*( v*fa + (1-v)*fb - f0) 

#Jacobian of total free energy
def J_f_total_v( phiPS_a_v, HP, u, phiPS_ori, f0=0 ):
    phiori  = phiPS_ori[0] 
    phisori = phiPS_ori[1]  

    phia  = phiPS_a_v[0]
    phisa = phiPS_a_v[1]
    v     = phiPS_a_v[2]
    phib  = (phiori-v*phia)/(1-v)
    phisb = (phisori-v*phisa)/(1-v)
    #print(phia ,phisa, phib,phisb, v)   


    xeffa  = tt.x_eff(HP, phia, phisa, u)
    fa = tt.f_eng(HP, phia, phisa, u, x=xeffa)
    dfa, dfsa = tt.df_eng(HP, phia, phisa, u, x=xeffa)

    xeffb  = tt.x_eff(HP, phib, phisb, u)
    fb = tt.f_eng(HP, phib, phisb, u, x=xeffb)
    dfb, dfsb = tt.df_eng(HP, phib, phisb, u, x=xeffb)
    
    J = np.empty(3)
    J[0] = v*( dfa - dfb )  
    J[1] = v*( dfsa - dfsb )  
    J[2] = fa - fb + (phib-phia)*dfb + (phisb-phisa)*dfsb 
    
    return invT*J

# Constraint functions
# 0 < phia < 1
# 0 < phisa < 1
# 0 < phib < 1   : v < phiori/phia   && v < (1-phiori)/(1-phia)
# 0 < phisb < 1  : v < phisori/phisa && v < (1-phisori)/(1-phisa)
# phia+phisa < 1
# phib+phisb < 1 : v < (1-phiori-phisori)/(1-phia-phisa)
# 0 < v < 1
def vmin(phiPS_a_v, phiPS_ori):
    phiori  = phiPS_ori[0] 
    phisori = phiPS_ori[1]  

    phia  = phiPS_a_v[0]
    phisa = phiPS_a_v[1]
    #v     = phi_12_a_v[2]

    return min(1, phiori/phia, (1-phiori)/(1-phia), \
                  phisori/phisa, (1-phisori)/(1-phisa), \
                  (1-phiori-phisori)/(1-phia-phisa)     )   


def ps_bi_solve( HP, u, phiPS_ori ,r_vini, useJ ): 
    err = phi_min_calc

    phiori = phiPS_ori[0] 
    phisori = phiPS_ori[1]  

    f0 = tt.f_eng(HP, phiori, phisori, u)

    phi_ini = [phiori*0.5, phisori*0.95 ]

    vini = [r_vini*vmin(phi_ini, phiPS_ori)]
    inis = phi_ini + vini

    #print(inis)
    cons_all = ( {'type':'ineq', 'fun': lambda x:   x[0]-err }, \
                 {'type':'ineq', 'fun': lambda x: 1-x[0]-err }, \
                 {'type':'ineq', 'fun': lambda x:   x[1]-err }, \
                 {'type':'ineq', 'fun': lambda x: 1-x[1]-err }, \
                 {'type':'ineq', 'fun': lambda x: 1-x[0]-x[1]-err }, \
                 {'type':'ineq', 'fun': lambda x:   x[2]-err }, \
                 {'type':'ineq', 'fun': lambda x: vmin(x,phiPS_ori)-x[2]-err } \
               )


    if useJ:
        result = sco.minimize( f_total_v, inis,\
                               args = (HP, u, phiPS_ori, f0), \
                               method = 'SLSQP', \
                               jac = J_f_total_v, \
                               constraints = cons_all, \
                               tol = err/100, \
                               options={'ftol': err, 'eps': err} )
    else:
        result = sco.minimize( f_total_v, inis,\
                               args = (HP, u, phiPS_ori, f0), \
                               method = 'COBYLA', 
                               constraints = cons_all, \
                               tol = err/100 )  

    #print(result.x)
    phia = result.x[0]
    phisa = result.x[1]
    v     = result.x[2]
    phib = (phiori-v*phia)/(1-v)
    phisb = (phisori-v*phisa)/(1-v)
    if phia > phib:
        t1, t2 = phib, phisb
        phib, phisb = phia, phisa
        phia, phisa = t1, t2
        v = 1-v
 
    return [phia, phisa, phib, phisb, v]


def bisolve( HP, u, phiPS_ori ):

    r_vini1 = 0.5
    phiall = ps_bi_solve( HP, u, phiPS_ori, r_vini1 , 1 )
    phi_test = np.array(phiall)
    try_max = 20
    try_i = 0
    while np.isnan(sum(phi_test) ) \
          and np.array(np.where(((0<phi_test) & (phi_test<1)))).size != 4 \
          and try_i <= try_max:
        phiall = ps_bi_solve( HP, u, phiPS_ori, np.random.rand() , 1)
        phi_test = np.array(phiall)
        try_i = try_i+1        
        #print(try_i)
    return phiall

#-------------------- Solve protein-salt spinodal boundary --------------------

# Critical point calculation
def cri_salt(HP, u, pl, pu, pmid=None, psl=None, psu=None, thesign=-1):
    dp = pu-pl 
    if pmid==None:
        pmid=(pl+pu)/2

    result = sco.brent(cri_phis_solve, args = (u, HP, psl,psu, thesign), \
                       brack= (pl, pmid, pu), \
                       full_output=1 )
    phi_top, phis_top = result[0], thesign*result[1]
    return phi_top, phis_top

def cri_phis_solve(phi, u, HP, psl,psu, thesign):
    phismax = (HP['zc']-(HP['zc']+HP['pc'])*phi )/(HP['zc']+HP['zs'])
    if psl==None:
       psl = 1e-6
    if psu==None:
        psu=phismax*0.9999

    return thesign*sco.brenth(ddf_phis, psl, psu, args=(phi, u, HP) )

def ddf_phis(phis, phi, u, HP):
    ddf = tt.ddf_eng(HP, phi, phis, u)[3]
    print(phi, phis, ddf, flush=True)
    return ddf 






