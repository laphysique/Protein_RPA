# Apr 27, 2020
# - Upload to GitHub
# - thermal_funcs.py is renamed as thermal_rgRPA_1p_1salt

# Dec 9, 2019
# - phi_min_calc is now controlled by thermal_funcs.py


# Feb 7, 2019
# - Salt-dept function is added by assuming 
#   salt concentration is a constant

# Dec12, 2018
# - Beyond-RPA: 2-field renormalization
#   following Fredrickson's field theory
# - Calculating critical point and phase boundary
#   in salt-free polymer solution


import numpy as np
import scipy.optimize as sco
import thermal_rgRPA_1p_1salt as tt

phi_min_calc=tt.phi_min_calc

cri_pre_calc = False

# All functions here relies on the HeteroPolymer dict in thermal_func.py
# Genetate HP = tt.HeteroPolymer(sigma) before using the functions 

#----------------------- Critical point calculation -----------------------
 
def cri_calc( HP, phis=0, ini1=1e-4, ini3=1e-1, ini2=2e-1 ):

    phi_max = (1-2*phis)/(1+HP['pc'])
    #ini1, ini3, ini2 = 1e-6, 1e-2, phi_max*2/3
    
    if cri_pre_calc:  
   
        u1 = cri_u_solve(ini1, HP)
        u2 = cri_u_solve(ini2, HP)
        u3 = cri_u_solve(ini3, HP)

        while min(u1,u2,u3) != u3:
            if u1 >= u2:
                ini3 = (ini2+ini3)/2
            else:
                ini3 = (ini1+ini3)/2
              
        u3   = cri_u_solve(ini3, 0, HP)

    result = sco.brent(cri_u_solve, args = (HP,phis), \
                       brack= (ini1, ini3, ini2), \
                       full_output=1 )
 
    phicr, ucr = result[0], result[1]

    return phicr, ucr


def cri_u_solve( phi, HP, phis):
    return sco.brenth(ddf_u, 0.0001, 1000, args=(phi, HP, phis) )

# Function handle for cri_u_solve
def ddf_u(u, phi, HP, phis):
    ddf = tt.ddf_eng(HP, phi, phis, u)[0]
    print(phi, u, ddf, flush=True)
    return ddf 

#--------------------- Solve salt-free spinodal points ---------------------
def ps_sp_solve( HP, phis, u, phi_ori ):       
    err = phi_min_calc
    phi_max = (1-2*phis)/(1+HP['pc'])-err
 
    phi1 = sco.brenth(ddf_phi, err, phi_ori, args=(u,HP, phis) )
    phi2 = sco.brenth(ddf_phi, phi_ori, phi_max, args=(u,HP, phis) )
    return phi1, phi2

# Function handle for ps_sp_solve
def ddf_phi(phi, u, HP, phis ) :
    return tt.ddf_eng(HP, phi, phis, u)[0]


#-------------------- Solve salt-free coexistence curve --------------------
def ps_bi_solve( HP, phis, u, phi_sps , phi_ori=None):
    err = phi_min_calc
    phi_max = (1-2*phis)/(1+HP['pc'])-err
    sps1 ,sps2 = phi_sps

    phi_all_ini = [ sps1*0.9, sps2*1.1] 
    if phi_ori==None:
        phi_ori = (sps1+sps2)/2

    f_ori = tt.f_eng(HP, phi_ori, 0, u)

    result = sco.minimize( Eng_all, phi_all_ini, \
                           args = (u, HP, phis, phi_ori, f_ori), \
                           method = 'L-BFGS-B', \
                           jac = J_Eng_all, \
                           bounds = ((err,sps1-err), (sps2+err,phi_max)), \
                           options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20} ) 
    bi1 = min(result.x)
    bi2 = max(result.x)
    return bi1, bi2
  

#-------------------- System energy for minimization --------------------
 
invT = 1e2

def Eng_all( phi_all, u, HP, phis, phi0, f0 ):

    phi1 = phi_all[0]
    phi2 = phi_all[1]
    v =( phi2 - phi0 )/( phi2 - phi1 )

    f1 = tt.f_eng(HP, phi1, phis, u)
    f2 = tt.f_eng(HP, phi2, phis, u)

    fall = invT*(v*f1 + (1-v)*f2 - f0 )
    #print(phi_all, fall)

    return fall

#Jacobian of Eng_all
def J_Eng_all( phi_all, u, HP, phis, phi0 , f0):
    phi1 = phi_all[0]
    phi2 = phi_all[1]
    v =( phi2 - phi0 )/( phi2 - phi1 )

    xeff1  = tt.x_eff(HP, phi1, phis, u) if tt.useleff else 1
    f1 = tt.f_eng(HP, phi1, phis, u, x=xeff1)
    df1 = tt.df_eng(HP, phi1, phis, u, x=xeff1)[0]

    xeff2  = tt.x_eff(HP, phi2, phis, u) if tt.useleff else 1
    f2 = tt.f_eng(HP, phi2, phis, u, x=xeff2)
    df2 = tt.df_eng(HP, phi2, phis, u, x=xeff2)[0]

    J = np.empty(2)

    J[0] = v*( (f1-f2)/(phi2-phi1) + df1 )
    J[1] = (1-v)*( (f1-f2)/(phi2-phi1) + df2 )

    #print(J)

    return invT*J







