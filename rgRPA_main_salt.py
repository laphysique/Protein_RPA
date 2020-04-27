# Apr 27, 2020
# - Upload to GitHub
# - thermal_funcs.py is renamed as thermal_rgRPA_1p_1salt.py
# - fmin_solve is renamed as fmin_rgRPA_1p_1salt.py
# - fsalt_solve is renamed as fsalt_rgRPA_1p_1salt.py

# Dec14, 2018
# - Beyond-RPA: 2-field renormalization
#   following Fredrickson's field theory
# - Calculating 2D phase diagram for 
#   protein-salt solution


import sys
import time
import multiprocessing as mp
import numpy as np

import fmin_rgRPA_1p_1salt as fs
import fsalt_rgRPA_1p_1salt as fss
import thermal_rgRPA_1p_1salt as tt
import seq_list as sl


u = float(sys.argv[2])

# Select a sequence in seq_list.py
seq_name = sys.argv[1]
sig, N, the_seq = sl.get_the_charge(seq_name)

if len(sys.argv) > 3:
    HP = tt.Heteropolymer(sig, w2=float(sys.argv[3]))
else:
    HP = tt.Heteropolymer(sig)

print('Seq:' , seq_name, '=' , the_seq )
print('w2=', HP['w2'])

#--------------------- Calculate salt-free boundary ---------------------

phi_cri, u_cri =  fs.cri_calc( HP )
if u_cri > u:
    print('u is too small, no phase separation')
    print('u has to be greater than ' + str(round(u_cri,2)) + "!" )
    sys.exit()

sp1, sp2 = fs.ps_sp_solve(HP, u, phi_cri)
bi1, bi2 = fs.ps_bi_solve(HP, u, (sp1, sp2), phi_cri )

#------------------ Calculate salt-dependent top point ------------------

phiTop, phisTop = fss.cri_salt(HP, u, sp1, sp2)

#------------------ Set up initial states (phi, phis) -------------------
nt = 100
dp = phisTop/nt
ddp = dp/5
pclose= phisTop*0.8
phisall = np.append( np.arange(dp,pclose, dp), \
                     np.arange(pclose, phisTop, ddp) )

#-------------------- Parallel calculate (phi, phis) --------------------

def salt_parallel(phis):
    x = fss.bisolve( HP, u, [phiTop, phis] )
    return x[0], x[1], x[2], x[3], x[4]

pool = mp.Pool(processes=40)
phia, phisa, phib, phisb, v = zip(*pool.map(salt_parallel, phisall))

#-------------------------- Prepare for output --------------------------

output = np.zeros((phisall.shape[0],5))
output[0] = np.array(phia)
output[1] = np.array(phisa)
output[2] = np.array(phib)
output[3] = np.array(phisb)
output[4] = np.array(v)

calc_info = seq_name + '_N' + str(N) + '_TwoFields' + \
            'u' + str(round(u,2)) + '_phisTop' + str(round(phisTop,5)) + \
            '_' + str(phisall.shape[0]) + 'samples' + \
            '.txt'
output_file = '../results/saltdept_' + calc_info
print(output_file)

cri_info = "      [phia phisa]          [phib phisb]           v  \
           "\n------------------------------------------------------------"           

np.savetxt(output_file, output, fmt = '%.10e', header= cri_info )


   
