# Apr 27, 2020
# - Upload to GitHub
# - thermal_funcs.py is renamed as thermal_rgRPA_1p_1salt.py
# - fmin_solve is renamed as fmin_rgRPA_1p_1salt.py
# - fsalt_solve is renamed as fsalt_rgRPA_1p_1salt.py


# Dec 4, 2019
# Short script for 2D diagram

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

# Select a sequence in seq_list.py
seqname = sys.argv[1]
sig, N, the_seq = sl.get_the_charge(seqname)

u = float(sys.argv[2])
zc = float(sys.argv[3])
zs = float(sys.argv[4])

phiTop, phisTop = float(sys.argv[5]), float(sys.argv[6])

if len(sys.argv) > 7:
    phiBot, phisBot = float(sys.argv[7]), float(sys.argv[8]) 
else:
    phiBot, phisBot = -1, -1

HP = tt.Heteropolymer(sig, zc=zc, zs=zs)

print('Seq:' , seqname, '=' , the_seq )

#------------------- Prepare phi origin ------------------------

nt = 100

if len(sys.argv) > 7:
    phiall  = np.linspace( phiBot + (phiTop-phiBot)*0.001, phiTop-(phiTop-phiBot)*0.001 , nt)
    phisall = np.linspace( phisBot + (phisTop-phisBot)*0.001, phisTop-(phisTop-phisBot)*0.001 , nt)
else:
    phiall  = np.linspace( 0.1, phiTop, nt)
    phisall = np.linspace(phisTop*0.01, phisTop*0.999, nt) 
        
phisall
pp = np.array([ [phiall[i], phisall[i]] for i in range(nt)])


#-------------------- Parallel calculate (phi, phis) --------------------

def salt_parallel(p):
    try:
        x = fss.bisolve( HP, u, p )
        print(p, x)       
        return p[0], p[1], x[0], x[1], x[2], x[3], x[4]
    except:
        for test in range(5):
            p[0] = p[0]*(0.95+0.1*np.random.rand())
            try:
                x = fss.bisolve( HP, u, p )
                print(p, x)       
                return p[0], p[1], x[0], x[1], x[2], x[3], x[4]
            except:
                pass
        return p[0], p[1], -1, -1, -1, -1, -1  

pool = mp.Pool(processes=80)
phi0, phis0, phia, phisa, phib, phisb, v = zip(*pool.map(salt_parallel, pp))

#-------------------------- Prepare for output --------------------------

output = np.zeros((pp.shape[0],7))

output[:,0] = np.array(phi0)
output[:,1] = np.array(phis0)
output[:,2] = np.array(phia)
output[:,3] = np.array(phisa)
output[:,4] = np.array(phib)
output[:,5] = np.array(phisb)
output[:,6] = np.array(v)

head = ' u=' + str(u) + ' , phiTop=' + str(phiTop) + ' , phisTop=' + str(phisTop) + \
                       ' , phiBot=' + str(phiBot) + ' , phisBot=' + str(phisBot) +  '\n' + \
       '  [phiori, phisori]  [phia, phisa]  [phib, phisb], v \n' + \
       '--------------------------------------------------------------------------------------------'
np.savetxt('saltdept_zc' + str(HP['zc']) + '_zs' +  str(HP['zs']) + '_' \
           + seqname + '_u' + str(u) + '_w2_4.189.txt',output,header=head)

   
