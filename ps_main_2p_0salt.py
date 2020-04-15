# RPA model (no FH) for 2 overall neutral charge sequences   
# No salt. No conterions.
# Constant permittivity 
# solute sizes are not tunable

# ver Git.1 Apr 14, 2020

# Upload to github
# Rewrite data structure: from class to dict
# Rewrtie the code for calculating S(k): from matrix product to linear summation

# ver 2 May 14, 2017

import sys
import time
import multiprocessing as mp
import numpy as np

import f_min_solve_2p_0salt as fs
import thermal_2p_0salt as tt
import seq_list as sl

# Command: python3 seq_name1 seq_name2 u

# Select two sequences from seq_list.py
seq_name1 = sys.argv[1]
seq_name2 = sys.argv[2]
the_seq1 = getattr(sl.polymers, seq_name1)
the_seq2 = getattr(sl.polymers, seq_name2)


# Function for generating charge pattern
def get_charge(the_seq):
    N = len(the_seq)
    sigmai = np.zeros(N)

    for i in range(0, N):
        if the_seq[i] in 'DE' :
            sigmai[i] = -1
        elif the_seq[i] in 'RK' :
            sigmai[i] = 1
        else:
            sigmai[i] = 0

    return sigmai, N

sig1, N1 = get_charge(the_seq1)
sig2, N2 = get_charge(the_seq2)

# Determine u
u = float(sys.argv[3])

# Determine initial concentrations
phi1min = 0.01
phi1max = 0.05
phi2min = 0.01
phi2max = 0.05
Nphi1 = 20
Nphi2 = 20

phi1ori = np.linspace(phi1min, phi1max, Nphi1)
phi2ori = np.linspace(phi2min, phi2max, Nphi2)
phioris = np.array([ [phi1ori[i], phi2ori[j]] \
                      for i in range(0,Nphi1)    \
                      for j in range(0,Nphi2) ]    )

# Function for parallel multiprocessing
HP = twoProteins(sig1, sig2):

def bi_parallel(phis_ori):
    ps = fs.bisolve( HP, u, phis_ori )

    print( phis_ori, ps, flush=True)
    
    return ps

pool = mp.Pool(processes=80);
bi_all = pool.map(bi_parallel, phioris ) 

bi_out = bi_all#np.array(bi_all)

print(bi_out)

calc_info = '_twoProteins_' + seq_name1 + '_' + seq_name2 + \
            '_u' + str(u) + \
            '_phi1_' + str(phi1min) + 'to' + str(phi1max)   + \
            '_phi2_' + str(phi2min) + 'to' + str(phi2max)   + \
            '.txt'
bi_file = '../results/bi' + calc_info;

print(bi_file)

np.savetxt(bi_file, bi_out, fmt = '%.8e', \
           header = '[phi1a, phi2a] [phi1b, phi2b]' )


   
