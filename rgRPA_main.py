# Apr 27, 2020
# - Upload to GitHub
# - thermal_funcs.py is renamed as thermal_rgRPA_1p_1salt.py
# - fmin_solve is renamed as fmin_rgRPA_1p_1salt.py

# Dec 9, 2019
# - Add SQL output function

# Feb 7, 2019
# - Salt-dept function is added by assuming 
#   salt concentration is a constant

# Dec12, 2018
# - Beyond-RPA: 2-field renormalization
#   following Fredrickson's field theory
# - Calculating critical point and phase boundary
#   in salt-free polymer solution

import sys
import time
import multiprocessing as mp
import numpy as np
import sqlite3 as sql

import fmin_rgRPA_1p_1salt as fs
import thermal_rgRPA_1p_1salt as tt
import seq_list as sl


def sql_table_cond(value):
    vv = [ a for a in value]
    for i, v in enumerate(vv):
        if v == '.' :
            vv[i] = 'p'
        elif v == '-':
            vv[i] = 'n'
        else:
            pass
    return ''.join(vv)

# small ion valencies
zs, zc = 1, 1

usesql = True
sqlname = 'rgRPA_naiveFH_results.db'
sql_salt_const_name = 'salt_const_paras'


# fgRPA = True if calculating fg-RPA
fgRPA = False

if sys.argv[2] == "cri_calc":
   cri_calc_end = 1
else:
   cri_calc_end = 0
   umax = float(sys.argv[2])

duD = int(2)
du = 10**(-duD)

# Select a sequence in seq_list.py
seqname = sys.argv[1];#'Ddx4_N1'


# reduced ealt concentration
#if len(sys.argv) > 3:
try:
    phis = float(sys.argv[3])
#else:
except:
    phis = 0

# eh, es: now it is a must before setup pH and w2r
try: 
    tt.eh = float(sys.argv[4])
    tt.es = float(sys.argv[5])    
except:
    tt.eh = 0
    tt.es = 0

# pH value
pHchange = False
pHvalue = 0
try:
    pHvalue = float(sys.argv[6])
    sig, N, the_seq = sl.get_the_charge(seqname, pH=pHvalue )
    pHchange = True
except:
    sig, N, the_seq = sl.get_the_charge(seqname)   

# short-range repulsion
try:
    wtwo_ratio = float(sys.argv[7])
    HP = tt.Heteropolymer(sig, zc=zc,zs=zs, w2=4*np.pi/3*wtwo_ratio)
except:
    wtwo_ratio = 1
    HP = tt.Heteropolymer(sig, zc=zc,zs=zs)


# doing fg-RPA
if fgRPA:
    tt.useleff=False
    HP = tt.Heteropolymer(sig,w2=0)


#----------------------- Calculate critical point -----------------------
 

print('Seq:' , seqname, '=' , the_seq )

print('w2=', HP['w2'])
print('phis=', phis)
print('eh=' + str(tt.eh) + ' , es=' + str(tt.es) )

t0 = time.time()
phi_cri, u_cri = fs.cri_calc( HP, phis )

print('Critical point found in', time.time() - t0 , 's')

print( 'u_cri =', '{:.8e}'.format(u_cri) , \
       ', phi_cri =','{:.8e}'.format(phi_cri) )

if(cri_calc_end):
    sys.exit();

#---------------------------- Set up u range ----------------------------
ddu = du/10
umin = (np.floor(u_cri/ddu)+1)*ddu
uclose = (np.floor(u_cri/du)+2)*du
if umax < u_cri:
    umax = np.floor(u_cri*1.5) 
if uclose > umax:
    uclose = umax
print(umax)
uall = np.append( np.arange(umin, uclose, ddu ), \
                  np.arange(uclose, umax+du, du ) )

#-------------------- Parallel calculate multiple u's -------------------
 
def bisp_parallel(u):
    sp1, sp2 = fs.ps_sp_solve(HP, phis, u, phi_cri)
    print( u, sp1, sp2, 'sp done!', flush=True)
    bi1, bi2 = fs.ps_bi_solve(HP, phis, u, (sp1, sp2), phi_cri )
    print( u, bi1, bi2, 'bi done!', flush=True)
    
    return sp1, sp2, bi1, bi2

pool = mp.Pool(processes=80)
sp1, sp2, bi1, bi2 = zip(*pool.map(bisp_parallel, uall))


#---------------------- Prepare for output ----------------------

ind = np.where(np.array(bi1) > fs.phi_min_calc)[0]
nout = ind.shape[0]

sp1t = np.array([ sp1[i] for i in ind])
sp2t = np.array([ sp2[i] for i in ind])
bi1t = np.array([ bi1[i] for i in ind])
bi2t = np.array([ bi2[i] for i in ind])
ut = np.array([ round(uu, duD+1) for uu in uall[ind]])
new_umax = np.max(ut)

# Store in the SQL file
if usesql:
    output = np.zeros((2*nout+1,3))
    output[:,0] = np.concatenate((sp1t[::-1], [phi_cri], sp2t))
    output[:,1] = np.concatenate((bi1t[::-1], [phi_cri], bi2t))
    output[:,2] = np.concatenate((ut[::-1], [u_cri], ut))


    conn = sql.connect(sqlname)
    c = conn.cursor()
    
    try:
        c.execute( 'CREATE TABLE ' + sql_salt_const_name + \
                   ' (seq TEXT, pH REAL, zc REAL, zs REAL, phis REAL,' + \
                   ' w2r REAL, eh REAL, es REAL,' + \
                   ' phi_cri REAL, u_cri REAL,' + \
                   ' umax REAL, du REAL, ddu REAL)')
    except:
        pass
    
    pH_val = ( '%g' % pHvalue) if pHchange else 'NULL'
    zs_val = ('%g' % zs) if phis != 0 else 'NULL'      

    c.execute('INSERT INTO ' +  sql_salt_const_name + \
                       " VALUES ('" + seqname + "'," \
                         + pH_val + ','              \
                         + ('%g' % zc) + ','         \
                         + zs_val + ','              \
                         + str(phis) + ','       \
                         + ('%g' % wtwo_ratio) + ',' \
                         + ('%g' % tt.eh) + ','         \
                         + ('%g' % tt.es) + ','         \
                         + ('%.10e' % phi_cri) + ','  \
                         + ('%.10e' % u_cri) + ','    \
                         + ('%g' % new_umax) + ','       \
                         + ('%g' % du) + ','         \
                         + ('%g' % ddu) + ')') 
    
    #pH_cond = '_pH' + ''.join([ 'p' if t=='.' else t for t in pH_val]) \
    pH_cond = '_pH' + sql_table_cond(pH_val) if pH_val is not 'NULL' else ''
    zc_cond = '_zc' + ''.join([ 'p' if t=='.' else t for t in ('%g' % zc)])
    zs_cond = '_zs' + ''.join([ 'p' if t=='.' else t for t in zs_val]) \
              if zs_val is not 'NULL' else ''
    phis_cond = '_phis' + ''.join([ 'p' if t=='.' else t for t in str(phis)])
    w2r_cond  = '_w2r'  + ''.join([ 'p' if t=='.' else t for t in ( '%g' % wtwo_ratio)])
    #ehs_cond  = '_eh' + ''.join(['p' if t == '.' elif 'n' if t =='-' else t for t in ('%g' % tt.eh)]) + \
    #            '_es' + ''.join(['p' if t == '.' elif 'n' if t == '-' else t for t in ('%g' % tt.es)])
    ehs_cond = '_eh' + sql_table_cond(('%g' % tt.eh)) + '_es' + sql_table_cond(('%g' % tt.es))


    tname = seqname + pH_cond + zc_cond + zs_cond + phis_cond + w2r_cond + ehs_cond
    
    c.execute('CREATE TABLE ' + tname + '(phi_sp REAL, phi_bi REAL, u REAL)')
    c.executemany('INSERT INTO '  + tname + ' VALUES (?,?,?) ', \
                  map(tuple, output.tolist()) )
    conn.commit()

# Otherwise output txt file
else:
    sp_out = np.zeros((2*nout+1,2))
    bi_out = np.zeros((2*nout+1,2))

    sp_out[:,1] = np.concatenate((ut[::-1], [u_cri], ut))
    sp_out[:,0] = np.concatenate((sp1t[::-1], [phi_cri], sp2t))
    bi_out[:,1] = sp_out[:,1]
    bi_out[:,0] = np.concatenate((bi1t[::-1], [phi_cri], bi2t))

    print(sp_out)
    print(bi_out)

    calc_info = seqname + '_N' + str(N) + '_TwoFields' + \
                '_phis' + str(phis) + \
                '_w2r' + str(wtwo_ratio) + \
                '_eh' + str(tt.eh) + '_es' + str(tt.es) + \
                '_umax' + str(round(new_umax, duD)) + \
                '_du' + str(du) + '_ddu' + str(ddu) + \
                '.txt'

    sp_file = '../results/sp_' + calc_info
    bi_file = '../results/bi_' + calc_info

    print(sp_file)
    print(bi_file)

    cri_info = "u_cri= " + str(u_cri) + " , phi_cri= " + str(phi_cri) + \
               "\n------------------------------------------------------------"           

    np.savetxt(sp_file, sp_out, fmt = '%.10e', header= cri_info )
    np.savetxt(bi_file, bi_out, fmt = '%.10e', header= cri_info )



