# This is a python script that will generate a LAMMPS molecule file for use in
# Polymer Brush
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy




def read_edp(filename,sheet):
    edp_df = pd.pandas.read_csv(filename, sheet_name=sheet, header = None)
    data = edp_df.values
    Ms = data[0,1:]
    zhis = data[1,1:]
    z = data[2:,0]
    edps = data[2:,1:]

    #zhi = edp_df
    return z,edps,zhis,Ms

def brush_height(z,dp,zhi):
    z0 = 1
    s = np.sum(dp)
    t0 = 1*s/100
    t1 = 90*s/100
    cs = np.cumsum(dp)
    id0 = np.where(cs < t0)
    id1 = np.where(cs < t1)
    i0 = id0[0]
    i1 = id1[0]
    z0 = z[i0[-1]]
    z1 = z[i1[-1]]
    h = (z1 - z0) * zhi

    return h

def heights(z, edps, zhis):
    m,n = edps.shape
    hs = np.zeros(n)
    for i in range(n):
        hs[i] = brush_height(z,edps[:,i],zhis[i])
    return hs
    
def fhh(sc,a,c):
    N = 60
    t = 1.0
    bo = 0.5799
    Rg = 4.6257
    L = (N-1)*bo
    return a * (((t * bo * sc * (Rg**2.0))/(L))**(1.0/3.0)) * L + c 
    
def fit_spline(X,Y):
    xnew = np.linspace(X[0], X[-1],80)
    spl = scipy.interpolate.make_interp_spline(X,Y,k=3)
    ynew = spl(xnew)
    return xnew, ynew


def dp_ids(dp):
    s = np.sum(dp)
    t0 = 0.001*s/100
    t1 = 99.9999*s/100
    cs = np.cumsum(dp)
    id0 = np.where(cs < t0)
    id1 = np.where(cs < t1)
    i0_del = id0[0]
    i1_del = id1[0]
    i0 = i0_del[-1]
    i1 = i1_del[-1]

    return i0,i1
       
def dp_z2d(zZ,dp_z,i0,i1):
    zs = (zZ[i0:i1] - zZ[i0])
    D = (zZ[i1]-zZ[i0])
    zD = zs/D
    dp_d = dp_z[i0:i1] 
    return zD,dp_d        

    
def read_dps(path,stage):
    ocwd = os.getcwd()
    os.chdir(path)
    if stage == 1:
        a = 'equil.csv'
        b = 'abdpe.csv'
        c = 'bbdpe.csv'
        d = 'tbdpe.csv'
    elif stage == 2:
        a = 'comp.csv'
        b = 'abdpc.csv'
        c = 'bbdpc.csv'
        d = 'tbdpc.csv'
    elif stage == 3:
        a = 'shear.csv'
        b = 'abdps.csv'
        c = 'bbdps.csv'
        d = 'tbdps.csv'
    zhi = (pd.pandas.read_csv(a).values)[-1,27]
    
    abeads = (pd.pandas.read_csv(b)).values
    zZ = abeads[1:,1]
    adp_z = abeads[1:,-1]
    i0,i1 = dp_ids(adp_z)
    zD,adp = dp_z2d(zZ,adp_z,i0,i1)
    zDadp = [zD, adp]
    
    bbeads = (pd.pandas.read_csv(c)).values
    zZ = bbeads[1:,1]
    bdp_z = bbeads[1:,-1]
    zD,bdp = dp_z2d(zZ,bdp_z,i0,i1)
    zDbdp = [zD, bdp]
    
    tbeads = (pd.pandas.read_csv(d)).values
    zZ = tbeads[1:,1]
    tdp_z = tbeads[1:,-1]
    zD,tdp = dp_z2d(zZ,tdp_z,i0,i1)
    zDtdp = [zD, tdp]
    
    #Calculate interpenetration
    I = np.trapz((bdp*tdp),zD)
    D = (zZ[i1]-zZ[i0])*zhi
    os.chdir(ocwd)
    return zDbdp,zDtdp,zDadp,zhi,I,D
    
def IvsD(path):
    ocwd = os.getcwd()
    os.chdir(path)
    dirs = filter(os.path.isdir ,os.listdir('.'))
    n = len(dirs)
    Ps = np.zeros(n)
    Is = np.zeros(n)
    Ds = np.zeros(n)
    i=0
    for dir in dirs:  
        S = dir.split("_")
        Ps[i] = float(S[0])
        c = '.\\' + dir
        x,y,z,zhi,Is[i],Ds[i] = read_dps(c,2)
        i = i + 1 
    Ps,Is,Ds = zip(*sorted(zip(Ps,Is,Ds)))
    os.chdir(ocwd)
    return Ps,Is,Ds

def read_thermo(path,cols,stage):
    ocwd = os.getcwd()
    os.chdir(path)
    dirs = list(filter(os.path.isdir ,os.listdir('.')))
    #print(dirs)
    n = len(dirs)
    m = len(cols)
    Vs = np.zeros(n)
    Cs = np.zeros((n,m))
    if stage == 1:
        f = "\equil.csv"
    elif stage == 2:
        f = "\comp.csv"
    elif stage == 3:
        f = "\shear.csv"
    i = 0    
    for dir in dirs:
        S = dir.split("_")
        Vs[i] = float(S[0])
        #print(Vs[i])
        a = dir + f
        b = '.\\' + a
        df = pd.pandas.read_csv(b)
        j = 0
        for col in cols:
            data = df.values[:,col+1]
            Cs[i,j] = np.mean(data[-200:])
         #   print(Cs[i,j])
            j = j + 1
        i = i + 1
    A = zip(Vs,Cs)    
    B = sorted(A)
    Vs,Cs = list(zip(*B))
    Cs = np.asarray(Cs)
    os.chdir(ocwd)
    return Vs,Cs
    
def thermo_time(path,cols,stage):
    ocwd = os.getcwd()
    os.chdir(path) 
    cols = list(np.asarray(cols) + 1)
    #print(cols)
    if stage == 1:
        f = "equil.csv"
    elif stage == 2:
        f = "comp.csv"
    elif stage == 3:
        f = "shear.csv"
    df = pd.pandas.read_csv(f)
    dt = df.iloc[:,1]
    ts = dt.values
    vdf = df.iloc[:,cols]
    Vt = vdf.values
    os.chdir(ocwd)
    return ts,Vt

def read_tvdp(path):
    ocwd = os.getcwd()
    os.chdir(path)
    zhi = (pd.pandas.read_csv('shear.csv').values)[-1,27]
    
    abeads = (pd.pandas.read_csv('abdps.csv')).values
    zZ = abeads[1:,1]
    adpZ = abeads[1:,-1]
    i0,i1 = dp_ids(adpZ)
    zD,adp = dp_z2d(zZ,adpZ,i0,i1)
    zDDp = [zD, adp]
    
    temps = (pd.pandas.read_csv('temps.csv')).values
    zZ = temps[1:,1]
    Tdata = temps[1:,-1]
    zD,Tp = dp_z2d(zZ,Tdata,i0,i1)
    zDTp = [zD, Tp]
    
    velps = (pd.pandas.read_csv('velps.csv')).values
    zZ = velps[1:,1]
    Vdata = velps[1:,-1]
    zD,Vp = dp_z2d(zZ,Vdata,i0,i1)
    zDVp = [zD, Vp]

    
    os.chdir(ocwd)
    return zDDp,zDTp,zDVp
    
def read_mops(path):
    ocwd = os.getcwd()
    os.chdir(path)
    
    abeads = (pd.pandas.read_csv('abdps.csv')).values
    zZ = abeads[1:,1]
    adpZ = abeads[1:,-1]
    i0,i1 = dp_ids(adpZ)
    zD,adp = dp_z2d(zZ,adpZ,i0,i1)
    zDDp = [zD, adp]
    
    mops = (pd.pandas.read_csv('mops.csv')).values
    zZ = mops[1:,1]
    Pdata = mops[1:,2:]
    zD,Pp = dp_z2d(zZ,Pdata,i0+1,i1+1)
    zDPp = [zD, Pp]
    
    
    os.chdir(ocwd)
    return zDDp,zDPp



if __name__ == '__main__':
    
#  0   1  2   3   4    5       6       7           8       9       10   11  12  13  14  15  16   17     18      19         20     21         22      23         24         25          26       27       28      29       30    31    32    33      34         35             36      37           38         39        40          41           
# step et ke pe epair temp c_melTemp c_wallTemp v_Fcatom v_Pcomp2 press pxx pyy pzz pxy pxz pyz c_Vir c_Vir[1] c_Vir[2] c_Vir[3] c_Vir[4] c_Vir[5] c_Vir[6] c_melPress c_wallPress v_melDens v_surfcov v_aveRg v_Vwall v_srate v_D v_bwzmax zhi c_fbwall[1] c_fbwall[3] c_ftwall[1] c_ftwall[3] c_ggbot[1] c_ggbot[3] c_ggtop[1] c_ggtop[3]
    
#   
#  
    print('Analysis')

    
#     #---Equilibrium---#
#     
#     
#     # Monomer Density profiles 
#     
#     # P=1
#     
#     zDb05,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.05\MDPBB-FS-0.5\1_P',1)
#     zDb1,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.05\MDPBB-FS-1\1_P',1)
#     zDb2,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.05\MDPBB-FS-2\1_P',1)
#     zDb4,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.05\MDPBB-FS-4\1_P',1)
#     zDb8,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.05\MDPBB-FS-8\1_P',1)
#     zDb16,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.05\MDPBB-FS-16\1_P',1)
#     
#     zzDb05,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-0.5\1_P',1)
#     zzDb1,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-1\1_P',1)
#     zzDb2,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-2\1_P',1)
#     zzDb4,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-4\1_P',1)
#     zzDb8,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-8\1_P',1)
#     zzDb16,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-16\1_P',1)
#     
#     zzzDb05,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.15\MDPBB-FS-0.5\1_P',1)
#     zzzDb1,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.15\MDPBB-FS-1\1_P',1)
#     zzzDb2,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.15\MDPBB-FS-2\1_P',1)
#     zzzDb4,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.15\MDPBB-FS-4\1_P',1)
#     zzzDb8,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.15\MDPBB-FS-8\1_P',1)
#     #zzzDb16,zDt,zDa,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-16\1_P',1)
#     
#     print(zhi)
#     
#     plt.rcParams.update({'font.size': 15})
#     fig, ax = plt.subplots(figsize=(10,8))
#     l05, = ax.plot(zzDb05[0] , zzDb05[1], 'r--')
#     l1, = ax.plot(zzDb1[0] , zzDb1[1], 'g--')
#     l2, = ax.plot(zzDb2[0] , zzDb2[1], 'b--')
#     l4, = ax.plot(zzDb4[0] , zzDb4[1], 'r-')
#     l8, = ax.plot(zzDb8[0] , zzDb8[1], 'g-')
#     l16, = ax.plot(zzDb16[0] , zzDb16[1], 'b-')
#     plt.legend([l05, l1, l2, l4, l8, l16], ['f=0.5','f=1','f=2', 'f=4', 'f=8', 'f=16'], loc=1)
#     ax.set(xlim=[0,0.25],xlabel='$z/D$', ylabel= r'$\phi(z)$' , title='')
#     ax.minorticks_on()
#     plt.savefig('1-EquilDP-GD-0.075-P-1.jpg')
#     
#     
#     
#     # Brush Height vs F
#     hs = np.zeros(6)
#     hss = np.zeros(6)
#     hsss = np.zeros(6)
#     
#     print(hs)
#     fs = [0.5,1,2,4,8,16]
#     
#     hs[0] = brush_height(zDb05[0],zDb05[1],zhi)
#     hs[1] = brush_height(zDb1[0],zDb1[1],zhi)
#     hs[2] = brush_height(zDb2[0],zDb2[1],zhi)
#     hs[3] = brush_height(zDb4[0],zDb4[1],zhi)
#     hs[4] = brush_height(zDb8[0],zDb8[1],zhi)
#     hs[5] = brush_height(zDb16[0],zDb16[1],zhi)
#     
#     hss[0] = brush_height(zzDb05[0],zzDb05[1],zhi)
#     hss[1] = brush_height(zzDb1[0],zzDb1[1],zhi)
#     hss[2] = brush_height(zzDb2[0],zzDb2[1],zhi)
#     hss[3] = brush_height(zzDb4[0],zzDb4[1],zhi)
#     hss[4] = brush_height(zzDb8[0],zzDb8[1],zhi)
#     hss[5] = brush_height(zzDb16[0],zzDb16[1],zhi)
#     
#     hsss[0] = brush_height(zzzDb05[0],zzzDb05[1],zhi)
#     hsss[1] = brush_height(zzzDb1[0],zzzDb1[1],zhi)
#     hsss[2] = brush_height(zzzDb2[0],zzzDb2[1],zhi)
#     hsss[3] = brush_height(zzzDb4[0],zzzDb4[1],zhi)
#     hsss[4] = brush_height(zzzDb8[0],zzzDb8[1],zhi)
#     
#     print(hs)
#     fig, ax = plt.subplots(figsize=(10,8))
#     l, = ax.plot(fs , hs, 'ro')
#     l2, = ax.plot(fs, hss, 'go')
#     l3, = ax.plot(fs, hsss, 'bo')
#     plt.legend([l, l2, l3], [r'$\rho_g = 0.05$',r'$\rho_g = 0.075$',r'$\rho_g = 0.15$'], loc=4)
#     ax.set(xlim=[0,18],ylim=[0,12] ,xlabel='$f$', ylabel= r'$h$' , title='')
#     ax.minorticks_on()
#     plt.savefig('2-h vs f.jpg')
#     
#     
#     
#     #--- Compression Curve ---#
#     
#     Ps05, Ds05 = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.15\MDPBB-FS-0.5',[24],2)
#     Ps1, Ds1 = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.15\MDPBB-FS-1',[24],2)
#     Ps2, Ds2 = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.15\MDPBB-FS-2',[24],2)
#     Ps4, Ds4 = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.15\MDPBB-FS-4',[24],2)
#     Ps8, Ds8 = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.15\MDPBB-FS-8',[24],2)
# #    Ps16, Ds16 = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-16',[24],2)
# 
#     
#     plt.rcParams.update({'font.size': 15})
#     fig, ax = plt.subplots(figsize=(10,8))
#     l05, = ax.plot(Ds05 , Ps05, 'r--')
#     l1, = ax.plot(Ds1 , Ps1, 'g--')
#     l2, = ax.plot(Ds2 , Ps2, 'b--')
#     l4, = ax.plot(Ds4 , Ps4, 'r-')
#     l8, = ax.plot(Ds8 , Ps8, 'g-')
# #    l16, = ax.plot(Ds16 , Ps16, 'b-')
#     plt.legend([l05, l1, l2, l4, l8], ['f=0.5','f=1','f=2', 'f=4', 'f=8'], loc=1)
#     ax.set(xlim=[0,30],xlabel='$D$', ylabel= r'$P_{comp}$' , title='')
#     ax.minorticks_on()
#     plt.savefig('3-CompCurve-GD-0.15.jpg')
#     
#     
#     #---Interpenetration Profiles---#
#     
#     b0751,t0751,a0751,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-1\1_P',2)
#     b07516,t07516,a07516,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-16\1_P',2)
#     
#     fig, ax = plt.subplots(figsize=(10,8))
#     lb1, = ax.plot(b0751[0] , b0751[1], 'r--')
#     lt1, = ax.plot(t0751[0] , t0751[1], 'b--')
#     la1, = ax.plot(a0751[0] , a0751[1], 'g--')
#     lb16, = ax.plot(b07516[0] , b07516[1], 'r-')
#     lt16, = ax.plot(t07516[0] , t07516[1], 'b-')
#     la16, = ax.plot(a07516[0] , a07516[1], 'g-')
#     plt.legend([la1, la16], ['f=1','f=16'], loc=1)
#     ax.set(xlim=[0,1],xlabel='$z/D$', ylabel= r'$\phi(z)$' , title='')
#     ax.minorticks_on()
#     plt.savefig('4-Interpenetration-GD-0.075-P-1.jpg')
#     
#     b0751,t0751,a0751,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-1\8_P',2)
#     b07516,t07516,a07516,zhi,I,D = read_dps(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-16\8_P',2)
#     
#     fig, ax = plt.subplots(figsize=(10,8))
#     lb1, = ax.plot(b0751[0] , b0751[1], 'r--')
#     lt1, = ax.plot(t0751[0] , t0751[1], 'b--')
#     la1, = ax.plot(a0751[0] , a0751[1], 'g--')
#     lb16, = ax.plot(b07516[0] , b07516[1], 'r-')
#     lt16, = ax.plot(t07516[0] , t07516[1], 'b-')
#     la16, = ax.plot(a07516[0] , a07516[1], 'g-')
#     plt.legend([la1, la16], ['f=1','f=16'], loc=1)
#     ax.set(xlim=[0,1],xlabel='$z/D$', ylabel= r'$\phi(z)$' , title='')
#     ax.minorticks_on()
#     plt.savefig('4-Interpenetration-GD-0.075-P-8.jpg')
#     
#     ocwd = os.getcwd()
#     print(ocwd)
#     
#     
#     #---Interpenetration vs D---#
#     Ps05,Is05,Ds05 = IvsD(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-0.5')
#     Ps1,Is1,Ds1 = IvsD(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-1')
#     Ps2,Is2,Ds2 = IvsD(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-2')
#     Ps4,Is4,Ds4 = IvsD(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-4')
#     Ps8,Is8,Ds8 = IvsD(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-8')
#     Ps16,Is16,Ds16 = IvsD(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-16')
# 
#     fig, ax = plt.subplots(figsize=(10,8))
#     l05, = ax.plot(Ds05,Is05, 'ro')
#     l1, = ax.plot(Ds1,Is1, 'go')
#     l2, = ax.plot(Ds2,Is2, 'bo')
#     l4, = ax.plot(Ds4,Is4, 'rs')
#     l8, = ax.plot(Ds8,Is8, 'gs')
#     l16, = ax.plot(Ds16, Is16, 'bs')   
#         
#     plt.legend([l05,l1,l2,l4,l8,l16],['$f=0.5$', '$f=1$','$f=2$','$f=4$','$f=8$','$f=16$'], loc=9)
#     ax.set(xlabel='$D$', ylabel= r'$I(D)$' , title='')
#     ax.minorticks_on()
#     plt.savefig('5-IvsD.jpg')
# 
#     
#     
#     #---T,V and density profiles---#
#       
#     fig, ax1 = plt.subplots(figsize=(10,8))
#     ax2 = ax1.twinx()
#     
#     D1,T1,V1 = read_tvdp(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-1\1_P')
#     Tp1, = ax1.plot(T1[0],T1[1], 'r--')
#     Dp1, = ax1.plot(D1[0],D1[1], 'g--')
#     Vp1, = ax2.plot(V1[0],V1[1], 'b--')
#     
#     
# 
#     D16,T16,V16 = read_tvdp(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-16\1_P')
#     Tp16, = ax1.plot(T16[0],T16[1], 'r-')
#     Dp16, = ax1.plot(D16[0],D16[1], 'g-')
#     Vp16, = ax2.plot(V16[0],V16[1], 'b-')
# 
# 
# 
#     plt.legend([Tp16,Vp16,Dp16], [r'Temperature','Velocity','Density'], loc=4)
#     ax1.set(xlim=[0,1],xlabel='$z/D$', ylabel= r'$T, \phi(z)$')
#     ax2.set_ylabel(r'$V$')
#     ax1.minorticks_on()
#     ax2.minorticks_on()
#     plt.savefig('6-TnVprofiles.jpg')    
#     ocwd = os.getcwd()
#     print(ocwd)
#     #--- Convergence check T---#
#     
#     ts,Vt = thermo_time(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-1\1_P',[5,24],3)
#     
#     ocwd = os.getcwd()
#     print(ocwd)
#     fig, ax1 = plt.subplots(figsize=(10,8))
#     l1, = ax1.plot(ts[1000:],Vt[1000:,0], 'ro')
#     ax2 = ax1.twinx()
#     l2, = ax2.plot(ts[1000:],Vt[1000:,1], 'bo')
#     ax1.minorticks_on()
#     ax2.minorticks_on()
#     plt.savefig('7-Time Evolution-f-1.jpg') 
# 
#     ts,Vt = thermo_time(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-16\1_P',[5,24],3)
#     
#     ocwd = os.getcwd()
#     print(ocwd)
#     fig, ax1 = plt.subplots(figsize=(10,8))
#     l1, = ax1.plot(ts[1000:],Vt[1000:,0], 'ro')
#     ax2 = ax1.twinx()
#     l2, = ax2.plot(ts[1000:],Vt[1000:,1], 'bo')
#     ax1.minorticks_on()
#     ax2.minorticks_on()
#     plt.savefig('7-Time Evolution-f-16.jpg') 
#     
#     #---Shear and Normal Pressure plot vs P_comp for different F---#
#     fig, ax1 = plt.subplots(figsize=(10,8))
#     ax2 = ax1.twinx()
#     
#     
#     P0751,Cs0751 = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-1',[13,15],3)
#     n1, = ax1.plot(P0751,Cs0751[:,0], 'ro')
#     s1, = ax2.plot(P0751,Cs0751[:,1], 'rs')
#     
#     
#     P0758,Cs0758 = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-8',[13,15],3)
#     n2, = ax1.plot(P0758,Cs0758[:,0], 'go')
#     s2, = ax2.plot(P0758,Cs0758[:,1], 'gs')
#     
#     
#     P07516,Cs07516 = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-16',[13,15],3)
#     n3, = ax1.plot(P07516,Cs07516[:,0], 'bo')
#     s3, = ax2.plot(P07516,Cs07516[:,1], 'bs')
#         
#     
# 
#     plt.legend([n1,s1], ['$P_{zz}$','$P_{xz}$'], loc=9)
#     ax1.set(xlabel='$P_{comp}$', ylabel= '$P_{zz}$')
#     ax2.set(ylabel = '$P_{xz}$')
#     ax1.minorticks_on()
#     ax2.minorticks_on()
#     plt.savefig('8-PzzPxz vs Pcomp - vF.jpg')
# 
# 
#     #---Plot the COF vs Pcomp---#
#     
#     P0151,Cs0151 = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.15\MDPBB-FS-1',[13,15],3)
#     P0158,Cs0158 = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.15\MDPBB-FS-8',[13,15],3)
#     #P01516,Cs01516 = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.15\MDPBB-FS-16',[13,15],3)
#     
#     mu1 = -1*Cs0751[:,1]/Cs0751[:,0]
#     mu8 = -1*Cs0758[:,1]/Cs0758[:,0]
#     mu16 = -1*Cs07516[:,1]/Cs07516[:,0]
#     
#     muu1 = -1*Cs0151[:,1]/Cs0151[:,0]
#     muu8 = -1*Cs0158[:,1]/Cs0158[:,0]
#     #muu16 = -1*Cs01516[:,1]/Cs01516[:,0]
#     
#     print(P0158)
#     print(muu8)
#     
#     fig, ax1 = plt.subplots(figsize=(10,8))
#     l1, = ax1.plot(P0751,mu1, 'ro')
#     l2, = ax1.plot(P0751,mu8, 'go')
#     l3, = ax1.plot(P07516,mu16, 'bo')
#     
#     ll1, = ax1.plot(P0151,muu1, 'rs')
#     ll2, = ax1.plot(P0158,muu8, 'gs')
#     plt.legend([l1, l2, l3], ['$f=1$','$f=8$','$f=16$' ], loc=9)
# 
#     
#     #ll3, = ax1.plot(P01516,muu16, 'bs')
#     
#     #ll3, = ax1.plot(P01516,muu16, 'bs')
#     
# 
#     
#     ax1.minorticks_on()
#     ax1.set(xlabel=r'$P_{comp}$', ylabel= r'$\mu$' , xlim=[0,10])
# 
#     plt.savefig('9-COF vs Pcomp - vF.jpg')
# #     
#     
#     #---Shear and Normal Pressure plot vs V_Wall for different grafting densities---#
#     fig, ax1 = plt.subplots(figsize=(10,8))
#     ax2 = ax1.twinx()
#     
#     
#     Vs1,Cs1 = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\P-1\MDPBB-FS-1',[13,15,23],3)
#     n1, = ax1.plot(Cs1[:,2],Cs1[:,0], 'ro')
#     s1, = ax2.plot(Cs1[:,2],Cs1[:,1], 'bo')
#     
#     Vs16,Cs16 = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\P-1\MDPBB-FS-16',[13,15,23],3)
#     n16, = ax1.plot(Cs16[:,2],Cs16[:,0], 'rs')
#     s16, = ax2.plot(Cs16[:,2],Cs16[:,1], 'bs')    
#   
#     
#     plt.legend([n1,s1], ['$P_{zz}$','$P_{xz}$'], loc=9)
#     ax1.set(xlabel=r'$\.\gamma$', ylabel= '$P_{zz}$')
#     ax2.set(ylabel = '$P_{xz}$')
#     ax1.minorticks_on()
#     ax2.minorticks_on()
#     plt.savefig('10-PzzPxz vs srate - vF.jpg')
# 
# 
#     #---Plot the COF vs srate---#
#     
#     mu1 = -1*Cs1[:,1]/Cs1[:,0]
#     mu16 = -1*Cs16[:,1]/Cs16[:,0]
#     
#     fig, ax1 = plt.subplots(figsize=(10,8))
#     l1, = ax1.plot(Cs1[:,2],mu1, 'bo')
#     l16, = ax1.plot(Cs16[:,2],mu16, 'rs')
#     plt.legend([l1,l16], [r'f=1',r'f=16'], loc=1)     
#     
#     ax1.minorticks_on()
#     ax1.set(xlabel=r'$\.\gamma$', ylabel= r'$\mu$', xlim=[0,0.3], ylim=[0,0.45])
# 
#     plt.savefig('11-COF vs srate - vF.jpg')
#       
    
#     
#     zDDp,zDPp = read_mops(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\Stresses\MOP\MDPBB-ECS7\1_V')
#     
#     fig, ax1 = plt.subplots(figsize=(10,8))
#     ax2 = ax1.twinx()
#     l1, = ax1.plot(zDPp[0],zDPp[1][:,0], 'r')
#     l2, = ax1.plot(zDPp[0],zDPp[1][:,1], 'g')
#     l3, = ax1.plot(zDPp[0],zDPp[1][:,2], 'b')
#     l4, = ax2.plot(zDDp[0], zDDp[1], 'g--')
# 
# 
#     plt.legend([l1,l2,l3,l4], [r'Pzx','Pzy','Pzz', 'Density'], loc=10)
#     ax1.set(xlim=[0,1],xlabel='$z$', ylabel= r'$P_z(i)$')
#     ax2.set(ylabel= r'$\phi(z)$')
#     ax1.minorticks_on()
#     ax2.minorticks_on()
#     plt.savefig('12-PnDprofiles.jpg')    
    