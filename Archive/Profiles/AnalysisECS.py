# This is a python script that will generate a LAMMPS molecule file for use in
# Polymer Brush
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy


def brush_height(z,dp):
    z0 = 1
    s = np.sum(dp)
    t0 = 0.1*s/100
    t1 = 90*s/100
    cs = np.cumsum(dp)
    id0 = np.where(cs < t0)
    id1 = np.where(cs < t1)
    i0 = id0[0]
    i1 = id1[0]
    z0 = z[i0[-1]]
    z1 = z[i1[-1]]
    h = (z1 - z0)

    return h

def heights(z, edps):
    m,n = edps.shape
    hs = np.zeros(n)
    for i in range(n):
        hs[i] = brush_height(z,edps[:,i])
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


def read_profs(path,stage):
    ocwd = os.getcwd()
    os.chdir(path)
    if stage == 1:
        a = 'abdpe.csv'
        b = 'bbdpe.csv'
        c = 'tbdpe.csv'
        d = 'temps.csv'
        e = 'velps.csv'
        f = 'mope.csv'
        g = 'equil.csv'
    elif stage == 2:
        a = 'abdpc.csv'
        b = 'bbdpc.csv'
        c = 'tbdpc.csv'
        d = 'temps.csv'
        e = 'velps.csv'
        f = 'mopc.csv'
        g = 'comp.csv'
    elif stage == 3:
        a = 'abdps.csv'
        b = 'bbdps.csv'
        c = 'tbdps.csv'
        d = 'temps.csv'
        e = 'velps.csv'
        f = 'mops.csv'
        g = 'shear.csv'
    
    abeads = (pd.pandas.read_csv(a)).values
    zZ = abeads[1:,1]
    Da = abeads[1:,-1]
    zZDa = [zZ, Da]
    
    bbeads = (pd.pandas.read_csv(b)).values
    zZ = bbeads[1:,1]
    Db = bbeads[1:,-1]
    zZDb = [zZ, Db]
    
    tbeads = (pd.pandas.read_csv(c)).values
    zZ = tbeads[1:,1]
    Dt = tbeads[1:,-1]
    zZDt = [zZ, Dt]
    
    temps = (pd.pandas.read_csv(d)).values
    zZ = temps[1:,1]
    Tp = temps[1:,-1]
    zZTp = [zZ, Tp]
    
    velps = (pd.pandas.read_csv(e)).values
    zZ = velps[1:,1]
    Vp = velps[1:,-1]
    zZVp = [zZ, Vp]
    
    mops = (pd.pandas.read_csv(f)).values
    zZ = mops[1:,1]
    Pp = mops[1:,2:]
    zZPp = [zZ, Pp] 
    
    thermo = (pd.pandas.read_csv(g)).values
    zo = thermo[-1,33]
    D = thermo[-1,32]
    
    os.chdir(ocwd)
    return zZDa,zZDt,zZDb,zZVp,zZTp,zZPp,zo,D
    
def zZ_profs(path,stage):
    ocwd = os.getcwd()
    os.chdir(path)
    zZDa,zZDt,zZDb,zZVp,zZTp,zZPp,zo,D = read_profs(path,stage)
    
    zZDa[0] = (zZDa[0]-zo)
    zZDt[0] = (zZDt[0]-zo)
    zZDb[0] = (zZDb[0]-zo)
    zZVp[0] = (zZVp[0]-zo)
    zZTp[0] = (zZTp[0]-zo)
    zZPp[0] = (zZPp[0]-zo)
    
    I = np.trapz((zZDb[1]*zZDt[1]),zZDb[0])

    os.chdir(ocwd)

    return zZDa,zZDt,zZDb,zZVp,zZTp,zZPp,I,D

    
def zD_profs(path,stage):
    ocwd = os.getcwd()
    os.chdir(path)
    zZDa,zZDt,zZDb,zZVp,zZTp,zZPp,zo,D = read_profs(path,stage)
    
    zZDa[0] = (zZDa[0]-zo)/D
    zZDt[0] = (zZDt[0]-zo)/D
    zZDb[0] = (zZDb[0]-zo)/D
    zZVp[0] = (zZVp[0]-zo)/D
    zZTp[0] = (zZTp[0]-zo)/D
    zZPp[0] = (zZPp[0]-zo)/D

    os.chdir(ocwd)

    return zZDa,zZDt,zZDb,zZVp,zZTp,zZPp
    
# def IvsD(path):
#     ocwd = os.getcwd()
#     os.chdir(path)
#     dirs = filter(os.path.isdir ,os.listdir('.'))
#     n = len(dirs)
#     Ps = np.zeros(n)
#     Is = np.zeros(n)
#     Ds = np.zeros(n)
#     i=0
#     for dir in dirs:  
#         S = dir.split("_")
#         Ps[i] = float(S[0])
#         c = os.path.join('.',dir)
#         zZDa,zZDt,zZDb,zZVp,zZTp,zZPp,Is[i],Ds[i] = zZ_profs(c,2)
#         i = i + 1 
#     Ps,Is,Ds = zip(*sorted(zip(Ps,Is,Ds)))
#     os.chdir(ocwd)
#     return Ps,Is,Ds
#     
#     

if __name__ == '__main__':
    
#  0   1  2   3   4    5       6       7           8       9       10   11  12  13  14  15  16   17     18      19         20     21         22      23         24         25          26       27       28      29       30    31    32    33      34         35             36      37           38         39        40          41           
# step et ke pe epair temp c_melTemp c_wallTemp v_Fcatom v_Pcomp2 press pxx pyy pzz pxy pxz pyz c_Vir c_Vir[1] c_Vir[2] c_Vir[3] c_Vir[4] c_Vir[5] c_Vir[6] c_melPress c_wallPress v_melDens v_surfcov v_aveRg v_Vwall v_srate v_D v_bwzmax zhi c_fbwall[1] c_fbwall[3] c_ftwall[1] c_ftwall[3] c_ggbot[1] c_ggbot[3] c_ggtop[1] c_ggtop[3]
    
#   
#  
#     print('Analysis')
# 
#     #---EQUILIBRIUM---#
# 
#     #---Density Profile---#
#     zZDa,zZDt,zZDb,zZVp,zZTp,zZPp,I,D = zZ_profs(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\Profiles\MDPBB-ECS1\1_V',1)
# 
#     plt.rcParams.update({'font.size': 15})
#     fig, ax = plt.subplots(figsize=(10,8))
#     l1, = ax.plot(zZDb[0] , zZDb[1], 'g--')
#     plt.legend([l1], [r'$\rho_g$'], loc=1)
#     ax.set(xlim=[0,D/3],xlabel='$z$', ylabel= r'$\phi(z)$' , title='')
#     ax.minorticks_on()
#     plt.savefig('1-EquilDP.jpg')   
#     
#     #---Brush Height---#
#     
#     h = brush_height(zZDb[0],zZDb[1])
#     print(h)
#     
#     #---COMPRESSION---#
# 
#     #---Compression Curve---#
#     
#     Ps, Ds = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.15\MDPBB-FS-1',[24],2)
# 
#     plt.rcParams.update({'font.size': 15})
#     fig, ax = plt.subplots(figsize=(10,8))
#     l1, = ax.plot(Ds , Ps, 'g--')
#     plt.legend([l1], [r'$\rho_g$'], loc=1)
#     ax.set(xlim=[0,30],xlabel='$D$', ylabel= r'$P_{comp}$' , title='')
#     ax.minorticks_on()
#     plt.savefig('2-CompCurve.jpg')
# 
# 
#     #---Interpenetration Profiles---#
#     
#     zDDa,zDDt,zDDb,zDVp,zDTp,zDPp = zD_profs(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\Profiles\MDPBB-ECS1\1_V',3)
# 
#     fig, ax = plt.subplots(figsize=(10,8))
#     lb1, = ax.plot(zDDb[0] , zDDb[1], 'r--')
#     lt1, = ax.plot(zDDt[0] , zDDt[1], 'b--')
#     la1, = ax.plot(zDDa[0] , zDDa[1], 'g')
#     
# #    plt.legend([la1], ['P=1'], loc=1)
#     ax.set(xlim=[0,1],xlabel='$z/D$', ylabel= r'$\phi(z)$' , title='')
#     ax.minorticks_on()
#     plt.savefig('3-Interpenetration.jpg')
#     
#     # #---Interpenetration vs D---#
#     # 
#     # 
#     # Ps,Is,Ds = IvsD(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-1')
#     # 
#     # fig, ax = plt.subplots(figsize=(10,8))
#     # l1, = ax.plot(Ds,Is, 'go')
#     # 
#     # plt.legend([l1],[r'$\rho_g$'], loc=9)
#     # ax.set(xlabel='$D$', ylabel= r'$I(D)$' , title='')
#     # ax.minorticks_on()
#     # plt.savefig('4-IvsD.jpg')
#     # 
#     # 
#     
#     #---SHEARING---#
#     
#     #---T,V and Density profiles---#
#     
#     zDDa,zDDt,zDDb,zDVp,zDTp,zDPp = zD_profs(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\Profiles\MDPBB-ECS1\1_V',3)
#     
#     fig, ax1 = plt.subplots(figsize=(10,8))
#     ax2 = ax1.twinx()   
#     T1, = ax1.plot(zDTp[0],zDTp[1], 'r-')
#     D1, = ax1.plot(zDDa[0],zDDa[1], 'g-')
#     V1, = ax2.plot(zDVp[0],zDVp[1], 'b-')
#     
#     plt.legend([T1,V1,D1], [r'Temperature','Velocity','Density'], loc=4)
#     ax1.set(xlim=[0,1],ylim=[0,1.5],xlabel='$z/D$', ylabel= r'$T, \phi(z)$')
#     ax2.set(ylim=[-0.25,0.25], ylabel= r'V')
#     ax1.minorticks_on()
#     ax2.minorticks_on()
#     plt.savefig('4-TnVprofiles.jpg') 
#     
#     #---Stress Profiles---#
#     
#     fig, ax1 = plt.subplots(figsize=(10,8))
#     ax2 = ax1.twinx()   
#     Pzx, = ax1.plot(zDPp[0],zDPp[1][:,0], 'r-')
#     Pzz, = ax1.plot(zDPp[0],zDPp[1][:,2], 'b-')
#     D, = ax2.plot(zDDa[0],zDDa[1], 'g-')
# 
#     plt.legend([Pzx,Pzz,D], [r'$P_(zx)$',r'$P_(zz)$',r'$\phi(z)$'], loc=4)
#     ax1.set(xlim=[0,1],ylim=[0,5],xlabel='$z/D$', ylabel= r'$\phi(z)$')
#     ax2.set(ylim=[0,1.5], ylabel= r'$\phi(z)$')
#     ax1.minorticks_on()
#     ax2.minorticks_on()
#     plt.savefig('5-PnDprofiles.jpg')
#     
#     #---Time Evolution---#
#     
#     
#     ts,Vt = thermo_time(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\GD-0.075\MDPBB-FS-1\1_P',[5,24],3)
# 
#     fig, ax1 = plt.subplots(figsize=(10,8))
#     l1, = ax1.plot(ts[-1000:],Vt[-1000:,0], 'ro')
#     ax2 = ax1.twinx()
#     l2, = ax2.plot(ts[-1000:],Vt[-1000:,1], 'bo')
#     ax1.minorticks_on()
#     ax2.minorticks_on()
#     plt.savefig('6-Time Evolution.jpg') 
#     
#     
#     #---Shear and Normal stress components vs srate---#
#     
#     Vs1,Cs1 = read_thermo(r'C:\Users\klay_\OneDrive - Imperial College London\PhD\PhD-Sims\FiniteSize\P-1\MDPBB-FS-1',[13,15,23],3)    
#     
#     fig, ax1 = plt.subplots(figsize=(10,8))
#     ax2 = ax1.twinx()
#     n1, = ax1.plot(Cs1[:,2],Cs1[:,0], 'ro')
#     s1, = ax2.plot(Cs1[:,2],Cs1[:,1], 'bo')
#     
#     plt.legend([n1,s1], ['$P_{zz}$','$P_{xz}$'], loc=9)
#     ax1.set(xlabel=r'$\.\gamma$', ylabel= '$P_{zz}$')
#     ax2.set(ylabel = '$P_{xz}$')
#     ax1.minorticks_on()
#     ax2.minorticks_on()
#     plt.savefig('7-PzzPxz vs srate - vF.jpg')
#     
#     
#     mu1 = -1*Cs1[:,1]/Cs1[:,0]
#   
#     fig, ax1 = plt.subplots(figsize=(10,8))
#     l1, = ax1.plot(Cs1[:,2],mu1, 'bo')
#     
#     plt.legend([l1], [r'f=1',], loc=1)     
#     ax1.set(xlabel=r'$\.\gamma$', ylabel= r'$\mu$', xlim=[0,0.3], ylim=[0,0.45])
#     ax1.minorticks_on()
#     plt.savefig('8-COF vs srate - vF.jpg')
#     
