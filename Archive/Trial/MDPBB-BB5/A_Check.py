# This is a python script that will generate a LAMMPS molecule file for use in
# Polymer Brush

import numpy as np
import os

def A_Check(r,n):

    with open("A_Check.txt","a+") as fdata:
        fdata.write('{} \n' .format(r))
        fdata.close()
    d = np.loadtxt("A_Check.txt", dtype=int)
    u = np.unique(d)
    if d.size == 1:
        c = 0
        print "a"
    elif d.size == n:
        if u.size == d.size:
            c = 0
            os.remove("A_Check.txt")
            print "b"
        else:
            c = 1
            print "c"
            os.remove("A_Check.txt")
            with open("A_Check.txt","w+") as fdata:
                for i in range(u.size):
                    fdata.write("%d \n" %(u[i]))
                fdata.close()
    else:
        if u.size == d.size:
            c = 0
            print "d"
        else:
            c = 1

            os.remove("A_Check.txt")
            with open("A_Check.txt","w+") as fdata:
                for i in range(u.size):
                    fdata.write("%d \n" %(u[i]))
                fdata.close()
            print "e"
    return c


# if __name__ == '__main__':
#
#     for i in range(10):
#         v = A_Check(i+1,10)
#         print(v)
