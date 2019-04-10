import numpy as np
import pandas as pd
import matlab.engine as matlab
import matlab as mat


def start_matlab():
    return matlab.start_matlab()
def quit_matlab(matlab_engine):
    matlab_engine.quit()
def ieee_feeder_mapper(matlab_engine):
    return matlab_engine.ieee_feeder_mapper(13,nargout=5) # As the function will return two outputs, nargout=2


def FBSfun(V0,loads,Z,B):
    n=len(Z)
    V=np.zeros(shape=(n,1),dtype=complex)
    s=np.copy(V)
    V[0,0]=V0
    print(V)
    I = np.zeros(shape=(n, 1),dtype=complex)
    I[0, 0] = 0

    T=[]
    J=[1]

    for k in range(2,n+1):
        t=np.sum(B[k-1,:])
        if (t==-1):
            T.append(k)
        elif (t>=1):
            J.append(k)

    tol = 0.0001
    iter = 0
    Vtest = 0
    #

    while(abs(Vtest-V0) >=tol):
        for k in range(0, n - 1):
            idx = np.where(B[k, :] > 0)
            V[idx,0]=V[k,0]-np.multiply(Z[idx,0],I[idx,0])
        print(V)

        for t in range(len(T)-1,-1,-1):
            print(T)
			  t=T[t]
			
            v=np.array([1, abs(V[t,0]),abs(V[t,0])**2])
            s[t,0]=np.dot(loads[t,:],np.transpose(v))
            I[t,0]=np.conj(s[t,0]/v[t,0])
            flag=True
            idx= np.where(B[t,:] == -1)
            while(flag):
                V[idx,0] = V[t,0] + Z[t,0]*I[t,0]
                v=np.array([1, abs(V[idx,0]),abs(V[idx,0])**2])
                s[idx,0]=np.dot(loads[idx,:],np.transpose(v))
                I[idx,0]=np.conj(s[idx,0]/v[idx,0])+I[t,0]
                if (len(np.where(J==idx))==0):
                    t=idx
                    idx=np.where(B[idx,:]==-1)
                else:
                    flag=False

        for k in range(len(J),1,-1):
            t=J[k]
            v = np.array([1, abs(V[t, 0]), abs(V[t, 0]) ** 2])
            s[t, 0] = np.dot(loads[t, :], np.transpose(v))
            load_current=np.conj(s[t, 0] / v[t, 0])
            idx=np.where(B[t,0]>0)
            Itot = load_current
            for y in range(0,len(idx)):
                Itot=Itot+I[idx[y],0]
            I[t,0]= Itot
            flag= True
            idx = np.where(B[t, :] == -1)
            while flag:
                V[idx, 0] = V[t, 0] + Z[t, 0] * I[t, 0]
                v = np.array([1, abs(V[t, 0]), abs(V[t, 0]) ** 2])
                s[t, 0] = np.dot(loads[t, :], np.transpose(v))
                I[idx, 0] = np.conj(s[idx, 0] / v[idx, 0]) + I[t, 0]
                if (len(np.where(J==idx))==0):
                    t=idx
                    idx = np.where(B[idx, :] == -1)
                else:
                    flag=False

        Vtest=V[0,0]
        iter+=1

    V[0,0] = V0
    S=np.multiply(V,I)
    return V,I,S,iter




matlab_engine=start_matlab()
FM, Z, paths, nodelist, loadlist=ieee_feeder_mapper(matlab_engine)
quit_matlab(matlab_engine)
FeederMap=np.array(FM)
Z=np.array(Z)
Vbase = 4.16e3
Sbase = 1

Zbase = Vbase*Vbase/Sbase
Ibase = Sbase/Vbase
FBSfun(1,[],Z,FeederMap)