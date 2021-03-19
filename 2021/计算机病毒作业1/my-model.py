import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

def plotfuc(func):
    S0 = 0.9
    I0 = 0.1
    R0 = 0.0
    t = np.linspace(0, 40, 10000)
    res = np.array(scipy.integrate.odeint(func, [S0, I0, R0], t,args=(0.35,0.1)))
    plt.figure(figsize=[6, 4])
    plt.plot(t, res[:, 0], label='S(t)')
    plt.plot(t, res[:, 1], label='I(t)')
    plt.plot(t, res[:, 2], label='R(t)')
    plt.legend()
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('proportions')
    plt.show()
    

def code_1_1_1(y, t, beta, gamma):
    s, i, r= y
    lambda_k = (1-(1-i)**4)*beta
    ds_dt = -lambda_k * s
    di_dt = lambda_k * s
    dr_dt = 0
    return ([ds_dt, di_dt, dr_dt])

def code_1_1_2(y,t,beta, gamma):
    s, i, r = y
    lambda_k = (1-(1-i)**4)*beta
    ds_dt = -lambda_k*s-gamma*s
    di_dt = lambda_k*s-gamma*i
    dr_dt = gamma*(s+i)
    return ([ds_dt, di_dt, dr_dt])

def code_2_1_1(y,t,beta, gamma):
    s, i, r  = y
    ds_dt = -beta*s*(1-i)
    di_dt = beta*s*(1-i)
    dr_dt = 0
    return ([ds_dt, di_dt, dr_dt])

def code_2_1_2(y,t,beta, gamma):
    s, i, r= y
    ds_dt = -beta*s*(1-i)
    di_dt = beta*s*(1-i)-gamma*i
    dr_dt = gamma*i
    return ([ds_dt, di_dt, dr_dt])

plotfuc(code_1_1_1)

plotfuc(code_1_1_2)

plotfuc(code_2_1_1)

plotfuc(code_2_1_2)
