import numpy as np
from scipy import optimize


def bihill(x,Pm,ka,Ha,ki,Hi):
    return Pm/((1+(ka/x)**Ha)*(1+(x/ki)**Hi))


def bihill_fit(x,y):
    x,y,z, result = np.array(x), np.array(y), [], []
    params, something = optimize.curve_fit(bihill,x,y,p0=[0.006, 25,3,60,1.8])
    for a in x:
        value = bihill(a,params[0],params[1], params[2], params[3],params[4])
        z.append(value)
    
    for a in params:
        result.append(a)
    
    coeff ,coeff1= np.corrcoef(y,z)
    R2 = round(coeff[1]**2, 3)
    result.append(R2)
    
    return result
