import numpy as np
from scipy import optimize
import pandas as pd


# function for optimizing, A = Area, mu = fit parameter, sig = fit parameter, y0 = baseline
def lognorm(x,A, mu, sig, y0):
    return y0 + (A/(np.sqrt(2*np.pi)*sig*x)) * np.exp((-np.log(x/mu)**2)/(2*sig**2))

# 1. optimize
def lognorm_fit(data):
    ini = pd.read_csv("ini.txt", delim_whitespace=True)
    xmin, xmax = ini.loc[1, "value"], ini.loc[2, "value"]
    data = data[data.iloc[:,0] >= xmin]
    data = data[data.iloc[:,0] <= xmax]
    x,y = data.iloc[:,0], data.iloc[:,1]
    x,y,z, result = np.array(x), np.array(y), [], []
    params, something = optimize.curve_fit(lognorm,x,y,p0=[0.3, 70, 0.6, 0.001])
    for a in x:
        value = lognorm(a,params[0],params[1], params[2], params[3])
        z.append(value)
    
    for a in params:
        result.append(a)
    
    coeff ,coeff1= np.corrcoef(y,z)
    R2 = round(coeff[1]**2, 3)
    result.append(R2)
    
    return result
