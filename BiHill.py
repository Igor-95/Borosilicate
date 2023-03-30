import numpy as np
from scipy import optimize
import pandas as pd


def bihill(x,Pm,ka,Ha,ki,Hi):
    return Pm/((1+(ka/x)**Ha)*(1+(x/ki)**Hi))

# data must be pandas dataframe 1st col = xdata, 2nd col = y data
def bihill_fit(data):
    ini = pd.read_csv("ini.txt", delim_whitespace=True)
    xmin, xmax = ini.loc[1, "value"], ini.loc[2, "value"]
    data = data[data.iloc[:,0] >= xmin]
    data = data[data.iloc[:,0] <= xmax]
    x,y = data.iloc[:,0], data.iloc[:,1]
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
