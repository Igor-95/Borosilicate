import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt

def lognorm(x,Amp, mu, sig, y0):
    return y0 + (Amp/(np.sqrt(2*np.pi)*sig*x)) * np.exp((-np.log(x/mu)**2)/(2*sig**2))


def lognorm_fit(x,y):
    params = optimize.curve_fit(lognorm,x,y,p0=[0.3, 70, 0.66,0.001])
    return params

df = pd.read_excel("bla.xlsx")
# df = df[df.iloc[:,1] > 10]
df = df[df.iloc[:,1] < 240]

x,y = np.array(df.iloc[:,1]), np.array(df.iloc[:,2])
params, something = lognorm_fit(x,y)
print(params)


amp_O = 0.30807
mu_O = 67.97528
sig_O = 0.65896
y0_O = 5.06532E-4
z, z1 = [], []
r2 = 0
for a in x:
    value = lognorm(a,amp_O,mu_O,sig_O, y0_O)
    z.append(value)
    value1 = lognorm(a,params[0],params[1], params[2], params[3])
    z1.append(value1)
    r2 += (value-value1)**2

print(f"mean_square_error = {r2}")
plt.plot(x,z)
plt.plot(x,z1)
plt.show()
