import numpy as np
from scipy.stats import lognorm
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("LogMormaltest.txt", delim_whitespace=True, decimal=",")

x,y  = df.iloc[:,0], df.iloc[:,1]

data = np.array(y)

params = lognorm.fit(data)
print(params)

def lognorm(x):
    pass

plt.plot(x,y)
plt.show()