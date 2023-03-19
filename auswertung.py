import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import glob


cwd = os.getcwd() + "\__data__"
all_files = glob.glob(os.path.join(cwd, "*"))

for a in all_files:
    df = pd.read_excel(f"{a}")
    df = df.sort_values("Raman Shift",axis=0, ascending=True)
    df1 = df[df.iloc[:,0] > 25]

columns = list(df.columns)

def integrate(x,y):

    area = np.trapz(y,x)
    return abs(area)


i = 0
for a in range(len(columns)):

    if i == 0:
        i+=1
        continue
    else:
        x,y = df.iloc[:,0], df.iloc[:,i]
        y = y - np.min(df1.iloc[:,i]) # muss für werte größer 100 cm^-1 gemacht werden
        y = y/integrate(x,y)
        print(integrate(x,y))
        df.iloc[:,i] = y
        plt.plot(x,y, label=columns[i])
        i +=1

df.to_excel("bla.xlsx")
plt.legend()
plt.show()