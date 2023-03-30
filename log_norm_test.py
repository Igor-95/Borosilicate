import numpy as np
from scipy.stats import lognorm
import pandas as pd
import matplotlib.pyplot as plt
import lognorm 
import ig_modules_plot as igp

df = pd.read_excel("bla.xlsx")

#low = [10, 12, 15, 20, 25, 32]
low = [10, 10, 10, 10, 10, 10]
high = [100, 125, 150, 200, 225, 55]
datasets = [2,3,4,5,6,7,8,9,10]
fit_quality_100, fit_quality_125 = [], []
colums = df.columns
data_columns = colums[2:len(colums)]
i = 0
itervalue = 2
ini = pd.read_csv("ini.txt", delim_whitespace=True)
xmin, xmax = ini.loc[1, "value"], ini.loc[2, "value"]

for dataset in datasets:
    
    df1 = pd.concat((df.iloc[:,1], df.loc[:,data_columns[dataset-2]]), axis=1)
    z = []
    params = Lognorm.lognorm_fit(df1)
    x = df.iloc[:,1]
    for a in x:
        z.append(Lognorm.lognorm(a,params[0],params[1],params[2],params[3]))
    if i == 0:
        fit_quality_100.append(params[4])
    else:
        fit_quality_125.append(params[4])
    plt.plot(x,z, label=f"w_low={xmin} , w_high={xmax}")
    print(f"R2 = {high[i]}:{params[4]}, {colums[dataset]}")
    i = i+1
       
    i = 0
    x,y = df.iloc[:,1], df.iloc[:,dataset]
    plt.plot(x,y, color="red", linewidth=2, label=f"Data {colums[dataset]}")
    plt.xlabel("cm^-1")
    plt.ylabel("a.u.")
    plt.legend()
    plt.show()

#igp.double_barplot(datasets,fit_quality_100,fit_quality_125,
                  # "100","125","Glass Type","R^2","R^2",0.95,1,0.95,1,"comparison 100 vs 125 cm^-1",data_columns)