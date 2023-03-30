import pandas as pd
import matplotlib.pyplot as plt
import center_of_gravity as cg
import numpy as np

df = pd.read_excel("bla.xlsx")
columns = df.columns
i = 0
w_min1 = [560, 500, 550]
w_max1 = [620, 600, 650]
y_colum = 2
results = pd.DataFrame()

for a in range(len(columns)-2):
    df1 = pd.concat((df.iloc[:,1], df.iloc[:,y_colum]), axis=1)
    print(df1)
    m,b, left_low, left_high, right_low, right_high = cg.Cog(df1).baseline_corr(parameters=True)
    x,y,z = df1.iloc[:,0], df1.iloc[:,1], []

    for a in x:
        z.append(m*a+b)
    
    y1 = np.array(y)-z

    results_temp = pd.DataFrame(columns=["spectra","slope", "y_interception","cog_wavenumber", "cog_area"])
    results_temp.loc[0,"spectra"] = columns[y_colum]
    results_temp.loc[0,"slope"] = m
    results_temp.loc[0,"y_interception"] = b
    area_x, area = cg.Cog(df1).center_of_gravity(plot=True)
    cog, egal = cg.Cog(df1).center_of_gravity(plot=False)
    results_temp.loc[0,"cog_wavenumber"] = cog
    results_temp.loc[0,"cog_area"] = egal
    results = pd.concat((results,results_temp), axis=0)
    plt.plot(x,y, label=f"original data {columns[y_colum]}")
    plt.plot(x,z, "--",label=f"baseline: left({left_low} - {left_high}), right({right_low} - {right_high})", color="red")
    plt.plot(x,y1, label=f"original data {columns[y_colum]} - baseline")
    plt.grid()
    plt.xlim(200,700)
    plt.ylim(-0.0001, 0.0015)
    plt.vlines(cog,0,0.0015,linestyles="--" ,label=f"center of gravity {cog}", color="black")
    plt.legend()
    #plt.show()
    plt.savefig(f"__fits__\COG_baseline\_baseline_{columns[y_colum]}.png")
    plt.close()
    
    plt.plot(area_x,area,"o-")
    plt.vlines(cog,0,0.5,linestyles="--" ,label=f"center of gravity {cog}", color="black")
    plt.hlines(0.5,0,cog,linestyles="--", color="black")
    plt.legend()
    plt.grid()
    plt.xlim(300,650)
    #plt.show()
    plt.savefig(f"__fits__\COG_baseline\_cog_integral_{columns[y_colum]}.png")
    plt.close()

    y_colum += 1

results.to_excel("__fits__\COG_baseline\_baselines&cog.xlsx", index=False)