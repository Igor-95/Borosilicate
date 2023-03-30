import center_of_gravity as cg
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_excel("bla.xlsx")
columns = df.columns
i = 0
w_min1 = [315]
w_max1 = [375]
results = pd.DataFrame()

for index in w_min1:
    y_colum = 2
    for a in range(len(columns)-2):
        df1 = pd.concat((df.iloc[:,1], df.iloc[:,y_colum]), axis=1)
        data = cg.Cog(df1).baseline_points(w_min=w_min1[i], w_max=w_max1[i])
        params = cg.Cog(df1).baseline_points(w_min=w_min1[i], w_max=w_max1[i],parameters=True)
        x,y,z = df1.iloc[:,0],df1.iloc[:,1],[]
        for b in x:
            z.append(params[0]*b**2 + params[1]*b + params[2])
        

        results_temp = pd.DataFrame(columns=["spectra", "x_min", "y_min", "range"])
        results_temp.loc[0,"spectra"] = columns[y_colum]
        results_temp.loc[0,"x_min"] = data[0]
        results_temp.loc[0,"y_min"] = data[1]
        results_temp.loc[0,"range"] = f"{w_min1[i]} - {w_max1[i]}"
        results = pd.concat((results,results_temp), axis=0)
        del results_temp
        plt.plot(x,y, label=f"{columns[y_colum]}")
        plt.plot(x,z, label=f"Fit{w_min1[i]} - {w_max1[i]}")
        plt.plot(data[0], data[1], ".", label="minimum")
        plt.xlim(w_min1[0]-150,w_max1[0]+250)
        plt.ylim(-0.0001, 0.0015)
        plt.legend()
        plt.grid()
        plt.savefig(f"__fits__\COG_left\{columns[y_colum]}_{w_min1[i]}-{w_max1[i]}.png")
        plt.close()
        #plt.show()
        y_colum += 1
    
    i += 1

results.to_excel("__fits__\COG_left\minimu_left.xlsx", index=False)