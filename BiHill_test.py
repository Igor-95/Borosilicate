import pandas as pd
import matplotlib.pyplot as plt
import BiHill
import ig_modules_plot as igp

df = pd.read_excel("bla.xlsx")

#low = [10, 12, 15, 20, 25, 32]
low = [10, 10, 10, 10, 10, 10]
high = [175, 200, 150, 200, 225, 55]
datasets = [2,3,4,5,6,7,8,9]#,10]
fit_quality_100, fit_quality_125 = [], []
colums = df.columns
data_columns = colums[2:len(colums)-1]
i = 0
itervalue = 2
print(colums)
print(data_columns)

for dataset in datasets:
    #dataset = 10
    while i < itervalue:
        df1 = df[df.iloc[:,1] >= low[i]]
        df1 = df1[df1.iloc[:,1] <= high[i]]
        x,y,z = df1.iloc[:,1], df1.iloc[:,dataset], []
        params = BiHill.bihill_fit(x,y)
        print(params)
        x = df.iloc[:,1]
        for a in x:
            z.append(BiHill.bihill(a,params[0],params[1],params[2],params[3],params[4]))
        if i == 0:
            fit_quality_100.append(params[5])
        else:
            fit_quality_125.append(params[5])
        plt.plot(x,z, label=f"w_low={low[i]} , w_high={high[i]}")
        print(f"R2 = {high[i]}:{params[5]}, {colums[dataset]}")
        i = i+1
       
    i = 0
    x,y = df.iloc[:,1], df.iloc[:,dataset]
    plt.plot(x,y, color="red", linewidth=2, label=f"Data {colums[dataset]}")
    plt.xlabel("cm^-1")
    plt.ylabel("a.u.")
    plt.legend()
    plt.show()

igp.double_barplot(datasets,fit_quality_100,fit_quality_125,
                   "100","125","Glass Type","R^2","R^2",0.95,1,0.95,1,"comparison 100 vs 125 cm^-1",data_columns)