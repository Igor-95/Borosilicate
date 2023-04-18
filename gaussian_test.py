import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gaussian
import ig_modules_plot as igp
from scipy import optimize


df = pd.read_excel("bla.xlsx")
datasets = [2,3,4,5,6,7,8,9]#,10]
ini = pd.read_csv("ini.txt", delim_whitespace=True)
columns = df.columns
i = 0
itervalue = 1
A_gaus_right = []
A_gaus_left = []
mu_left = []
mu_right = []
col_name = []
    

for dataset in datasets:
    xmin, xmax = ini.loc[9, "value"], ini.loc[10, "value"]
    df1 = pd.concat((df.iloc[:,1], df.iloc[:,dataset]), axis=1)
    col_name.append(columns[dataset])
    params = gaussian.Gausfit(df1).gaussian_fit()
    df1 = df1[df1.iloc[:,0] >= xmin]
    df1 = df1[df1.iloc[:,0] <= xmax]
    x,y = df1.iloc[:,0], df1.iloc[:,1]
    gaus1, gaus2,gaus3, combined= [], [], [], []
    for a in x:
        if len(params) == 8:
            gaus1.append(gaussian.Gausfit(df1).gaussian(a,params[0],params[1],params[2],params[3]))
            gaus2.append(gaussian.Gausfit(df1).gaussian(a,params[4],params[5],params[6],params[3]))
            combined.append(gaussian.Gausfit(df1).double_gaussian(a,params[0],params[1],params[2],params[3],params[4],params[5],params[6])) 
        elif len(params) == 11:
            gaus1.append(gaussian.Gausfit(df1).gaussian(a,params[0],params[1],params[2],params[3]))
            gaus2.append(gaussian.Gausfit(df1).gaussian(a,params[4],params[5],params[6],params[3]))
            gaus3.append(gaussian.Gausfit(df1).gaussian(a,params[7],params[8],params[9],params[3]))
            combined.append(gaussian.Gausfit(df1).tripple_gaussian(a,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]))
        else:
            gaus1.append(gaussian.Gausfit(df1).gaussian(a,params[0],params[1],params[2],params[3]))
            gaus2.append(0)
            gaus3.append(0)
            combined.append(0)
    A_gaus_right.append(params[4])
    A_gaus_left.append(params[0])
    mu_left.append(params[1])
    mu_right.append(params[5])
    plt.plot(x,y, label=f"spectra = {columns[dataset]}", color="royalblue", linewidth=1.8)
    plt.plot(x,gaus1, label=f"Gaus_left, A={round(params[0],4)} mu={round(params[1], 2)}, sig={round(params[2],2)}", linestyle="--", color="red")
    if len(params) >= 6:
        plt.plot(x,gaus2, label=f"Gaus_right, A={round(params[4],4)} mu={round(params[5], 2)}, sig={round(params[6],2)}", linestyle="--", color="red")
    if len(params) >= 10:
        plt.plot(x,gaus3, label=f"Gaus_shoulder, A={round(params[7],4)} mu={round(params[8], 2)}, sig={round(params[9],2)}", linestyle="--", color="red")
    plt.plot(x,combined, label="Gaus_left + Gaus_right", color="darkorange")
    plt.grid()
    plt.xlabel("wavenumber cm^-1")
    plt.ylabel("a.u.")
    plt.legend()
    #plt.show()
    plt.savefig(f"C:\\Users\\Igor\\Desktop\\Arbeit\\Borosilicat\\__fits__\\Gaussian\\double_{columns[dataset]}.png")
    plt.close()



def exponentiall(x,t, x0):
    y = np.exp((x-x0)*-t)
    return y

def exponentiall_func(x,t,x0):
    return np.exp((x-x0)*-t)


col = [1, 2 , 3, 4, 5, 6, 7, 8]
x_array = [4.5,7,9.5,12,14.5,17,19.5,22]
igp.double_barplot(col,A_gaus_left,A_gaus_right,"left peak", "right peak", "spectrum","area","area",0,0.11,0,0.11,"Peak comparison",col_name)
#igp.double_barplot(col,mu_left,mu_right,"left peak", "right peak", "spectrum","position","position",750,810,750,810,"Peak comparison",col_name)

comparison, i = [], 0

for a in A_gaus_right:
    comparison.append(A_gaus_right[i]/A_gaus_left[i])
    i += 1

x_array, comparison,fit= np.array(x_array), np.array(comparison), []
covar = []
exp_p, something = optimize.curve_fit(exponentiall,x_array,comparison)
print(exp_p)
x = np.linspace(4.5,22,1000)

for a in col_name:
    covar.append(exponentiall_func(a,exp_p[0], exp_p[1]))

for a in x:
    fit.append(exponentiall_func(a,exp_p[0], exp_p[1]))

coeff, coeff1 = np.corrcoef(covar,comparison)
R2_comb = round(coeff[1]**2, 4)

plt.plot(col_name,comparison, ".", color="blue", label="spectra", markersize=10)
plt.plot(x,fit, color="darkorange", label=f"x0 = {round(exp_p[1],2)} +- {round(exp_p[1]*(1-R2_comb),2)}, R2 = {round(R2_comb,4)}")
#plt.xticks(ticks=col,labels=col_name)
plt.grid()
plt.ylabel("area_right / area left [a.u.]")
plt.xlabel("mol % Na")
plt.legend()
plt.show()

