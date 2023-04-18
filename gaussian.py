import numpy as np
from scipy import optimize
import pandas as pd


class Gausfit:

    def __init__(self,data) -> None:
        self.df = data 
        self.ini = pd.read_csv("ini.txt", delim_whitespace=True)

    def gaussian(self,x, A, mu, sig, y0):
        
        return (A/(sig*np.sqrt(np.pi/2)))*np.exp(-2*((x-mu)**2/(sig**2))) + y0
    
    def double_gaussian(self,x, A1, mu1, sig1, y0, A2, mu2, sig2):
        
        return (A1/(sig1*np.sqrt(np.pi/2)))*np.exp(-2*((x-mu1)**2/(sig1**2))) + y0 + (A2/(sig2*np.sqrt(np.pi/2)))*np.exp(-2*((x-mu2)**2/(sig2**2)))
    
    def tripple_gaussian(self,x, A1, mu1, sig1, y0, A2, mu2, sig2, A3, mu3, sig3):
        
        return (A1/(sig1*np.sqrt(np.pi/2)))*np.exp(-2*((x-mu1)**2/(sig1**2))) + y0 + (A2/(sig2*np.sqrt(np.pi/2)))*np.exp(-2*((x-mu2)**2/(sig2**2))) + (A3/(sig3*np.sqrt(np.pi/2)))*np.exp(-2*((x-mu3)**2/(sig3**2)))
    
    def filter_data(self):
        xmin, xmax = self.ini.loc[9, "value"], self.ini.loc[10, "value"]
        data = self.df[self.df.iloc[:,0] >= xmin]
        data = data[data.iloc[:,0] <= xmax]
        x,y = data.iloc[:,0], data.iloc[:,1]
        x,y, = np.array(x), np.array(y)
        return x,y

    def gaussian_fit(self):
        #parameters read from ini.txt
        mu1, sig1 = self.ini.loc[12, "value"], self.ini.loc[13, "value"]
        mu2, sig2 = self.ini.loc[14, "value"], self.ini.loc[15, "value"]
        mu3, sig3 = self.ini.loc[16, "value"], self.ini.loc[17, "value"]
        # data parsed as two dimensional pd dataframe, filtered for left_gaus and right_gaus
        x,y, =self.filter_data()
        # constrains on optimization function A, mu, sig, y0, A .....
        c_double_gaus = ([0,766,0,0,0,800,0],[1,775.5,40,0.5,1,815,40])
        c_double_gaus_shoulder = ([0,766,0,0,0,650,0],[1,776,60,0.5,1,725,50])
        c_tripple_gaus = ([0,766,0,0,0,800,0,0,650,0],[1,775.5,40,0.5,1,815,40,1,720,50])
        # optimizing the function to data
        params, something = optimize.curve_fit(self.double_gaussian,x,y,p0=[0.02,mu1,sig1,0.0005,0.04,mu2,sig2],bounds=c_double_gaus)
        params_2, something = optimize.curve_fit(self.double_gaussian,x,y,p0=[0.04,mu1,sig1,0.0005,0.02,mu3,sig3],bounds=c_double_gaus_shoulder)
        p_t, something = optimize.curve_fit(self.tripple_gaussian,x,y,p0=[0.02,mu1,sig1,0.0005,0.04,mu2,sig2,0.005,mu3,sig3],bounds=c_tripple_gaus)
        combined, combined2, tripple,= [],[],[]
        # determining fit quality by calculating correlation coefficient
        for a in x:
            combined.append(self.double_gaussian(a,params[0],params[1],params[2],params[3],params[4],params[5],params[6]))
            combined2.append(self.double_gaussian(a,params_2[0],params_2[1],params_2[2],params_2[3],params_2[4],params_2[5],params_2[6]))
            tripple.append(self.tripple_gaussian(a,p_t[0],p_t[1],p_t[2],p_t[3],p_t[4],p_t[5],p_t[6],p_t[7],p_t[8],p_t[9]))

        params, params_2,p_t = list(params), list(params_2),list(p_t)
        coeff ,coeff1= np.corrcoef(y,combined)
        R2_comb = round(coeff[1]**2, 4)
        params.append(R2_comb)
        coeff ,coeff1= np.corrcoef(y,combined2)
        R2_comb2 = round(coeff[1]**2, 4)
        params_2.append(R2_comb2)
        coeff ,coeff1= np.corrcoef(y,tripple)
        R2_tripple = round(coeff[1]**2, 4)
        p_t.append(R2_tripple)

        #print(f"R2_comb = {R2_comb}", f"R2_comb2 = {R2_comb2}",f"R2_tripple = {R2_tripple}")
        
        if R2_comb >= R2_tripple:
            if R2_comb >= R2_comb2:
                return params
            else:
                return params_2
        elif R2_comb2 >= R2_tripple:
            return p_t
        else:
            return p_t

       
    def gaussian_least_squares(self):
        xmin, xmax = self.ini.loc[9, "value"], self.ini.loc[10, "value"]
        mu1, sig1 = self.ini.loc[12, "value"], self.ini.loc[13, "value"]
        mu2, sig2 = self.ini.loc[14, "value"], self.ini.loc[15, "value"]
        mu3, sig3 = self.ini.loc[16, "value"], self.ini.loc[17, "value"]

