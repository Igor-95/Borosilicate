import numpy as np
import pandas as pd
from scipy import optimize


class Cog:

    # data must be pandas data frame with x and y values
    def __init__(self,data) -> None:
        
        result = []
        fit_params = []
        self.df = data 
        self.ini = pd.read_csv("ini.txt", delim_whitespace=True)
    
    # needs lower and upper bound for wavenumber default 250 & 400
    def baseline_points(self,w_min=250,w_max=400, parameters=False, primitive=False):
        procedure = self.ini.loc[3, "value"]
        df = self.df[self.df.iloc[:,0] >= w_min]
        df = df[df.iloc[:,0] <= w_max]
        xmin_simple, ymin_simple = self.min_x_y(df)
        results = []

        if procedure == 1:
            results.append(xmin_simple)
            results.append(ymin_simple)
            return results

        elif procedure == 0:
            x,y = list(df.iloc[:,0]), list(df.iloc[:,1])
            params = np.polyfit(x,y,2)
            xmin = (-1*params[1])/(2*params[0]) # -> 1. derivative solved for x
            ymin = params[0]*xmin**2 + params[1]*xmin + params[2]

            # check if fit makes sense
            if params[0] <= 0:
                results.append(xmin_simple)
                results.append(ymin_simple)

            elif xmin < w_min:
                results.append(xmin_simple)
                results.append(ymin_simple)

            elif xmin > w_max:         
                results.append(xmin_simple)
                results.append(ymin_simple)

            else:
                if primitive == False:
                    results.append(xmin)
                    results.append(ymin)
                else:
                    results.append(xmin_simple)
                    results.append(ymin_simple)

            if parameters is False:
                return results # -> returns x_min and corresponding y value
            else:
                return params # -> returns fit parameter
        
    def baseline_corr(self, parameters=False):
        #read ini values left_low - right_high
        left_low, left_high = self.ini.loc[4,"value"], self.ini.loc[5,"value"]
        right_low, right_high = self.ini.loc[6,"value"], self.ini.loc[7,"value"]
        x_left, y_left = self.baseline_points(w_min=left_low, w_max=left_high)
        x_right, y_right = self.baseline_points(w_min=right_low, w_max=right_high)
        m, b = self.linear_function(x_left, y_left, x_right, y_right)

        if parameters == True:
            return m, b, left_low, left_high, right_low, right_high
        
        else:
            df = self.df[self.df.iloc[:,0] >= x_left]
            df = df[df.iloc[:,0] <= x_right]
            x,y,z = df.iloc[:,0], df.iloc[:,1], []

            for a in x:
                z.append(m*a+b)
            y = np.array(y)-z
            df_temp = pd.DataFrame(columns=["x", "y"])
            df_temp.loc[:,"x"] = x
            df_temp.loc[:,"y"] = y
            return df_temp
    
    def center_of_gravity(self, plot=False):
        df = self.baseline_corr()
        x,y = list(df.iloc[:,0]), list(df.iloc[:,1])
        i = 0 
        area = 0 
        sum_area = []
        for a in x:
            if i < len(x)-1:
                area += self.trapezoid(x[i],x[i+1],y[i], y[i+1])
                sum_area.append(area)
                i += 1
            elif i < len(x):
                sum_area.append(area)
                i += 1
            else:
                break
        sum_area = np.array(sum_area)/area

        if plot == True:
            return x, sum_area
        else:
            df1 = pd.DataFrame(columns=["x", "area"])
            df1.loc[:,"x"] = x
            df1.loc[:,"area"] = abs(sum_area - 0.5)
            df1 = df1.sort_values("area",axis=0, ascending=True)
            return df1.iloc[0,0], df1.iloc[0,1]
                



    def min_x_y(self, data):
        columns = data.columns
        df = data.sort_values(columns[1],axis=0,ascending=True)
        x,y = df.iloc[0,0], df.iloc[0,1]
        return [x, y] 
    
    def linear_function(self, x1,y1,x2,y2):
        m = (y2-y1)/(x2-x1)
        b = y1 - x1*m
        return m , b

    def trapezoid(self, x,x1,y,y1):
        area = (y1+y)*0.5*(x1-x)
        return area

    def testing(self):
        print(self.df)
        return self.df
        
