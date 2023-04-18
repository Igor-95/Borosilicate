import pandas as pd
from pandas import ExcelWriter
import numpy as np
import matplotlib as plt
import sys
import os 
import glob


import center_of_gravity
import lognorm
import bihill
import gaussian
import utils

"""
Defining where the data is and where to put the results
"""
working_directory = utils.get_cwd()
data_location = utils.get_cwd() + "\__data__"
files = utils.get_all_files(data_location)
relevant_files = []
Error_message = "Press Enter to terminate"


"""
First extract only .txt files from __data__ location and put them togther
"""
for a in files:
    if ".txt" in a:
        relevant_files.append(a)
    else:
        continue

del files

""" 
creating "excel files" to stroe data in
"""
center_of_gravity_data = pd.DataFrame(columns=["spectra","center_of_gravity"])
double_gaus_data = pd.DataFrame(columns=["spectra","Area_left","Area_right","ratio_A","mu_left","mu_right","sig_left","sig_right","y0"])
lognormal_data = pd.DataFrame(columns=["spectra","peak_position", "peak_height", "Fit_A","Fit_mu","Fit_sig","Fit_y0"])
bihill_data = pd.DataFrame(columns=["spectra","peak_position", "peak_height"])
baseline_corr_normalized_spec = pd.DataFrame()

list_of_df = [baseline_corr_normalized_spec, bihill_data, lognormal_data, center_of_gravity_data, double_gaus_data]
list_names_df = ["baseline_corr_spectra", "bihill","lognormal", "center_of_gravity","double_gaus"]

""" 
read ini.txt
"""
ini = pd.read_csv("ini.txt", delim_whitespace=True) # reads ini.txt file 
i, y_col_number = 0, 0 

""" 
cycle trough every txt file in __data__
"""
for data in relevant_files:
    df = pd.read_csv(data, delim_whitespace=True, decimal=".")
    if df.iloc[:,0].dtypes != "float64":
        print("Data corrupted -> string in columns")
        input("Press Enter to terminate")
        break
    
    "Sorting, baselinecorrection, normalizing"
    columns = df.columns
    sort_colum_name = columns[0]
    df = df.sort_values(sort_colum_name,axis=0, ascending=True)
    df = utils.normalize_bc_spectra(df)
    
    "determining dataset name" 
    position_name = len(data_location)
    data_set_name = data[position_name+1:position_name+9]
    
    "Filling nomralized, baselinecorrected spectra to dataframe"
    baseline_corr_normalized_spec.loc[:, data_set_name+"_x"] = df.iloc[:,0]
    baseline_corr_normalized_spec.loc[:,data_set_name+"_y"] = df.iloc[:,1]
    
    "Boson Peak in Lognormal / Bihill Fit parameters and peak position"
    procedure_boson = ini.loc[0, "value"]
    
    if procedure_boson == 0:
        try:
            peak_bihill = bihill.bihill_peak(df)
            bihill_data.loc[i,"spectra"] = data_set_name
            bihill_data.loc[i,"peak_position"] = peak_bihill[0]
            bihill_data.loc[i,"peak_height"] = peak_bihill[1]
            
        except RuntimeError:
            print(f"{data_set_name} cannot be fitted by BiHill function, try new bounds")
            bihill_data.loc[i,"spectra"] = data_set_name
            bihill_data.loc[i,"peak_position"] = 0
            bihill_data.loc[i,"peak_height"] = 0
            
    elif procedure_boson == 1:
        peak_lognorm = lognorm.lognorm_peak(df)
        params_lognorm = lognorm.lognorm_fit(df)

        lognormal_data.loc[i,"spectra"] = data_set_name
        lognormal_data.loc[i,"peak_position"] = peak_lognorm[0]
        lognormal_data.loc[i,"peak_height"] = peak_lognorm[1]
        lognormal_data.loc[i,"Fit_A"] = params_lognorm[0]
        lognormal_data.loc[i,"Fit_mu"] = params_lognorm[1]
        lognormal_data.loc[i,"Fit_sig"] = params_lognorm[2]
        lognormal_data.loc[i,"Fit_y0"] = params_lognorm[3]
        
    else:
        try:
            peak_bihill = bihill.bihill_peak(df)
            bihill_data.loc[i,"spectra"] = data_set_name
            bihill_data.loc[i,"peak_position"] = peak_bihill[0]
            bihill_data.loc[i,"peak_height"] = peak_bihill[1]
            
        except RuntimeError:
            print(f"{data_set_name} cannot be fitted by BiHill function, try new bounds")
            bihill_data.loc[i,"spectra"] = data_set_name
            bihill_data.loc[i,"peak_position"] = 0
            bihill_data.loc[i,"peak_height"] = 0

        peak_lognorm = lognorm.lognorm_peak(df)
        params_lognorm = lognorm.lognorm_fit(df)

        lognormal_data.loc[i,"spectra"] = data_set_name
        lognormal_data.loc[i,"peak_position"] = peak_lognorm[0]
        lognormal_data.loc[i,"peak_height"] = peak_lognorm[1]
        lognormal_data.loc[i,"Fit_A"] = params_lognorm[0]
        lognormal_data.loc[i,"Fit_mu"] = params_lognorm[1]
        lognormal_data.loc[i,"Fit_sig"] = params_lognorm[2]
        lognormal_data.loc[i,"Fit_y0"] = params_lognorm[3]


    " Center of gravity"

    cg_params = center_of_gravity.Cog(df).center_of_gravity()
    center_of_gravity_data.loc[i,"spectra"] = data_set_name
    center_of_gravity_data.loc[i,"center_of_gravity"] = cg_params[0]
    center_of_gravity_data.loc[i,"height_of_cog"] = cg_params[1]    

    "B3 / B4 peaks"

    gaussian_params = gaussian.Gausfit(df).gaussian_fit()
    ratiob3_b4 = gaussian_params[4]/gaussian_params[0]

    if len(gaussian_params) == 8:
        double_gaus_data.loc[i,"spectra"] = data_set_name
        double_gaus_data.loc[i,"Area_left"] = gaussian_params[0]
        double_gaus_data.loc[i,"Area_right"] = gaussian_params[4]
        double_gaus_data.loc[i,"ratio_A"] = ratiob3_b4
        double_gaus_data.loc[i,"mu_left"] = gaussian_params[1]
        double_gaus_data.loc[i,"mu_right"] = gaussian_params[5]
        double_gaus_data.loc[i,"sig_left"] = gaussian_params[2]
        double_gaus_data.loc[i,"sig_right"] = gaussian_params[6]
        double_gaus_data.loc[i,"y0"] = gaussian_params[3]
        double_gaus_data.loc[i,"A_3"] = 0
        double_gaus_data.loc[i,"mu3"] = 0
        double_gaus_data.loc[i,"sig3"] = 0
    
    else:
        double_gaus_data.loc[i,"spectra"] = data_set_name
        double_gaus_data.loc[i,"Area_left"] = gaussian_params[0]
        double_gaus_data.loc[i,"Area_right"] = gaussian_params[4]
        double_gaus_data.loc[i,"ratio_A"] = ratiob3_b4
        double_gaus_data.loc[i,"mu_left"] = gaussian_params[1]
        double_gaus_data.loc[i,"mu_right"] = gaussian_params[5]
        double_gaus_data.loc[i,"sig_left"] = gaussian_params[2]
        double_gaus_data.loc[i,"sig_right"] = gaussian_params[6]
        double_gaus_data.loc[i,"y0"] = gaussian_params[3]
        double_gaus_data.loc[i,"A_3"] = gaussian_params[7]
        double_gaus_data.loc[i,"mu3"] = gaussian_params[8]
        double_gaus_data.loc[i,"sig3"] = gaussian_params[9]

    i += 1
    y_col_number += 2

"Save data to excel"
j= 0
for df_name in list_of_df:
    if len(df_name) == 0:
        j += 1
        continue
    else:
        excel_name = f"{list_names_df[j]}.xlsx"
        df_name.to_excel(excel_name, index=False)
        j += 1
