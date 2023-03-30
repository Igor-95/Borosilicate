import pandas as pd 


df = pd.read_csv("ini.txt", delim_whitespace=True)
#df.columns = df.iloc[0,:]
#df = df.drop(["Boson_peak"])
print(df)