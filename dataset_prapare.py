import pandas as pd 
import os
import re
import warnings
import javalang
warnings.filterwarnings('ignore')
# data = pd.read_csv("mereged_file.csv") 
data = pd.read_csv("ant_1.7.csv")
# replacing . with /
file_paths=data['name']
file_paths=file_paths.str.replace(".", "/")
data['name']=file_paths
# replacing the bug value with 0 and one
bug=data['bug']
#data.loc[(data['bug'] > 0)] = 1
data['bug'].loc[(data['bug'] > 0)] = 1
data.to_csv("new_ant_1.7.csv",index = False)
print("saved file with new_version_of_jedit-4.3.csv name")
