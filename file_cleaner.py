import pandas as pd 
import os
import re
import warnings
import javalang
warnings.filterwarnings('ignore')
data = pd.read_csv("ant_1.7_with_nods.csv")
code=" "
file_paths=data['name']
j=1
for i in range(0,len(file_paths)):
	file_name=file_paths[i]+".java"
	#print(file_name)
	if j!=0:
		fd = open(file_name, "r")
		lines=fd.readlines()
		comment_tags = ["/*", "*", "//","*"]
		clean = []
		for line in lines:
			line = line.strip()
			line=str(line)
			#print(line)
			if len(line)==0:
				continue
			elif any(line.startswith(p) for p in comment_tags):
				continue
			else :
				clean.append(line)	
		j=1
		result = '\n'.join(map(str, clean))
		#print(clean)
		newfile="cleaned_dataset/"+file_name
		with open(newfile,"w") as fp:
			fp.writelines(result)
