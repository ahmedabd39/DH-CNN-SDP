
import pandas as pd 
import os
import re
import warnings
import javalang
warnings.filterwarnings('ignore')
# here you have change your csv file for the dataset
data = pd.read_csv("new_ant_1.7.csv")
# data = pd.read_csv("our_dataset/xalan-2.7.csv") 
code=" "
file_paths=data['name']
# file_paths=file_paths.str.replace(".", "/")

file_token=[]
# files_word=[]
j=1
for i in range(0,len(file_paths)):
	#file_name="our_dataset/jedit4.3.3source/jEdit/"+file_paths[i]+".java"
	file_name = file_paths[i] + ".java"
	print(file_name)
	fd = open(file_name, "r")
	words=''
	treenode = javalang.parse.parse(fd.read())
	for path,node in treenode:
		if isinstance(node, javalang.tree.ClassDeclaration):
			words=words+' '+str(node.name)
		if isinstance(node, javalang.tree.MethodDeclaration):
			words=words+' '+str(node.name)
			# print(words)
		elif isinstance(node, javalang.tree.ForStatement):
			words=words+' '+"for"
		elif isinstance(node, javalang.tree.WhileStatement):
			words=words+' '+"while"
		elif isinstance(node, javalang.tree.DoStatement):
			words=words+' '+"do"
		elif isinstance(node, javalang.tree.IfStatement):
			words=words+' '+"if"
		elif isinstance(node, javalang.tree.SwitchStatementCase):
			words=words+' '+"switch"
	
	file_token.append(words)

df = pd.DataFrame(file_token)
data['nodes']=file_token
# df.to_csv("declaration2.csv",index = False)
# data.to_csv("with_nods.csv",index = False)
data.to_csv("ant_1.7_with_nods.csv",index = False)
print("saved")
# for j in range(1,len(file_token)):
	# print(file_token[j])
	