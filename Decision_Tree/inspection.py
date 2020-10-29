import csv
import sys
import numpy as np

def dataInput(file_input):
    i = 0
    data_set = []
    labels = []
    with open(file_input, 'r') as finput:
        data = csv.reader(finput, delimiter="\t")
        for line in data:
            data_set.append(line[0:])
        data_set.remove(data_set[0])  
        # print(data_set)           
        return data_set
# dataInput('politicians_train.tsv')
# '''
def checkType(file_input):
    data_set=dataInput(file_input)
    type1=[]
    type2=[]
    type1=data_set[0][-1]
    for index in range(len(data_set)):
        if data_set[index][-1]!=data_set[0][-1]:
            type2=data_set[index][-1]
    # print('type1', type1)
    # print(type2)
    return type1,type2

def gini(file_input):
    data_set=dataInput(file_input)
    type1,type2=checkType(file_input)
    count=0
    ratio=0
    gini=0
    error=0
    for i in range(len(data_set)):
        if data_set[i][-1]==type1:
            count+=1
    ratio=count/len(data_set)
    gini=(ratio*(1-ratio))*2
    if ratio < 0.5:
        error=ratio
    else:
        error=1-ratio

    return gini,error

if __name__ == '__main__':

    file_input = sys.argv[1]
    file_output = sys.argv[2]
     
    gini_impurity,error_rate=gini(file_input)
    with open(file_output, 'w') as f:
        f.write('gini_impurity: '+ str(gini_impurity) + '\n')
        f.write('error: ' + str(error_rate))
#   '''
 






