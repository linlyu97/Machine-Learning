import numpy as np
import sys
import csv

def dataInput(data):
    words=[]
    labels=[]  
    file = open(data, "r")
    lines = file.readlines()
    # print(type(lines))
    file.close()
    for line in lines:
        # print(type(line))
        labels.append(line[0])
        example=line.split('\t')
        # print(type(example))
        words.append(example[1].split(' '))
        # print('------------')
        # print(line)   
        # print(type(line))  
    # print(labels)
        # print(words)
        # break
    # print(type(words))
    return words,labels
dataInput('train_data.tsv')
def dictInput(dict_input):
    dict1={}
    file = open(dict_input,'r')
    for line in file.readlines():
        two=line.split(' ')
        # print(two)
        k = two[0]
        one=two[1]
        # print('one=',one)
        v = one[:-1]
        # print('v=',v)
        dict1[k] = v
    # print(dict1)
    # print(two)
    file.close()
    # print(dict1)
    return dict1

def check1(data,dict_input):
    words,labels=dataInput(data)
    dict1=dictInput(dict_input)
    dict_index=[]
    
    for row in words:
        index=[]
        for item in row:
            if item in dict1.keys() and dict1[item] not in index:
                index.append(dict1[item])
        dict_index.append(index)
    # print(dict_index)
    return dict_index

def output_file_model1(data,dict_input,file_out):
    words,labels=dataInput(data)
    dict_index=check1(data,dict_input)
    # print('len-labels=',len(labels))
    # print('len-words=',len(words))
    file = open(file_out, 'w')
    i=-1
    # str1=''
    for row in dict_index:
        i+=1
        str0=''
        for item in row:
            str0+=item+':'+'1'+'\t'
        str1=labels[i]+'\t'+str0[:-1]+'\n'
        file.write(str1)
    file.close()
    # print(str1)

def check2(data,dict_input,threshold):
    words,labels=dataInput(data)
    dict1=dictInput(dict_input)
    index=[]
    for row in words:
        d={}
        dict_index=[]
        for item in row:
            if item in dict1.keys():
                if item not in d.keys():
                    d[item]=1
                else:
                    d[item]+=1
    # print(d)
        for key in d:
            if d[key]<threshold:
                # print(dict1[key])
                dict_index.append(dict1[key])
        # # print(222)
        index.append(dict_index)
    # print(2222)
    # print(index)                
    return index

def output_file_model2(data,dict_input,file_out,threshold):
    words,labels=dataInput(data)
    dict_index=check2(data,dict_input,threshold)
    # print('len-labels=',len(labels))
    # print('len-words=',len(words))
    file = open(file_out, 'w')
    i=-1
    # str1=''
    for row in dict_index:
        i+=1
        str0=''
        for item in row:
            str0+=item+':'+'1'+'\t'
        str1=labels[i]+'\t'+str0[:-1]+'\n'
        file.write(str1)
    file.close()

# dictInput('dict.txt')
# dataInput('train_data.tsv')
# check2('train_data.tsv','dict.txt',4)
# output_file_model1('train_data.tsv','dict.txt','model1_train.tsv')

if __name__ == '__main__':

    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_validation_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag=int(sys.argv[8])
    if feature_flag==1:
        output_file_model1(train_input,dict_input,formatted_train_out)
        output_file_model1(validation_input,dict_input,formatted_validation_out)
        output_file_model1(test_input,dict_input,formatted_test_out)
    else:
        output_file_model2(train_input,dict_input,formatted_train_out,4)
        output_file_model2(validation_input,dict_input,formatted_validation_out,4)
        output_file_model2(test_input,dict_input,formatted_test_out,4)
