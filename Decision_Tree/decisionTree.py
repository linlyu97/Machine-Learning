import csv
import sys
import numpy as np
import copy

class Node():
    def __init__(self,value=None,dataSet=[]):
        self.val=value
        self.dataSet=dataSet
        self.left=None
        self.right=None
        self.attribute=None
        self.depth=0
        self.dataSet=[]
        self.labelCount={}
        self.predict_label=[]

    def printTree(self,initial_labelCount):
        
        str0='['
        # print('initial_labelCount=',initial_labelCount)
        for key in initial_labelCount:
            if key not in self.labelCount:
                self.labelCount[key]=0
            str0=str0+str(self.labelCount[key])+' '+str(key)+' /'
            # print('str0=',str0)
        str1=str0[:-2]+']'
        # print('str1=',str1)
        str2=''
        # if self:
        if self.depth == 0:
            print(str1)
        else:
            start2='|'
            for i in range(self.depth):
                str2+=start2
            str3=str2+str(self.val)+' '+'='+' '+str(self.attribute)+':'+' '
            print(str3+str1)
        
        if self.left:
            self.left.printTree(initial_labelCount)
        if self.right:
            self.right.printTree(initial_labelCount)
        # else:
        #     return

def dataInput(train_input):
    i = 0
    data_set = []
    labels = []
    attributeName=[]
    with open(train_input, 'r') as finput:
        data = csv.reader(finput, delimiter="\t")
        for line in data:
            data_set.append(line[0:])
        attributeName=data_set[0]
        attributeName.pop(-1)
        # print(attributeName)
        data_set.remove(data_set[0])  
        #print(data_set)           
        return data_set, attributeName
# print(dataInput('small_train.tsv'))
# dataInput('small_train.tsv')


def giniImpurity(data_set):
    type1=[]
    type2=[]
    type1=data_set[0][-1]
    # print('type=',type1)
    # print(data_set)
    for index in range(len(data_set)):
        if data_set[index][-1]!=data_set[0][-1]:
            type2=data_set[index][-1]
    # print('type1:', type1)
    # print('type2:', type2)
    count=0
    ratio=0
    for i in range(len(data_set)):
        if type1 == data_set[i][-1]:
            count+=1
    ratio=count/len(data_set)
    initialGI=ratio*(1-ratio)*2
    # print('gini',initialGI)
    return initialGI
# giniImpurity('small_train.tsv') 

def splitDataset(data_set,col,value):
    # data_set=dataInput(data_set)
    # headName=head[col]
    subDataSet=[]
    for i in range(len(data_set)):
        subRow=[]
        if data_set[i][col]==value:
            subRow.extend(data_set[i][:col])
            # print(subRow)
            subRow.extend(data_set[i][col+1:])
            # print(subRow)
            subDataSet.append(subRow)
            # print(subDataSet)     
    return subDataSet
# y='y'
# print(splitDataset('small_train.tsv',1,'n'))

# ‘’‘
def bestAttribute(data_set):
    # data_set,attributeNameList=dataInput(data_set)
    attributeNum=len(data_set[0])-1
    # print('attributeNum',attributeNum)
    initialGI=giniImpurity(data_set)
    # print('initialGI:', initialGI)
    bestGG=0
    bestAttribute=0
    for i in range(attributeNum):
        # print('i in 1 for:',i)
        valueList=[]
        for j in range(len(data_set)):
            # valueList=set(data_set[j][i])
            if data_set[j][i] not in valueList:
                valueList.append(data_set[j][i])
        # print("temp: ")
        # print(valueList)
        
        newGI=0
        GG=0
        # print('valueList',valueList)
        for value in valueList:
            subDataSet=splitDataset(data_set,i,value)
            p=len(subDataSet)/len(data_set)
            # print("type: ")
            # print(type(subDataSet))
            GI=giniImpurity(subDataSet)
            # print('GI in for',GI)
            newGI+=p*GI
        GG=initialGI-newGI
        # print('GG:', GG)
        # print('i:', bestAttribute)
        if GG>bestGG:
            bestGG=GG
            # print('i in final:', i)
            bestAttribute=i
            # print('ibest:', bestAttribute)
    return bestAttribute
# print(bestAttribute('small_train.tsv'))
# '''
def labelCount(data_set):
    # data_set,attributeName=dataInput(data_set)
    d={}
    for row in data_set:
        currentLabel=row[-1]
        if currentLabel not in d.keys():
            # print(row[-1])
            d[currentLabel]=0
        d[currentLabel]+=1

    return d
# print(labelCount('small_train.tsv'))

def createTree(node,data_set,depth,max_depth,head):
    if data_set==[]:
        return node

    labelNum={}
    labelNum=labelCount(data_set)
    node.labelCount=labelNum

    print('labelNum=',labelNum)
    value=0
    max_key=[]

    a=[]
    b=[]
    for key in labelNum:
        a.append(key)
        b.append(labelNum[key])
    a.sort()
    if len(a)==2 and b[0]==b[1]:
        max_key=a[1]
    else:
        for key in labelNum:
            if labelNum[key]>value:
                value=labelNum[key]
                max_key=key
    node.predict_label=max_key
    
    # print('max_key',max_key)
    # print('dataset=',data_set)
    # print('labeldict',labelNum)
    # print('len(labelNum)=',len(labelNum))

    colCount=len(data_set[0])

    if colCount==1 or len(labelNum) ==1 or (depth > max_depth) or depth == (max_depth):
        return node
    else:
        leftChild=Node()
        rightChild=Node()

        bestIndex=bestAttribute(data_set)
        
        valueList=[]
        for i in range(len(data_set)):
            if data_set[i][bestIndex] not in valueList:
                valueList.append(data_set[i][bestIndex])
        leftChild.attribute=valueList[0]
        subDataSet1=splitDataset(data_set,bestIndex,valueList[0])
        # print('subDataSet1=',subDataSet1)
        if len(valueList)==2:
            subDataSet2=splitDataset(data_set,bestIndex,valueList[1])
            rightChild.attribute=valueList[1]
            # print('subDataSet2=',subDataSet2)
        else:
            subDataSet2=[]
        # print('leftA',leftChild.attribute)
        # print('leftB',rightChild.attribute)
        leftChild.val=head[bestIndex]
        rightChild.val=head[bestIndex]
        # print('leftname',leftChild.val)
        # print('rightname',rightChild.val)
        head=np.delete(head,bestIndex)

        leftChild.dataSet=subDataSet1
        rightChild.dataSet=subDataSet2

        leftChild.depth=depth+1
        rightChild.depth=depth+1
        # print('leftdepyth=',leftChild.depth)
        # print('rightdepyth=',rightChild.depth)

        node.left=leftChild
        node.right=rightChild
        # print('1',subDataSet1)
        # print('2',subDataSet2)
        # times+=1
        # print('---------')
        createTree(node.left,subDataSet1,node.left.depth,max_depth,head)
        createTree(node.right,subDataSet2,node.right.depth,max_depth,head)
    return node


# data,headData=dataInput('small_train.tsv')
# headData1=np.asarray(headData)
# print(createTree(Node(None,data),data,0,3,headData1))

def testTree(tree,test_dataset,test_head):
    
    attribute_name=[]
    data_index=0
    label=[]
    error_num=0
    for row in test_dataset:
        node=tree
        while node.left is not None and node.right is not None:
            # attribute_name=node.left.val
            # data_index=test_head.index(attribute_name)
            data_index = np.where(test_head==node.left.val)
            # print(data_index)
            if node.left.attribute==row[data_index]:
                # print('node.left.attribute=',node.left.attribute)
                node=node.left
            elif node.right.attribute==row[data_index]:
                node=node.right
        key=node.predict_label
        # print(key)
        label.append(key)
    
        if key != row[-1]:
            error_num+=1
    error_rate=error_num/len(test_dataset)
    # print(label)

    return label, error_rate

data1,headData1=dataInput('politicians_train.tsv')
data2,headData2=dataInput('politicians_test.tsv')

headData01=np.asarray(headData1)
data01=np.asarray(data1)
headData02=np.asarray(headData2)
data02=np.asarray(data2)

node=createTree(Node(None,data1),data1,0,3,headData01)
node.printTree(node.labelCount)

label1, error_rate1=testTree(node,data01,headData01)
label2, error_rate2=testTree(node,data02,headData02)

#print('error_rate1:',error_rate1)
#print('error_rate2:',error_rate2)



if __name__ == '__main__':

    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6] 

    data1,headData1=dataInput(train_input)
    data2,headData2=dataInput(test_input)


    headData01=np.asarray(headData1)
    data01=np.asarray(data1)
    headData02=np.asarray(headData2)
    data02=np.asarray(data2)

    node=createTree(Node(None,data1),data1,0,max_depth,headData01)
    node.printTree(node.labelCount)
    label1, error_rate1=testTree(node,data01,headData01)
    label2, error_rate2=testTree(node,data02,headData02)


    with open(metrics_out, 'w') as f:
        f.write('error(train): '+ str(error_rate1) + '\n')
        f.write('error(test): ' + str(error_rate2))

    with open(train_out, 'w') as f:
        for item in label1:
            f.write(str(item) + '\n')

    with open(test_out, 'w') as f:
        for item in label2:
            f.write(str(item) + '\n')


   






