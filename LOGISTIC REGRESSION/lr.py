import numpy as np
import sys
import csv

def file_input(input):
    labels = []
    feature_index=[]
    file = open(input, "r")
    lines = file.readlines()
    # print(lines)
    file.close()
    for line in lines:
        index=[]
        labels.append(int(line.split('\t')[0]))
        example=line.split('\t')
        example.pop(0)
        for item in example:
            index.append(int(item.split(':')[0]))
        index.append(39176)
        # print('index=',index)
        feature_index.append(index)
    # print('all=',feature_index)
    # print(labels)
    return labels, feature_index

def new_data(input):
    labels,feature_index=file_input(input)
    new_list=[]
    # i=0
    for row in feature_index:
        # i+=1
        d={}
        for index in row:
            d[index]=1
        new_list.append(d)
        # print('new_list=',new_list)
        # if i==2:
        #     break
    return new_list

# new_data('model1_formatted_train.tsv')
def sparse_dot_product(d,w):
    pro=0.0
    for key,value in d.items():
        # print("key, value :", key, value)
        # print(w)
        pro+=w[key]
    return pro


def sgd(input,rate,epoch):
    labels,feature_index=file_input(input)
    new_list=new_list=new_data(input)
    theta=np.zeros(39177)
    
    # j=1
    # if j<=epoch:
    #     j+=1
    for times in range(epoch):
        i=-1
        for d in new_list:
            # print('row=',row)
            i+=1
            xi=np.zeros(39177)
            for index in d:
                xi[index]=1
            # print(xi)
            product=sparse_dot_product(d,theta)
            # print('product=',product)
            theta+=rate*xi*(labels[i]-(np.exp(product)/(1+np.exp(product))))
            # break
    # print('theta=',theta)
        # break
    return theta

def predict(input,rate,epoch,file_out,theta):
    labels,feature_index=file_input(input)
    new_list=new_list=new_data(input)
    # theta=sgd(input,rate,epoch)
    num=0
    i=-1
    file = open(file_out, 'w')
    for d in new_list:
        i+=1
        product=sparse_dot_product(d,theta)
        prob=(np.exp(product))/(1+np.exp(product))
        if prob>0.5:
            predict=1
        else:
            predict=0
        file.write(str(predict)+'\n')
        if predict!=labels[i]:
            num+=1
    error=num/len(new_list)
    file.close()
    print(error)
    return error
theta=sgd('formatted_train.tsv',0.1,30)
predict('formatted_train.tsv',0.1,30,'1.labels',theta)


# file_input('model1_formatted_train.tsv')


if __name__ == '__main__':

    formatted_train_input = sys.argv[1]
    formatted_validation_input = sys.argv[2]
    formatted_test_input = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = int(sys.argv[8])
    theta=sgd(formatted_train_input,0.1,num_epoch)
    train_error=predict(formatted_train_input,0.1,num_epoch,train_out,theta)
    test_error=predict(formatted_test_input,0.1,num_epoch,test_out,theta)
    
    with open(metrics_out, 'w') as f:
        f.write('error(train): '+ str(train_error) + '\n')
        f.write('error(test): ' + str(test_error))

    

    
    
