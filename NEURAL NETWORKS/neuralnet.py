import numpy as np
import sys
import csv

def dataInput(data):
    labels=[] 
    feature=[] 
    data_feature=[]
    with open(data, "r") as f:
        lines = f.readlines()
    print(lines)
    for line in lines:
        labels.append(int(line.split(',')[0]))
        print(labels)
        feature = [float(item) for item in line.split(',')[1:]]
        # feature.insert(0,1.0)
        print(feature)
        data_feature.append(feature)
        # print(data_feature)
    return labels,data_feature
    
dataInput('tinyTrain.csv')

# '''
def ini_para(ini_flag,hidden_units,n):
    if ini_flag==1:
        alpha_ini = np.random.uniform(-0.1,0.1,(hidden_units,n))
        beta_ini = np.random.uniform(-0.1,0.1,(10,hidden_units+1))
        alpha_ini[:, 0] = 0  
        beta_ini[:,0] = 0
    else:
        alpha_ini = np.zeros((hidden_units,n))        #   4 * 6
        beta_ini = np.zeros((10,hidden_units+1))
        # print(alpha_ini)
        # print(beta_ini)
    return alpha_ini,beta_ini

# ini_para(2,4)

def sigmoid(a):
    z=np.zeros((len(a),1))
    for i in range(len(a)):
        z[i]=1/(1+np.exp(-a[i]))
    z=np.insert(z,0,1,axis=0)
    return z

def softmax(b):
    sum=0
    for i in range(len(b)):
        sum+=np.exp(b[i])
    max_index=-1
    max=-1
    y_prob=np.zeros(10)
    for i in range(len(b)):
        y_prob[i]=np.exp(b[i])/sum
        if y_prob[i]>max:
            max=y_prob[i]
            max_index=i
    y_prob = y_prob[:, np.newaxis]
    # print('max_index=',max_index)
    # print('y_prob=',y_prob)
    return y_prob,max_index

def NNforward(labels,item,alpha_ini,beta_ini,j):
    x = np.array([item])
    x=np.transpose(x)
    # print(x)
    a = np.dot(alpha_ini,x)
    # print(len(a))
    # print('a=',a)
    z = sigmoid(a)
    # print('z=',z)
    b = np.dot(beta_ini,z)
    # print('b=',b)
    # print(j)
    y=labels[j]
    # print(y)
    y_star=np.zeros(10)
    y_star[int(y)]=1
    y_star = y_star[:, np.newaxis]
    # print(y_star)
    y_prob,max_index=softmax(b)
    # print(y_prob)
    return x,a,z,y_star,y_prob,max_index

def Backpropagation(y_star,y_prob,z,beta_ini,x):
    g_b=y_prob-y_star
    # print('g_b=',g_b)
    g_beta=np.dot(g_b, np.transpose(z))
    # print('g_beta=',g_beta)
    beta = np.delete(beta_ini,0,axis = 1)
    z = np.delete(z,0,axis=0)
    g_z=np.dot(np.transpose(beta), g_b)
    # print('g_z=',g_z)
    g_a = g_z * z * (1 - z)
    # print('g_a=',g_a)
    g_alpha = np.dot(g_a, np.transpose(x))
    # print('g_alpha=',g_alpha)
    return g_alpha,g_beta         

def cross_entropy(data_feature,labels,alpha_ini,beta_ini):
    sum=0
    j=-1
    for item in data_feature:
        j+=1
        x,a,z,y_star,y_prob,max_index=NNforward(labels,item,alpha_ini,beta_ini,j)
        y_star=np.ravel(y_star,'F')
        # print(y_star)
        y_prob=np.ravel(y_prob,'F')
        # print(y_prob)
        y_log=np.log(y_prob)
        # print(y_log)
        mul=-np.dot(y_star,y_log)
        # print(mul)
        sum+=mul
    cross_entropy=sum/(len(labels))
    return cross_entropy


def error(data_feature,labels,alpha_ini,beta_ini,file_out):
    j=-1
    error=0
    file = open(file_out, 'w')
    for item in data_feature:
        j+=1
        x,a,z,y_star,y_prob,max_index=NNforward(labels,item,alpha_ini,beta_ini,j)
        # print('max_index=',max_index)
        # print('y_prob=',y_prob)
        file.write(str(max_index)+'\n')
        if labels[j]!=max_index:
            error+=1
    error_rate=error/(len(labels))
    file.close()
    # print(error_rate)
    return error_rate

    
# cross_entropy,alpha_ini,beta_ini=SGD('tinyTest.csv',1,4,1,0.1)     
# error('tinyTest.csv',alpha_ini,beta_ini,'aaa.txt')       

if __name__ == '__main__':

    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units= int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])

    labels,data_feature=dataInput(train_input)
    labels1,data_feature1=dataInput(test_input)
    n=len(data_feature[0])
    print(n)
    alpha_ini,beta_ini=ini_para(init_flag,hidden_units,n)
    metrics_out = open(metrics_out,'wt',encoding='utf-8')
    sum=0
    for i in range(num_epoch):
        j=-1
        for item in data_feature:
            j+=1
            x,a,z,y_star,y_prob,max_index=NNforward(labels,item,alpha_ini,beta_ini,j)
            g_alpha,g_beta=Backpropagation(y_star,y_prob,z,beta_ini,x)
            beta_ini = beta_ini - learning_rate * g_beta
            alpha_ini = alpha_ini - learning_rate * g_alpha
            # print('final_beta=',beta_ini)
            # print('final_alpha=',alpha_ini)

        train_entropy=cross_entropy(data_feature,labels,alpha_ini,beta_ini)
        test_entropy=cross_entropy(data_feature1,labels1,alpha_ini,beta_ini)
        # print('----')
        # print('train_entropy=',train_entropy)
        # print('test_entropy=',test_entropy)
        metrics_out.write('epoch=' + str(i+1) + ' crossentropy(train): ' + str(train_entropy) + '\n')
        metrics_out.write('epoch=' + str(i+1) + ' crossentropy(test): ' + str(test_entropy) + '\n')
    train_error=error(data_feature,labels,alpha_ini,beta_ini,train_out)
    test_error=error(data_feature1,labels1,alpha_ini,beta_ini,test_out)
    metrics_out.write('error(train): ' + str(train_error) + '\n')
    metrics_out.write('error(test): ' + str(test_error) + '\n')
# '''


