import numpy as np
import sys

def load_data(train_input,index_to_word,index_to_tag):
    sentences=[]
    with open(train_input, "r") as f:
        lines = f.readlines()
        # print(lines)
        for line in lines:
            # print(line)
            line=line.replace("\n","")
            line=line.split(' ')
            sentences.append(line)
        # print(sentences)
    with open(index_to_word, 'r') as f:
        dict_index_word={}
        lines=f.readlines()
        # print(lines)
        i=0
        for line in lines:
            line=line.replace("\n","")
            dict_index_word[line]=i
            i+=1
        # print(dict_index_word)
    with open(index_to_tag, 'r') as f:
        dict_index_tag={}
        lines=f.readlines()
        i=0
        for line in lines:
            line=line.replace('\n','')
            dict_index_tag[line]=i
            i+=1
        # print(dict_index_tag)
    return sentences, dict_index_word, dict_index_tag

def calculate(train_input,index_to_word,index_to_tag):
    sentences, dict_index_word, dict_index_tag=load_data(train_input,index_to_word,index_to_tag)
    a=len(dict_index_tag)
    prior=np.zeros(a)
    # print(prior)
    transition=np.zeros((a,a))
    # print(transition)
    emit=np.zeros((a,len(dict_index_word)))
    # print(emit)
    for line in sentences:
        # print(line)
        for i in range(len(line)):
            # print(item)
            word,tag=line[i].split('_')
            # print(word,tag)
            word_index=dict_index_word[word]
            tag_index=dict_index_tag[tag]
            emit[tag_index][word_index]+=1
            if i==0:
                prior[tag_index]+=1
            if (i+1)<len(line):
                word_next,tag_next=line[i+1].split('_')
                word_index_next=dict_index_word[word_next]
                tag_index_next=dict_index_tag[tag_next]
                transition[tag_index][tag_index_next]+=1
    prior+=1
    transition+=1
    emit+=1
    prior_new=prior/np.sum(prior)
    # print(prior)
    tran_sum=np.sum(transition,axis=1)
    # print(tran_sum)
    tran_sum=tran_sum[:,np.newaxis]
    # print(tran_sum)
    tran_new=transition/tran_sum
    # print(tran)
    emit_sum=np.sum(emit,axis=1)
    emit_sum=emit_sum[:,np.newaxis]
    emit_new=emit/emit_sum
    # print(emit_new)
    return prior_new,emit_new,tran_new

if __name__ == '__main__':

    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    prior_new,emit_new,tran_new=calculate(train_input,index_to_word,index_to_tag)

    hmmprior = open(hmmprior,'wt',encoding='utf-8')
    for item in prior_new:
        hmmprior.write(str(item) + '\n')
    hmmemit = open(hmmemit,'wt',encoding='utf-8')
    a=[]
    b=[]
    emit_list=emit_new.tolist()
    # print(emit_list)
    for item in emit_list:
        str1=''
        for i in item:
            str1+=str(i)+' '
        str1=str1[:-1]
        a.append(str1)
    # print(a)
    for item in a:
        hmmemit.write(item + '\n')

    tran_list=tran_new.tolist()
    for item in tran_list:
        str1=''
        for i in item:
            str1+=str(i)+' '
        str1=str1[:-1]
        b.append(str1)
    
    hmmtrans = open(hmmtrans,'wt',encoding='utf-8')
    for item in b:
        hmmtrans.write(item + '\n')
