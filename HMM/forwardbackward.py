import sys
import numpy as np

def load_data(prior_path, emit_path, trans_path):
    load = lambda path, dim: np.loadtxt(path, delimiter=" ", ndmin=dim) 
    prior = load(prior_path, 1)
    emit = load(emit_path, 2)
    trans = load(trans_path, 2)
    return prior, emit, trans
    
    
def cal_alpha(pi, b, a, t, j, alpha, sequence):
    """
    pi: prior
    a: transitional prob
    b: emission prob
    t: time at t
    j: the hidden state `j`.
    alpha: np.array, shape = (T, J)
    sequence: store the index of words. shape = (T,)
    """
    if (t == 0):
        alpha[t, j] = pi[j] * b[j, sequence[0]]
    else:
        alpha[t, j] = b[j, sequence[t]] * (alpha[t-1] @ a[:, j])
    return alpha

def cal_beta(pi, b, a, t, j, alpha, beta, sequence, T, J):
    """
    pi: prior
    a: transitional prob
    b: emission prob
    t: time at t, raging from [0, T)
    j: the hidden state `j`, raging from [0, J)
    alpha: np.array, shape = (T, J)
    beta: np.array, shape = (T, J)
    sequence: store the index of words. shape = (T,)
    """
    if (t == T-1):
        beta[t, j] = 1
    else:
        for k in range(J):
            beta[t, j] += b[k, sequence[t+1]] * beta[t+1, k] * a[j, k]
    return alpha

def get_tag(t, alpha, beta):
    """
    t: time at t, raging from [0, T)
    j: the hidden state `j`, raging from [0, J)
    
    return: str
    """
    prob = alpha[t] * beta[t]
    idx = np.argmax(prob)
    return idx

def predict_one_sample(sequence, a, b, pi):
    T = len(sequence)
    J = a.shape[0]
    # forward
    alpha = np.zeros((T, J))
    for t in range(T):
        for j in range(J):
            cal_alpha(pi, b, a, t, j, alpha, sequence)
    beta = np.zeros((T, J))
    for t in reversed(range(T)):
        for j in range(J):
            cal_beta(pi, b, a, t, j, alpha, beta, sequence, T, J)
    tags = []
    for t in range(T):
        tag = get_tag(t, alpha, beta)
        tags.append(tag)
    # print(tags)
    likelihood = log_likelihood(alpha, T)
    return tags, likelihood

def log_sum_exp_trick(var):
    m = var.max()
    return m + np.log(np.sum(np.exp(var-m)))
    
def accuracy(y, y_pred):
    c=0
    num=0
    for i,j in zip(y,y_pred):
        num+=len(i)
        for a,b in zip(i,j):
            if a==b:
                c+=1
    # true = sum([1 if i == j else 0 for i, j in zip(y, y_pred)])
    acc = c / num
    return acc

def log_likelihood(alphas, T):
    return np.log(np.sum(alphas[T-1, :]))
    
def inference(in_file, out_file, metric_file, index_to_word,index_to_tag, a, b, pi):
    in_file = open(in_file, "r")
    out_file = open(out_file, "w")
    dict_index_word = {}
    dict_index_word_reversed = {}
    with open(index_to_word, 'r') as f:
        lines=f.readlines()
        i=0
        for line in lines:
            line=line.replace("\n","")
            dict_index_word[line]=i
            dict_index_word_reversed[i] = line
            i+=1

    dict_index_tag = {}  
    dict_index_tag_reversed = {}          
    with open(index_to_tag, 'r') as f:
        lines=f.readlines()
        i=0
        for line in lines:
            line=line.replace('\n','')
            dict_index_tag[line]=i
            dict_index_tag_reversed[i] = line
            i+=1
    acc_sum = 0
    log_sum = 0
    n_samples = 0
    all_labels=[]
    all_y=[]
    for line in in_file.readlines():
        line = line.strip()
        sentence = line.split(" ")
        x = [dict_index_word[i.split("_")[0]] for i in sentence]
        y = [dict_index_tag[i.split("_")[1]] for i in sentence]
        labels, likelihood = predict_one_sample(x, a, b, pi)
        all_labels.append(labels)
        all_y.append(y)
        acc = accuracy(all_y, all_labels)
        acc_sum += acc
        log_sum += likelihood
        string = " ".join(["{}_{}".format(dict_index_word_reversed[m], dict_index_tag_reversed[n]) for m, n in zip(x, labels)])
        out_file.write(string + "\n")
        n_samples += 1
    avg_log = log_sum / n_samples
    avg_acc = acc_sum / n_samples
    string = "Average Log-Likelihood: {}\nAccuracy: {}".format(avg_log, avg_acc)
    with open(metric_file, "w") as f:
        f.write(string)
        
    in_file.close()
    out_file.close()
        

if __name__ == "__main__":
    test_input_file = sys.argv[1]
    index_to_word_file = sys.argv[2]
    index_to_tag_file = sys.argv[3]
    hmmprior_file = sys.argv[4]
    hmmemit_file = sys.argv[5]
    hmmtrans_file = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]
    
    prior, emit, trans = load_data(hmmprior_file, hmmemit_file, hmmtrans_file)
    print("prior: {}, emission: {}, transition: {}".format(prior.shape, emit.shape, trans.shape))
    
    inference(test_input_file, 
              predicted_file, 
              metric_file,
              index_to_word_file, 
              index_to_tag_file, 
              trans, 
              emit, 
              prior)
    
    