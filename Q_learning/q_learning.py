from environment import MountainCar
import sys
import numpy as np

def action_choice(epsilon, Q):
    if np.random.binomial(1,epsilon) == 1:
        return np.random.choice([0,1,2])
    else:
        return np.argmax(Q, axis = 0)[0]

def get_s(state):
    s = np.zeros(env.state_space)
    # print('initial',s)
    for i in state.keys():
        # print('k',k)
        s[i] = state[i]
        # print('sk',s[k])
    s = s[np.newaxis, :]
    # print('s',s)
    return s

def get_d_wq(A,s):
    d_wq = np.zeros((env.action_space, env.state_space))
    # print('d_wq', d_wq)
    d_wq[A, :] = s
    return d_wq

def q_learning(episodes, max_iterations, env, epsilon, gamma, lr):
    b=0
    w = np.zeros((env.action_space, env.state_space))
    sum_list=[]
    for episode in range(episodes):
        s = env.reset()
        sum_reward = 0
        for step in range(max_iterations):
            s = get_s(s)
            Q = np.dot(w,s.T) + b
            a = action_choice(epsilon, Q)
            q = Q [a,0]

            d_q = get_d_wq(a,s)
            
            s, reward, done = env.step(a)

            sum_reward += reward

            s_next = get_s(s)

            Q_next = np.dot(w,s_next.T) + b
            
            a_next = action_choice(epsilon, Q_next)
            q_next = Q_next[a_next]

            w -= lr * (q - reward - gamma * q_next) * d_q
            b -= lr * (q - reward - gamma * q_next)
            if done == True:
                break
        sum_list.append(sum_reward)
    return b, sum_list, w


if __name__ == "__main__":
    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iteration = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    lr = float(sys.argv[8])

    env = MountainCar(mode)
    # b=0
    # w = np.zeros((env.action_space, env.state_space))
    b, sum_list, w = q_learning(episodes, max_iterations, env, epsilon, gamma, lr)
    with open(weight_out, 'w') as f:
        f.write(str(b)+'\n')
        for lines in w.T:
            for i in lines:
                f.write(str(float(i))+'\n')

    with open(returns_out, 'w') as f:
        for i in sum_list:
            f.write(str(i)+'\n')