import numpy as np
import gym
from tqdm import tqdm

seed = 1
np.random.seed(seed)


def backward_pass(P, n_states, n_actions, traj_length, terminal, rewards):
    z_states = np.zeros((n_states))
    z_actions = np.zeros((n_states, n_actions))
    z_states[terminal] = 1
    for n in range(traj_length):
        for i in range(n_states):
            for j in range(n_actions):
                curr_sum = 0
                for k in range(n_states):
                    c1 = P[i, j, k]
                    # print('c1',c1)
                    c2 = np.exp(rewards[i])
                    # print('c2',c2)
                    c3 = z_states[k]
                    # print('c3',c3)
                    curr_sum += c1*c2*c3
                    # if(curr_sum<=0):
                    #     print("{0},{1},{2}".format(c1,c2,c3))
                z_actions[i, j] = curr_sum
            z_states[i] = np.sum(z_actions[i, :])
            z_states[i] = z_states[i] + 1 if i == terminal else z_states[i]

    return (z_states, z_actions)


def local_action_probability_computation(z_states, z_actions):
    policy = np.zeros(z_actions.shape)
    for i in range(z_actions.shape[0]):
        for j in range(z_actions.shape[1]):
            policy[i, j] = z_actions[i, j] / z_states[i] if z_states[i]>0 else 0
    return (policy)


def forward_pass(P, policy, trajectories, traj_length):
    D_t = np.zeros((policy.shape[0], traj_length))
    for i in trajectories:
        D_t[i[0][0], :] += 1
    D_t[:, :] = D_t[:, :] / len(trajectories)

    for s in range(policy.shape[0]):
        for t in range(traj_length-1):
            D_t[s, t+1] = sum([sum([D_t[k, t] * policy[k, a] * P[s, a,k] for k in range(policy.shape[0])]) for a in
                             range(policy.shape[1])])

    D = np.sum(D_t, 1)

    return (D)


def expected_edge_frequency_calculation(P, trajectories, terminal, rewards):
    n_states = P.shape[0]
    n_actions = P.shape[1]
    traj_length = len(trajectories[0])
    z_s, z_a = backward_pass(P, n_states, n_actions, traj_length, terminal, rewards)
    policy = local_action_probability_computation(z_s, z_a)
    D = forward_pass(P, policy, trajectories, traj_length)
    return (D)


def update(theta, alpha, f_expert, f, D):
    gradient = f_expert - np.dot(f.T,D)
    theta += alpha * gradient
    return (theta)


def expert_feature_expectations(trajectories, features):
    exps = np.zeros((features.shape[1]))
    for i in trajectories:
        for s in i:
            f = features[s[0], :]
            exps += f / len(trajectories)
    return (exps)


def irl(features, P,trajectories, epochs,alpha):
    terminal = trajectories[0][-1][0]
    theta = np.random.uniform(size=(features.shape[1]))
    exps = expert_feature_expectations(trajectories, features)
    for i in tqdm(range(epochs)):
        rewards = np.dot(features, theta)
        D = expected_edge_frequency_calculation(P, trajectories, terminal, rewards)
        theta = update(theta, alpha, exps, features, D)

    rewards = np.dot(features, theta)

    np.save('rewards', rewards)
    return (rewards)
