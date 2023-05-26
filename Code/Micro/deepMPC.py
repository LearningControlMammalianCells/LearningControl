import keras
import numpy as np



def rewardG(state, target):
    C = 0.1
    x = (1-abs(target-state))
    r = C*x
    return r

def deepMPC(model_LSTM, initial_state, target, K = 4 , T = 6 , plot=False):
    # T number of step ahead
    # K number of simulation per action
    gamma = 0.9
    act = [0, 1]
    G_action = []
    for a in act:


        Glist = []
        for k in range(K):
            reward_list = []
            action_list = []
            trajectory = []

            x0 = np.copy(initial_state)
            trajectory.append(x0[-1][0])
            x0[-1][-1] = np.copy(a)

            next_state = model_LSTM.predict(np.array([x0]), verbose=0)[0][0]
            x0 = np.vstack((x0[1:], [next_state, -100]))

            trajectory.append(next_state)

            reward_list.append(gamma ** (0) * rewardG(next_state, target))
            action_list.append(a)
            for t in range(T - 1):
                action_now = np.random.choice(act)
                action_list.append(action_now)
                x0[-1][-1] = np.copy(action_now)
                next_state = model_LSTM.predict(np.array([x0]), verbose=0)[0][0]
                trajectory.append(next_state)
                x0 = np.vstack((x0[1:], [next_state, -100]))
                reward_list.append(gamma ** (t + 1) * rewardG(next_state, target))
            G = np.sum(reward_list)
            # G=np.around(G,4)


            Glist.append(G)


        G_action.append(np.mean(Glist))
    print(G_action)
    # bestAction=moves[np.argmax(Glist)][0]
    #print(G_action)
    BestAction = np.argmax(G_action)
    #print(BestAction)
    return BestAction

