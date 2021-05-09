import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import copy


class QFunction(chainer.Chain):
    def __init__(self, obj_size, n_actions, n_hidden_channels=10):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(obj_size, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l3 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x, test=False):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = chainerrl.action_value.DiscreteActionValue(self.l3(h2))
        return y
# action = 0: Pull the string!, action = 1: Tilt the vat!


def random_action():
    return np.random.choice([0, 1])


def step(_state, action):
    state = _state.copy()
    reward = 0
    if state[0] == 0 and state[1] == 1:
        if action == 0:
            state[0] = 1
    elif state[0] == 1 and state[1] == 1:
        if action == 0:
            state[0] = 0
        elif action == 1:
            state[1] = 0
            reward = 1
    elif state[0] == 1 and state[1] == 0:
        if action == 0:
            state[0] = 0
            state[1] = 1

    return np.array(state), reward


gamma = 0.8
alpha = 0.5
max_number_of_steps = 15
num_episodes = 50

q_func = QFunction(2, 2)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.0, decay_steps=num_episodes, random_action_func=random_action)
replay_buffer = chainerrl.replay_buffer.PrioritizedReplayBuffer(
    capacity=10 ** 6)


def phi(x): return x.astype(np.float32, copy=False)


agent = chainerrl.agents.DoubleDQN(q_func, optimizer, replay_buffer, gamma, explorer,
                                   replay_start_size=50, update_interval=1, target_update_interval=10, phi=phi)

# agent.load('agent')

for episode in range(num_episodes):
    state = np.array([0, 1])
    R = 0
    reward = 0
    done = True

    for t in range(max_number_of_steps):
        action = agent.act_and_train(state, reward)
        next_state, reward = step(state, action)
        print(state, action, reward, next_state)
        R += reward
        state = next_state
    agent.stop_episode_and_train(state, reward, done)
    print('episode: ', episode+1, 'R', R,
          'statistics:', agent.get_statistics())

agent.save('agent')
