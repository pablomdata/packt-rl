import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# The estimator is of the form (s,a) -> scalar value


class FunctionEstimator:
    def __init__(self,n_actions):
        self.n_actions = n_actions
        self.initial_state = env.reset()
        self.model = self._build_model()
        self.memory_buffer = deque(maxlen=2000)
        self.update_buffer = []

    def _concat(self, state, action):
        return np.hstack([state,action]).reshape(1,-1)

    def _build_model(self):   
        model = Sequential()
        model.add(Dense(24, input_dim=5, activation='tanh'))
        model.add(Dense(24, activation='tanh'))
        model.add(Dense(1, activation='linear'))
        model.compile(
                    loss='mse'
                    , optimizer= Adam(lr = 0.001)
                    )      
        return model
    
    def update(self,buffer):
        states = [buffer[ix][0] for ix in range(len(buffer))]
        actions = [buffer[ix][1] for ix in range(len(buffer))]
        td_targets = [buffer[ix][2] for ix in range(len(buffer))]
        for state, action, target in zip(states, actions, td_targets):
            self.model.fit(self._concat(state,action), [td_target], verbose=0)
        
    def predict(self,state):
        concats = [np.array([self._concat(state, a)]).reshape(1,-1) for a in range(self.n_actions)]
        return [self.model.predict(c) for c in concats]

    
    def remember(self, state, action,td_target):
        self.memory_buffer.append((state,action,td_target))
    
    
    def replay(self,batch_size):
        # Experience replay
            # choose only a sample from the collected experience
        update_buffer_idxs = np.random.choice(len(self.memory_buffer)
                                                , size=min(len(self.memory_buffer), batch_size)
                                                , replace=False
                                                    ) 
        update_buffer_idxs = np.ravel(update_buffer_idxs)
                
        for ix in range(len(update_buffer_idxs)):
            saved_ix = update_buffer_idxs[ix]
            self.update_buffer.append(self.memory_buffer[saved_ix])
            
        self.update(self.update_buffer)


## Auxiliary function for the policy
def make_policy(estimator, n_actions, ep):
    def policy_fn(state):
        preds = np.ravel(estimator.predict(state))
        noise = np.ravel(np.random.randn(1,n_actions)*(1./(ep+1)))
        action = np.argmax(preds+noise)
        return action
    return policy_fn

if __name__ == "__main__":
    import gym
    gym.logger.set_level(40)

    env = gym.make('CartPole-v0')

    n_episodes = 1000
    gamma = 1
    estimator = FunctionEstimator(env.action_space.n)

    score = []
    

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        policy = make_policy(estimator, env.action_space.n, ep)
        ep_reward = 0

        while not done:
            action = policy(state)
            new_state, reward, done, _ = env.step(action)
            ep_reward += reward

            # Update the Q-function
            if done:
                td_target = reward
            else:
                td_target = reward + gamma*np.argmax(estimator.predict(new_state))

            estimator.remember(state,action, td_target)
			# Update the state
            state = new_state
        #
        estimator.replay(32)
        
        # Show stats
        if done:
            if len(score) < 100:
                score.append(ep_reward)
            else:
                score[ep % 100] = ep_reward
                
            if (ep+1) % 100 == 0:
                print("Number of episodes: {} . Average 100-episode reward: {}".format(ep+1, np.mean(score)))

