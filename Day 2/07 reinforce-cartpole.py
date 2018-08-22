import numpy as np
import gym 
import argparse

gym.logger.set_level(40) # Get rid of warnings

parser = argparse.ArgumentParser(description="Run REINFORCE for 'CartPole'")
parser.add_argument('--agent', default='random')
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--play')
parser.add_argument('--n_steps', default=20, type=int)
parser.add_argument('--n_episodes', default=1000, type=int)

args = parser.parse_args()

class RandomAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def predict(self, state):
        action = 0 if np.random.randn() < 0 else 1
        return action

    def update(self,state,reward,done):
        pass


class HardCodedAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def predict(self, state):
        if state[2] < 0: # just looking at the angle
            action = 0
        else: 
            action = 1
        return action

    def update(self,state,reward,done):
        pass



class Master:
    def __init__(self):
        self.game_name = 'CartPole-v0'

    def play(self, n_steps, agent, render=False):
        env = gym.make(self.game_name)
        state = env.reset()
        ep_reward = 0
        for _ in range(n_steps):
            if render:
                env.render()
            action = agent.predict(state)
            state, reward, done, info = env.step(action)
            agent.update(state, reward, done)
            ep_reward += reward
        env.close()
        return ep_reward
        

    def train(self, n_episodes, n_steps=200, agent=RandomAgent(n_actions=2), window=100):
        rewards = []
        for ep in range(n_episodes):
            ep_reward = self.play(n_steps, agent)

            if len(rewards) < window:
                rewards.append(ep_reward)
            else:
                rewards[ep % window] == ep_reward

            if (ep+1) % window == 0:
                print("Episode: {}/{}. Average reward in {} consecutive episodes: {}.".format(ep+1,    
                                                                                                    n_episodes,
                                                                                                    window,
                                                                                                    np.mean(rewards)
                                                                                                    )
                                                                                                    )

def main():
    master = Master()
    if args.play:
        master.play(args.n_steps, RandomAgent(n_actions=2))
    else:
        master.train(args.n_episodes)

if __name__ == "__main__":
    main()