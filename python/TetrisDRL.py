#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
    This Notebook is for learning only
'''


# In[2]:


# Import
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import deque
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tetris as tetris


# In[3]:



# Hyperparameter 

num_episodes = 500
num_exploration_episodes = 100
max_len_episode = 1000
batch_size = 40
learning_rate = 0.01
gamma = 0.95
initial_epsilon = 1.0
final_epsilon = 0.01

eps_decay = 0.995
eps_min = 0.01


# In[ ]:





# In[ ]:



class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.state_dim  = 216
        self.action_dim = 40
        self.epsilon = 1.
        self.dense1 = tf.keras.layers.Dense(units=216, input_dim=216,activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dropout(0.5)#tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=40, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(units=self.action_dim)
        
        
        
        self.model = self.create_model()
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
    
    def create_model(self):
#         model = tf.keras.Sequential([
#             Input((self.state_dim,)),
#             Dense(32, activation='relu'),
#             Dense(16, activation='relu'),
#             Dense(self.action_dim)
#         ])
        model = tf.keras.models.Sequential()
        model.add(self.dense1)
        model.add(self.dense2)
        model.add(self.dense3)
        model.add(self.dense4)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate))
        return model
    
    def predict(self, state):
        return self.model.predict(state)
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= eps_decay
        self.epsilon = max(self.epsilon, eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, 39)
        
        
        return np.argmax(q_value)
    
    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
        return states, actions, rewards, next_states, done
    
    def size(self):
        return len(self.buffer)
    
class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = 216
        self.action_dim = 40

        self.model = QNetwork()
        self.target_model = QNetwork()
        self.target_update()

        self.buffer = ReplayBuffer()

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)
    
    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            #print(states, actions, rewards, next_states, done,next_q_values )
            targets[range(batch_size), actions] = rewards + (1-done) * next_q_values * gamma
            self.model.train(states, targets)
    
    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            done, total_reward = False, 0
            state = self.env.reset()
            while not done:
                action = self.model.get_action(state)
                #print(action)
                next_state, reward, done, info = self.env.step_action(action)
                self.buffer.put(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
            print("Score",self.env.score,"Total Steps",self.env.total_steps,flush=True)
            
            
            if self.buffer.size() >= batch_size:
                self.replay()
            self.target_update()
            print('EP{} EpisodeReward={}'.format(ep, total_reward),flush=True)
            #wandb.log({'Reward': total_reward})


def main():
    env = tetris.Tertris(10,20)
    agent = Agent(env)
    agent.train(max_episodes=100000)

if __name__ == "__main__":
    main()


# In[ ]:




