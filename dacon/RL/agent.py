#!/usr/bin/env python
# coding: utf-8

# In[1]:


from puzzle_game import Game
from collections import deque
from model import Linear_QNet, QTrainer
import random


# In[ ]:


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.model = Linear_QNet(11, 256, 3) #5, 16, 4
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) #vrati verovatnoce
            move = torch.argmax(prediction).item() #argmax vrati index kao LongTensor, sa item() ga pretvaramo u broj
            final_move[move] = 1

        return final_move
    
    
def train():
    agent = Agent()
    game = Game()
    total_score = 0
    record = 0
    
    while True:
        pass


if __name__ == '__main__':
    train()

# %%
import torch
a = [1, 2, 3]
a = torch.tensor(a, dtype=torch.float)
#len(a.shape) #ako je 1 onda je niz
a = torch.unsqueeze(a, 0) #sad smo dobili kao neku matricu
print(a)

print(torch.argmax(a).item())

