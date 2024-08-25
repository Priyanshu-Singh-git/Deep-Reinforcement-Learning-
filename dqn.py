# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 23:37:08 2024

@author: Priyanshu singh
"""
import torch 
import torch.nn as nn
import random as rd
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
buffer_size = int(1e5)
minibatch = 100
discount_factor = 0.99
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Neural_net(nn.Module):
    def __init__(self,state_size,action_size):
        
        super(Neural_net,self).__init__()
        torch.manual_seed(42)
        self.main = nn.Sequential(
                    nn.Linear(state_size,64),
                    nn.ReLU(True),
                    nn.Linear(64,64),
                    nn.ReLU(True),
                    
                    nn.Linear(64,action_size)
            )
    def forward(self,input):
        return self.main(input)

class Duelling_Net(nn.Module):
    def __init__(self,state_size,action_size):
        super(Duelling_Net, self).__init__()
        
        self.features = nn.Sequential(
                    nn.Linear(state_size,64),
                    nn.ReLU(True),
                    nn.Linear(64,64),
                    nn.ReLU(True),
                    
                        )
        self.value = nn.Sequential(
                     nn.Linear(64, 64),
                     nn.ReLU(True),
                     nn.Linear(64,1)
                )
        
        self.advantage = nn.Sequential(
                         nn.Linear(64, 64),
                         nn.ReLU(True),
                         nn.Linear(64,action_size))
    def forward(self,input):
        features = self.features(input)
        value = self.value(features)
        advantage = self.advantage(features)
        
        q = value + (advantage - advantage.mean(dim=1,keepdim = True))
        
        return q
    
class Replay_Memory():
    def __init__(self,buffer_size,device):
        self.memory = []
        self.capacity = buffer_size
        self.device = device
        
    def push(self,event):
        self.memory.append(event)
        if len(self.memory)>self.capacity:
            del self.memory[0]
            
    def sample(self,minibatch):
        experience = rd.sample(self.memory,k=minibatch)
        states = torch.from_numpy(np.vstack([e[0] for e in experience if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experience if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experience if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experience if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experience if e is not None]).astype(np.uint8)).float().to(self.device)   
        return states,actions,rewards,next_states,dones


class ARL():
    def __init__(self,state_size,action_size,device,buffer_size,architecture):
        self.device = device
        self.t_step = 0
        self.state_size = state_size
        self.action_size = action_size
        self.localq = architecture(self.state_size, self.action_size).to(device)
        self.targetq = architecture(self.state_size, self.action_size).to(device)
        self.memory = Replay_Memory(buffer_size, self.device)
        self.optimizer = torch.optim.Adam(self.localq.parameters(), lr = 0.0006)
        self._loss= None
    
    def step(self,state,action,reward,next_state,done,algorithm):
        self.t_step+=1
        self.memory.push((state,action,reward,next_state,done))
        
        if self.t_step%4 ==0:
            if len(self.memory.memory)>minibatch:
                experiences = self.memory.sample(minibatch)
                self._loss = algorithm(experiences, discount_factor)
        return self._loss
                
                
    def dqn_learn(self,experiences,discount_factor):
        s,a,r,ns,d = experiences
        next_q = self.targetq(ns).detach().max(1)[0].unsqueeze(1)
        q_targets = r + discount_factor*next_q*(1-d)
        q_exp = self.localq(s).gather(1,a)
        loss = nn.MSELoss()(q_exp,q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.localq, self.targetq, 0.001)
        
        return loss.mean().item()
        
    def ddqn_learn(self,experiences,discount_factor):
        s,a,r,ns,d = experiences
        with torch.no_grad():
            a_ns = self.localq(ns).argmax(dim=1,keepdim=True)
        
        next_q = self.targetq(ns).gather(1,a_ns)
        q_targets = r + discount_factor*next_q*(1-d)
        q_exp = self.localq(s).gather(1,a)
        loss = nn.MSELoss()(q_exp,q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.localq, self.targetq, 0.001)
        
        return loss.mean().item()
        
        
    def soft_update(self,local,target,interpolate):
        for target_param,local_param in zip(target.parameters(),local.parameters()):
            target_param.data.copy_(interpolate*local_param.data+(1-interpolate)*target_param.data)
    
    def act(self,state,epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.localq.eval()
        with torch.no_grad():
            action_values = self.localq(state)
        self.localq.train()
        if rd.random()>epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return rd.choice(np.arange(self.action_size ))

env = gym.make("LunarLander-v2")
score_100 = deque(maxlen=100)
num_episode = 2000
steps_per_episode = 1000
ep1 = 1.0
ep_end = 0.01
ep_dec = 0.995
epsilon = ep1
state_size=env.observation_space.shape[0]
action_size=env.action_space.n

#%%<-------------------------------DQN---------------------------------------->
agent = ARL(state_size, action_size, device, buffer_size,Neural_net)
dqn_loss_list = []
dqn_score_list = []

for episode in range(num_episode+1):
    state,_ = env.reset()
    score = 0
    
    for t in range(steps_per_episode):
        
        action = agent.act(state, epsilon)
        next_state,reward,done,_,_ = env.step(action)
        dqn_loss = agent.step(state, action, reward, next_state, done,agent.dqn_learn)
        state = next_state
        score += reward
        
        if done:
            break
    score_100.append(score)
    dqn_score_list.append(score)
    dqn_loss_list.append(dqn_loss)

    epsilon=max(ep_end,ep_dec*epsilon)
    print(f"Episode :{episode}-------------DQN---------------Score :{np.mean(score_100):.2f}")
    if episode%100==0:
        print(f"Episode :{episode}------------DQN----------------Average_Score :{np.mean(score_100):.2f}")
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].plot(dqn_loss_list,label='DQN LOSS',color = 'magenta')
        
        ax[0].set_xlabel("episodes")
        ax[0].set_ylabel("DQN Loss")
        ax[0].legend()
        ax[1].plot(dqn_score_list,label = 'DQN SCORE',color='teal')
        ax[1].set_xlabel("episodes")
        ax[1].set_ylabel("DQN Score")
        ax[1].legend()
        plt.show()
        
#MAX SCORE 260
        
env.close()
#%%<-------------------------Saving DQN Parameters---------------------------->
path1 = "D:/pythonProject/DRL/dqn_local.pth"
path2 = "D:/pythonProject/DRL/dqn_target.pth"

torch.save(agent.localq.state_dict(),path1)
torch.save(agent.targetq.state_dict(),path2)
#%%<-------------------------------DDQN--------------------------------------->

agent = ARL(state_size, action_size, device, buffer_size,Neural_net)
ddqn_loss_list = []
ddqn_score_list = []

for episode in range(num_episode+1):
    state,_ = env.reset()
    score = 0
    
    for t in range(steps_per_episode):
        
        action = agent.act(state, epsilon)
        next_state,reward,done,_,_ = env.step(action)
        ddqn_loss = agent.step(state, action, reward, next_state, done,agent.ddqn_learn)
        state = next_state
        score += reward
        
        if done:
            break
    score_100.append(score)
    ddqn_score_list.append(score)
    ddqn_loss_list.append(ddqn_loss)

    epsilon=max(ep_end,ep_dec*epsilon)
    print(f"Episode :{episode}-------------DDQN---------------Score :{np.mean(score_100):.2f}")
    if episode%100==0:
        print(f"Episode :{episode}------------DDQN----------------Average_Score :{np.mean(score_100):.2f}")
        
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].plot(ddqn_loss_list,label='DDQN LOSS',color = 'magenta')
        
        ax[0].set_xlabel("episodes")
        ax[0].set_ylabel("DDQN Loss")
        ax[0].legend()
        ax[1].plot(ddqn_score_list,label = 'DDQN SCORE',color='teal')
        ax[1].set_xlabel("episodes")
        ax[1].set_ylabel("DDQN Score")
        ax[1].legend()
        plt.show()
        
# Max Score 235     
env.close()
#%%
path3 = "D:/pythonProject/DRL/double_dqn_local.pth"
path4 = "D:/pythonProject/DRL/double_dqn_target.pth"

torch.save(agent.localq.state_dict(),path3)
torch.save(agent.targetq.state_dict(),path4)
        
#%%<------------------------Duelling DQN-------------------------------------->
agent = ARL(state_size, action_size, device, buffer_size,Duelling_Net)
duelling_dqn_loss_list = []
duelling_dqn_score_list = []

for episode in range(num_episode+1):
    state,_ = env.reset()
    score = 0
    
    for t in range(steps_per_episode):
        
        action = agent.act(state, epsilon)
        next_state,reward,done,_,_ = env.step(action)
        dqn_loss = agent.step(state, action, reward, next_state, done,agent.dqn_learn)
        state = next_state
        score += reward
        
        if done:
            break
    score_100.append(score)
    duelling_dqn_score_list.append(score)
    duelling_dqn_loss_list.append(dqn_loss)

    epsilon=max(ep_end,ep_dec*epsilon)
    print(f"Episode :{episode}------DuellingDQN----------Score :{np.mean(score_100):.2f}")
    if episode%100==0:
        print(f"Episode :{episode}------DuellingDQN------Average_Score :{np.mean(score_100):.2f}")
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].plot(duelling_dqn_loss_list,label='Duelling DQN LOSS',color = 'magenta')
        
        ax[0].set_xlabel("episodes")
        ax[0].set_ylabel("Duelling DQN Loss")
        ax[0].legend()
        ax[1].plot(duelling_dqn_score_list,label = 'Duelling DQN SCORE',color='teal')
        ax[1].set_xlabel("episodes")
        ax[1].set_ylabel("Duelling DQN Score")
        ax[1].legend()
        plt.show()
        
#MAX SCORE 221
        
env.close()
#%%
path5 = "D:/pythonProject/DRL/duelling_dqn_local.pth"
path6 = "D:/pythonProject/DRL/duelling_dqn_target.pth"


torch.save(agent.localq.state_dict(),path5)
torch.save(agent.targetq.state_dict(),path6)
#%%
agent = ARL(state_size, action_size, device, buffer_size,Duelling_Net)
duelling_ddqn_loss_list = []
duelling_ddqn_score_list = []

for episode in range(num_episode+50001):
    state,_ = env.reset()
    score = 0
    
    for t in range(steps_per_episode):
        
        action = agent.act(state, epsilon)
        next_state,reward,done,_,_ = env.step(action)
        ddqn_loss = agent.step(state, action, reward, next_state, done,agent.ddqn_learn)
        state = next_state
        score += reward
        
        if done:
            break
    score_100.append(score)
    duelling_ddqn_score_list.append(score)
    duelling_ddqn_loss_list.append(ddqn_loss)

    epsilon=max(ep_end,ep_dec*epsilon)
    print(f"Episode :{episode}------DuellingDDQN----------Score :{np.mean(score_100):.2f}")
    if episode%100==0:
        print(f"Episode :{episode}------DuellingDDQN------Average_Score :{np.mean(score_100):.2f}")
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].plot(duelling_ddqn_loss_list,label='Duelling DDQN LOSS',color = 'magenta')
        
        ax[0].set_xlabel("episodes")
        ax[0].set_ylabel("Duelling DQN Loss")
        ax[0].legend()
        ax[1].plot(duelling_ddqn_score_list,label = 'Duelling DDQN SCORE',color='teal')
        ax[1].set_xlabel("episodes")
        ax[1].set_ylabel("Duelling DDQN Score")
        ax[1].legend()
        plt.show()
        
#MAX SCORE 266
        
env.close()
#%%
path5 = "D:/pythonProject/DRL/duelling_ddqn_local.pth"
path6 = "D:/pythonProject/DRL/duelling_ddqn_target.pth"


torch.save(agent.localq.state_dict(),path5)
torch.save(agent.targetq.state_dict(),path6)