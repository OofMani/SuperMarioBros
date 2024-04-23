import torch

import matplotlib.pyplot as plt
import random

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

from agent import Agent

from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers

import os

from utils import *

model_path = os.path.join("models/first_run/model.pt")
os.makedirs(os.path.join("models/first_run"), exist_ok=True)

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 100
NUM_OF_EPISODES = 10000

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)

env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
agent.load_model(model_path)

if not SHOULD_TRAIN:
    folder_name = ""
    ckpt_name = ""
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.2
    agent.eps_min = 0.0
    agent.eps_decay = 0.0

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)


plt.ion() # turning interactive mode on

x=[0]
y=[0]
tensum=0
hunsum=0

ten_avg=[0]
curr_ten_avg=0
hun_avg=[0]
curr_hun_avg=0

# plotting the first frame
graph = plt.plot(x,y)[0]
graph2 = plt.plot(x,ten_avg)[0]
graph3 = plt.plot(x,hun_avg)[0]

plt.pause(1)

curr=0


for i in range(NUM_OF_EPISODES):   


    print("Episode: ", i+1)
    done = False
    state, _ = env.reset()
    total_reward = 0
    while not done:
        a = agent.choose_action(state)
        new_state, reward, done, truncated, info  = env.step(a)
        total_reward += reward

        if SHOULD_TRAIN:
            agent.store_in_memory(state, a, reward, new_state, done)
            agent.learn()

        state = new_state

    print("Total reward: ", total_reward, " Epsilon: ", agent.epsilon, " Size of replay buffer: ", len(agent.replay_buffer), " Learn step counter: ", agent.learn_step_counter)

    # preparing the data
    y.append(total_reward)
    x.append(i)


    tensum+=total_reward
    hunsum+=total_reward

    # removing the older graph
    graph.remove()

    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        print("\n")
        print("-"*35)
        print("Saving Weights...")
        print("-"*35)

        agent.save_model(model_path)
    else:
        print("Not Saving...\n")
        
    print("Total reward: ", total_reward)



    if((i+1)%100==0):
        
        curr_hun_avg=hunsum/100
        print("/n -----------------ADDING HUNSUM = ", curr_hun_avg, "-------------------------/n")
        for i in range(100):
            hun_avg.append(curr_hun_avg)

        graph3.remove()
        graph3 = plt.plot(x,hun_avg,color = 'r')[0]
        hunsum=0

        curr_ten_avg=tensum/10
        print("/n -----------------ADDING TENSUM = ", curr_ten_avg, "-------------------------/n")
        for i in range(10):
            ten_avg.append(curr_ten_avg)
        
        graph2.remove()
        graph2 = plt.plot(x,ten_avg,color = 'b')[0]
        tensum=0
        plt.pause(1)

    elif((i+1)%10==0):


        curr_ten_avg=tensum/10
        print("/n -----------------ADDING TENSUM = ", curr_ten_avg, "-------------------------/n")
        for i in range(10):
            ten_avg.append(curr_ten_avg)
        
        graph2.remove()
        graph2 = plt.plot(x,ten_avg,color = 'b')[0]
        tensum=0
        plt.pause(1)
    

    #else:
    #ten_avg.append(curr_ten_avg)

    
    #else:
    #hun_avg.append(curr_hun_avg)

	
	# plotting newer graph
    graph = plt.plot(x,y,color = 'g')[0]
    plt.xlim(max(x[-1]-150,x[0]), x[-1])
    curr = max(curr, y[-1])
    plt.ylim(0,curr+10)

    

env.close()






#------------------------------------------------------------------------------------------------------------------------------------------------------------











	