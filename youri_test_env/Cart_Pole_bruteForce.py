import gym
import numpy as np
import random

env_name = 'CartPole-v0'
model_name = env_name + " model"
env = gym.make(env_name)

numEpisodes = 100


def main():
    bestQ = np.random.rand(4)
    bestScore = 0
    for episode in range(numEpisodes):
        done = False
        state = env.reset()
        totalscore = 0
        Q = np.random.rand(4)
        while not done:
            res = np.dot(Q, state)
            action = 1 if res > 0 else 0
            state_, reward, done, info = env.step(action)
            state = state_
            totalscore += reward
        print("ended episode ", episode, " with score ", totalscore, " actual best is ", bestScore)
        if totalscore > bestScore:
            bestQ = Q
            bestScore = totalscore
        if bestScore >= 200:
            print("Found best at episode ", episode)
            break
    print("done")
    return Q


def play(numGames=1):
    for game in range(numGames):
        state = env.reset()
        env.render()
        done = False
        totalscore = 0
        while not done:
            res = np.dot(Q, state)
            action = 1 if res > 0 else 0
            state_, reward, done, info = env.step(action)
            state = state_
            totalscore += reward
            env.render()
        print("ended game ", game, " with score ",
              totalscore)


Q = main()
play(10)
