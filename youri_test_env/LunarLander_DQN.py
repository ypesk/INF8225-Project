import gym
from gym import wrappers
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten
import random
from collections import deque
import math
import cv2
import pickle

# CartPole-v1, LunarLander-v2, BipedalWalker-v2, CarRacing-v0, Riverraid-v0, MsPacman-v0
env_name = 'LunarLander-v2'
model_name = env_name + " model"
env = gym.make(env_name)
# input_dims = env.reset().shape

numEpisodes = 5000
discount = 0.9  # Ratio de discount des reward futures, correspond au gamma dans pas mal d'équations
batch_size = 50


def preprocessing(image):
    # image = image[::2, ::2]  # Downsample
    # image = np.mean(image, axis=2).astype(np.uint8)
    image = np.resize(image, (1, image.shape[0]))
    return image


def buildModel():

    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))

    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')
    return model


def learn(model, D):
    minibatch = random.sample(D, batch_size)

    x_batch = []
    y_batch = []
    for phi, action, reward, phi_, done in minibatch:
        y_target = model.predict(phi)
        if done:
            y_target[0][action] = reward
        else:
            y_target[0][action] = reward + discount * np.amax(model.predict(phi_)[0])
        x_batch.append(phi[0])
        y_batch.append(y_target[0])

    x_batch = np.resize(x_batch, (batch_size, 8))
    y_batch = np.resize(y_batch, (batch_size, env.action_space.n))

    # x_batch = np.resize(x_batch, (batch_size, input_dims))
    # y_batch = np.resize(y_batch, (batch_size, env.action_space.n))

    model.fit(x_batch, y_batch, verbose=0)
    return model


def train_model():
    model = buildModel()
    # On va mémoriser certaines variables a chaque épisode pour voir un peu leur progression
    # Pas encore utilisé
    listScore = []
    listEps = []
    listNumFrames = []
    eps = 1  # Proba de choisir une action random pour la phase d'exploration

    eps_update = 0.99999539484  # arrive a 0.1 en 500 000 frames
    # eps_update = 0.999997697418 #arrive a 0.1 en 1 millions de frames dans l'article
    # eps_update = 0.9997004716 #arrive a 0.05 en 10000 frames dans l'article

    D = deque(maxlen=20000)  # Replay memory (20 000 dernières frames)
    for episode in range(numEpisodes):
        print("starting episode ", episode, " with eps ", eps)
        done = False
        state = env.reset()
        phi = preprocessing(state)
        totalscore = 0
        numFrame = 1
        # env.render()
        nextActionIn = 0
        while not done:
            # env.render()
            numFrame += 1
            if (random.random() > eps):
                Q = model.predict(phi)
                action = np.argmax(Q)
            else:
                action = random.randint(0, env.action_space.n-1)
            state_, reward, done, info = env.step(action)
            totalscore += reward
            # if done:
            #     # On pénalise quand on fail, il parait que c'est mieux, j'aurais peut être pas du le faire pour le vaisseau
            #     if totalscore < 500:
            #         reward = -100
            phi_ = preprocessing(state_)
            experience = (phi, action, reward, phi_, done)
            D.append(experience)
            phi = phi_
            # env.render()
            if (len(D) > batch_size):
                model = learn(model, D)
                eps = eps*eps_update
            # On garde toujours une action random avec proba 0.05
            eps = 0.1 if eps < 0.1 else eps

        print("score ", totalscore, " at frame ", numFrame)
        listEps.append(eps)
        listScore.append(totalscore)
        listNumFrames.append(numFrame)
        model.save(model_name)
    print("done")


def play(numGames=1, record=True):
    model = keras.models.load_model(model_name)
    env = gym.make(env_name)

    if record:
        env = wrappers.Monitor(env, "./renders/"+env_name, force=True)
    for game in range(numGames):
        state = env.reset()
        env.render()
        phi = preprocessing(state)
        done = False
        totalscore = 0
        numFrame = 1
        while not done:
            numFrame += 1
            Q = model.predict(phi)
            action = np.argmax(Q)
            state_, reward, done, info = env.step(action)
            phi_ = preprocessing(state_)
            phi = phi_
            totalscore += reward
            env.render()
    env.close()


# train_model()
play(1, record=True)
