import gym
from gym import wrappers
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten
import random
from collections import deque
import math
import pickle

# CartPole-v1, LunarLander-v2, BipedalWalker-v2, CarRacing-v0, Riverraid-v0, MsPacman-v0
env_name = 'LunarLander-v2'
model_name = env_name + "_model_ddqn"
env = gym.make(env_name)
# input_dims = env.reset().shape

numEpisodes = 500
discount = 0.99  # Ratio de discount des reward futures, correspond au gamma dans pas mal d'équations
batch_size = 50
eps_min = 0.01


def preprocessing(image):
    # Only to have the right dimensions
    image = np.resize(image, (1, image.shape[0]))
    return image


def buildModel():
    model = Sequential()
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))

    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='mse')
    return model


def build_target_model(model):
    # Target Neural Net
    # target_model = Sequential()
    # target_model.add(Dense(16, activation='relu'))
    # target_model.add(Dense(16, activation='relu'))
    # target_model.add(Dense(16, activation='relu'))
    # target_model.add(Dense(env.action_space.n, activation='linear'))
    #
    # target_model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')
    target_model.set_weights(self.model.get_weights())
    return target_model


def target_q_value(next_state, target_model, model):
    # formule appliquée : Q_Target = r + γQ(s’,argmax(Q(s’,a,ϴ),ϴ’))
    # a'_max = argmax(Q(s’,a,ϴ),ϴ’) (DQN normal)
    action = np.argmax(model.predict(next_state)[0])
    # target : Q = Q_target(s',a'_max)
    q_value = target_model.predict(next_state)[0][action]

    return q_value


def update_target_model(target_model, model):
    # copy the weights of the Q network to the target network
    target_model.set_weights(model.get_weights())
    return target_model


def learn(model, target_model, D):

    # This is the experience replay function of DeepMind.
    # We take a minibatch from the memory D
    # we do a SGD on the tuple (state, action, reward)
    minibatch = random.sample(D, batch_size)

    x_batch = []
    y_batch = []
    for phi, action, reward, phi_, done in minibatch:
        y_target = model.predict(phi)
        if done:
            y_target[0][action] = reward
        else:
            y_target[0][action] = reward + discount*target_q_value(phi_, target_model, model)
        x_batch.append(phi[0])
        y_batch.append(y_target[0])

    x_batch = np.resize(x_batch, (batch_size, 8))
    y_batch = np.resize(y_batch, (batch_size, env.action_space.n))

    model.fit(x_batch, y_batch, verbose=0)
    return model


def train_model():
    model = buildModel()
    target_model = buildModel()
    #model = keras.models.load_model(model_name)

    # On va mémoriser certaines variables a chaque épisode pour voir un peu leur progression
    # Pas encore utilisé
    listScore = []
    listEps = []
    listNumFrames = []
    listAvgScore = []
    eps = 1  # Proba de choisir une action random pour la phase d'exploration
    eps_update = 0.995
    # eps_update = 0.999997697418 #arrive a 0.1 en 1 millions de frames dans l'article
    # eps_update = 0.9997004716 #arrive a 0.05 en 10000 frames dans l'article

    D = deque(maxlen=50000)  # Replay memory
    S = deque(maxlen=100)  # Score memory
    env = gym.make(env_name)
    for episode in range(numEpisodes):

        print("starting episode ", episode, " with eps ", eps)
        done = False
        # Uncomment to record one episode every 50 to check the progression
        # if (episode % 50 == 0):
        #     print("Recording this episode")
        #     env = wrappers.Monitor(env, "./renders/"+str(episode)+"_"+str(eps), force=True)
        #     env.render()
        state = env.reset()
        phi = preprocessing(state)
        totalscore = 0
        numFrame = 1
        while not done:
            numFrame += 1
            # Take a random action, based on epsilon value
            if (random.random() > eps):
                Q = model.predict(phi)
                action = np.argmax(Q)
            else:
                action = random.randint(0, env.action_space.n-1)
            state_, reward, done, info = env.step(action)
            totalscore += reward

            phi_ = preprocessing(state_)
            experience = (phi, action, reward, phi_, done)

            # Stock the experience in the memory for the experience replay
            D.append(experience)
            phi = phi_
            if (len(D) > batch_size):
                model = learn(model, target_model, D)

            # Uncomment to record one episode every 50 to check the progression
            # if (episode % 50 == 0):
            #     env.render()
        if (len(D) > batch_size):
            eps = eps*eps_update
        eps = eps_min if eps < eps_min else eps
        S.append(totalscore)
        if (episode % 5):
            target_model = update_target_model(target_model, model)
        print("score ", totalscore, " at frame ", numFrame,
              " average score on last 100 : ", np.mean(S))
        listEps.append(eps)
        listScore.append(totalscore)
        listNumFrames.append(numFrame)
        listAvgScore.append(np.mean(S))
        f = open('objs.pkl', 'wb')
        pickle.dump([listEps, listScore, listNumFrames, listAvgScore], f)
        f.close()
        model.save(model_name)
    print("done")


def play(numGames=1, record=True):
    # Play numGames with a trained model
    model = keras.models.load_model("models/LunarLander-v2_model_ddqn")
    # target_model = keras.models.load_model(model_name)
    env = gym.make(env_name)

    if record:
        env = wrappers.Monitor(env, "./renders/LunarLander-v2_DDQN_500", force=True)
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
        print("Game : ", game, ", score : ", totalscore)
    env.close()


train_model()
# play(1, record=True)
