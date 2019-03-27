import numpy as np
import gym

env = gym.make('CartPole-v0')

bestLength = 0
episode_lengths = []

best_weights = np.zeros(4)


for i in range(100):
    new_weights = np.random.uniform(-1, 1, 4)

    length = []

    for j in range(100):
        observation = env.reset()
        done = False
        count = 0
        while not done:

            # env.render()
            count += 1
            action = 1 if np.dot(observation, new_weights) > 0 else 0

            observation, reward, done, _ = env.step(action)

            if done:
                break
        length.append(count)
    average_length = float(sum(length)/len(length))
    if average_length > bestLength:
        bestLength = average_length
        best_weights = new_weights
    episode_lengths.append(average_length)
    print("best length is ", bestLength)


observation = env.reset()
done = False
count = 0
while not done:

    env.render()
    count += 1
    action = 1 if np.dot(observation, best_weights) > 0 else 0

    observation, reward, done, _ = env.step(action)

    if done:
        break
print("lasted ", count)
env.close()
