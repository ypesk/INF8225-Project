Deep Q learning
Experience replay

Deepmind papers on atari AI :
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf

Tuto très interessant
https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

Ici il explique que le DQN ne converge pas car il a tendance à surestimer les valeurs de Q
Le Double DQN peut aider
Un réseau pour choisir l'action et un pour générer les Q values pour cette action
il faut aussi les syncroniser de tps en tps
https://towardsdatascience.com/atari-reinforcement-learning-in-depth-part-1-ddqn-ceaa762a546f?fbclid=IwAR1Lp1978bvsgyDdOFUJ97sx6hSLCseUPXzjK_G2Y8sAqRfeQFOdU_1bVBI


Tuto en pleins de parties avec un peu tout (QL, DQL, Policy Gradient, Actor critip, PPO, etc)
https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe

Policy gradients (rapide ??)
https://medium.com/@gabogarza/deep-reinforcement-learning-policy-gradients-8f6df70404e6


DDPG (actor critic)
https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html#Tensorflow

Actor critic :
truc important, dans le modèle, la dernière couche est répétée pour chaque action
ça doit être ça qui fait qu'on peut faire du continu
https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html
