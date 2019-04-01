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
