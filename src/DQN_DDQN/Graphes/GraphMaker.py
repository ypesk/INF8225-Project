import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.signal import savgol_filter

LunarLanderDQN = pickle.load(open("LunarLander-v2/objs_dqn.pkl", "rb"))
LunarLanderDDQN = pickle.load(open("LunarLander-v2/objs_ddqn.pkl", "rb"))

[EpsDQN, ScoreDQN, FramesDQN, AvgScoreDQN] = LunarLanderDQN
[EpsDDQN, ScoreDDQN, FramesDDQN, AvgScoreDDQN] = LunarLanderDDQN

newFramesDQN = savgol_filter(FramesDQN, 55, 3)
newFramesDDQN = savgol_filter(FramesDDQN, 55, 3)

fig, ax1 = plt.subplots()

ax1.set_xlabel('Episodes')
ax1.set_ylabel('Score moyen sur les 100 derniers episodes', color="blue")
ax1.plot(AvgScoreDDQN, color="blue")
ax1.tick_params(axis='y', labelcolor="blue")

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Frames filtrées', color="green")  # we already handled the x-label with ax1
ax2.plot(newFramesDDQN, color="green")
ax2.tick_params(axis='y', labelcolor="green")

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
