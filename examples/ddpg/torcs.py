import gym
import keras

# Pixel observation space (96x96z3)
# Continous action space (1x3) w/ different ranges
env = gym.make('CarRacing-v0')