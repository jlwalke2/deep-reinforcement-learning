"""
Playing Atari with Deep Reinforcement Learning (https://arxiv.org/pdf/1312.5602v1.pdf)

Input: raw pixel vector x_t, and reward, r_t

Atari frames are 210x160 pixels w/ 128 color palette


Represents the state at time t, s_t as the sequence of all observations and rewards up to t.  s_t = x1, a1, x2, a2, x_t-1, a_t-1, x_t
Phi(s_t) = preprocessed sequence.
Phi applies the following transformations & then stacks the last 4 frames to produce the input to Q-network:
 - convert to gray-scale
 - downsample to 110x84
 - crop to 84x84 region that captures the playing area.

 Architecture:
  - input: 84x84x4 image produced by Phi
  - 16 8x8 convolution filters w/ stride 4 + rectified nonlinearity
  - 32 4x4 convolution filters w/ stride 2 + rectified nonlinearity
  - 256 dense w/ rectifier
  - output = # actions (varied by game)

 Used reward shaping: all positive rewards -> +1 and all negative -> -1 to ensure even scale across games

 Training:
  - RMSProp w/ minibatches of size 32
  - epsilon-greedy annealed from 1.0 to 0.1 over first 1 million frames, then fixed at 0.1
  - trained for 10 million frames
  - replay buffer of 1 million most recent
  - action replay = 4 (3 for space invaders)


What is a training epoch? (plots)
    "One epoch corresponds to 50000 minibatch weight updates or roughly 30 minutes of training time"
"""