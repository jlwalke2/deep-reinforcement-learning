from keras.layers import Dense, Input, concatenate, Lambda
from keras.models import Sequential, Model
from keras.optimizers import sgd
from keras.regularizers import l2
import keras.backend as K
from deeprl.memories import Memory
import logging, random as rnd, numpy as np, gym, gym.spaces
from gym.envs.registration import register, spec
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().handlers[0].setLevel(logging.INFO)
logging.getLogger().addHandler(logging.FileHandler('ddpg_tf.log'))
logging.getLogger().handlers[1].setLevel(logging.DEBUG)

from collections import deque
debug_queue = deque(maxlen=500)
DEBUG = False

np.seterr(all='raise')

# Validate actor input & output
# Validate critic input & output
# Clone actor/critic for targets
# Policy / choose action





"""
TODO:
        Actor-Critic PG?  DDPG?
        Single continuous output
            Mask = 1 always
        Multiple continuous output
            Mask??  Can't sum to get objective output?  What does TORCS DDPG do?
        Add target critic & target actor models
        Enable clipping of action values?

        Custom environments?
"""

def create_actor_and_critic(num_feaures, num_actions):
    shared_h1 = Dense(16, activation='linear', name='SharedH1')


    actor_in = Input(shape=(num_features,), name='ActorIn')
    actor_h1 = shared_h1(actor_in)
    actor_out = Dense(num_actions, activation='linear', name='ActorOut')(actor_h1)     # Pendulum
    actor = Model(inputs=[actor_in], outputs=[actor_out])

    grad_q_wrt_a = Input(shape=(num_actions,), name='QFuncGrad')
    actor_grads = tf.gradients(actor_out, actor.trainable_weights, -grad_q_wrt_a)
    actor_grads = [tf.clip_by_value(grad, -1., 1.) for grad in actor_grads] # Clip gradient
    updates = zip(actor_grads, actor.trainable_weights)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).apply_gradients(updates)
    K.get_session().run(tf.global_variables_initializer())

    def train_func(state, grad):
        K.get_session().run(optimizer, feed_dict={
            actor_in: state,
            grad_q_wrt_a: grad
        })



    critic_state_input = Input(shape=(num_features,), name='CriticStateIn')
    critic_action_input = Input(shape=(num_actions,), name='CriticActionIn')
    critic_merged_input = concatenate([critic_state_input, critic_action_input])
    critic_h1 = shared_h1(critic_merged_input)
    critic_out = Dense(1, activation='linear', name='CriticOut')(critic_h1)
    critic = Model(inputs=[critic_state_input, critic_action_input], outputs=[critic_out])

    def loss(y_true, y_pred):
        err = K.clip(K.square(y_true - y_pred), 1e-30, 1e10)
        return K.mean(err, axis=-1)


    critic.compile('sgd', loss)

    return actor, train_func, critic, critic.train_on_batch



def create_actor(num_features, num_actions):
    actor_in = Input(shape=(num_features,), name='ActorIn')
    actor_h1 = Dense(400, activation='relu', name='ActorH1')(actor_in)
    actor_h2 = Dense(300, activation='relu', name='ActorH2')(actor_h1)
    actor_out = Dense(num_actions, activation='tanh', name='ActorOut')(actor_h2)     # Pendulum
    actor = Model(inputs=[actor_in], outputs=[actor_out])

    grad_q_wrt_a = Input(shape=(num_actions,), name='QFuncGrad')
    actor_grads = tf.gradients(actor_out, actor.trainable_weights, -grad_q_wrt_a)
    actor_grads = [tf.clip_by_value(grad, -1., 1.) for grad in actor_grads] # Clip gradient
    updates = zip(actor_grads, actor.trainable_weights)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).apply_gradients(updates)
    K.get_session().run(tf.global_variables_initializer())

    def train_func(state, grad):
        K.get_session().run(optimizer, feed_dict={
            actor_in: state,
            grad_q_wrt_a: grad
        })

    return actor, train_func



def create_critic(num_features, num_actions):
    critic_state_input = Input(shape=(num_features,), name='CriticStateIn')
    critic_action_input = Input(shape=(num_actions,), name='CriticActionIn')
    critic_merged_input = concatenate([critic_state_input, critic_action_input])
    critic_h1 = Dense(400, activation='relu', name='CriticH1', kernel_regularizer=l2(1e-2), bias_regularizer=l2(1e-2))(critic_merged_input)
    critic_h2 = Dense(300, activation='relu', name='CriticH2', kernel_regularizer=l2(1e-2), bias_regularizer=l2(1e-2))(critic_h1)
    critic_out = Dense(1, activation='linear', name='CriticOut', kernel_regularizer=l2(1e-2), bias_regularizer=l2(1e-2))(critic_h2)
    critic = Model(inputs=[critic_state_input, critic_action_input], outputs=[critic_out])

    # def loss(y_true, y_pred):
    #     err = K.clip(K.square(y_true - y_pred), 1e-30, 1e10)
    #     return K.mean(err, axis=-1)


    critic.compile(sgd(lr=1e-3, clipnorm=5.), 'mse')

    return critic, critic.train_on_batch




env = gym.make('Pendulum-v0')
is_continuous = isinstance(env.action_space, gym.spaces.Box)

def get_space_dim(space):
    if isinstance(space, gym.spaces.Box):
        return space.shape[0]
    elif isinstance(space, gym.spaces.Discrete):
        return space.n

num_features = get_space_dim(env.observation_space)
num_actions = get_space_dim(env.action_space)

# actor, actor_train_func, critic, critic_train_func = create_actor_and_critic(num_features, num_actions)

actor, actor_train_func = create_actor(num_features, num_actions)
critic, critic_train_func = create_critic(num_features, num_actions)
actor_target = Model.from_config(actor.get_config())
actor_target.compile(loss='mse', optimizer='sgd')
critic_target = Model.from_config(critic.get_config())
critic_target.compile(loss='mse', optimizer='sgd')

#grad_q_wrt_a = tf.clip_by_value(tf.gradients(critic.output, critic.inputs[1]), -1., 1.)
grad_q_wrt_a = tf.gradients(critic.output, critic.inputs[1])

memory = Memory(maxlen=10**6, sample_size=64)
gamma = 0.99
epsilon = 10.
min_epsilon = 0.5
warmup_steps = 0
max_episodes = 1000
max_steps_per_episode = 500
max_steps = 2500000
s1 = env.reset()
tau = 1e-3

"""
From Continuous Control with Deep Reinforcement Learning
https://arxiv.org/pdf/1509.02971.pdf

* Actor lr = 1e-4
* Critic lr = 1e-3
* Q network included L2 weight decay of 10^-2
* Discount factor = 0.99
* soft target update = 0.001
* ReLu for all hidden layers
* Actor output = tanh layer
* 2 hidden layers: 400 & 300 units
Actions not included until 2nd hidden layer of Q network
Weight initializations for output layers & non-output layers specified (see paper appendix)
* Minibatch size = 64
* memmory replay size = 10^6

Noise: theta = 0.15
Noise: sigma = 0.2
"""


# Ornstein-Uhlenbeck noise
# Theta & mu taken from DDPG paper
noise = 0       # Initial noise value
theta = 0.15
sigma = 0.2
mu = 0.

try:
    total_steps = 0
    for episode in range(1, max_episodes+1):
        s = env.reset()
        s = np.asarray(s).reshape((1, -1))
        episode_return = 0

        for step in range(1, max_steps_per_episode+1):
            total_steps += 1

            if episode % 1 == 0:
                env.render()

            # Generate an action from the policy network
            choice = actor.predict_on_batch([s]).reshape((1, -1))  # pi(s)
            if np.any(np.isnan(choice)):
                debug_queue.append(f'Action chosen by Actor was NaN: {choice}')
                debug_queue.append(str(actor.get_weights()))
                print(f'Action is NaN: {choice}')

            noise = noise + theta * (mu - noise) + sigma * np.random.randn()

            # Allow for periodic noiseless episodes
            if episode % 10 != 0:
                choice += noise

            # Take action and observe transition + reward
            s_prime, r, done, _ = env.step(choice)
            s_prime = np.asarray(s_prime).reshape((1, -1))
            episode_return += r

            memory.append((s, choice, r, s_prime, done))

            if step == max_steps_per_episode:
                done = True

            # Sample experiences for replay
            s_batch, a_batch, r_batch, s_prime_batch, episode_done = memory.sample()

            s_predicted_val = critic.predict_on_batch([s_batch, a_batch])                       # Predicted Q(s, a)
            s_prime_action = actor_target.predict_on_batch(s_prime_batch)                       # pi(s')
            s_prime_values = critic_target.predict_on_batch([s_prime_batch, s_prime_action])    # Expected Q(s', a')
            s_observed_val = gamma * s_prime_values
            s_observed_val[episode_done] = 0.                                       # R(s') = 0 if s' is terminal
            s_observed_val += r_batch                                               # Observed reward

            # Get gradient of Q value wrt action taken
            grad_tf = K.get_session().run(grad_q_wrt_a, feed_dict={
                critic.inputs[0]: s_batch,
                critic.inputs[1]: a_batch
            })[0]   # Returns list of 1 gradient
            if DEBUG:
                debug_queue.append(f'Current Step:  {total_steps}')
                debug_queue.append(f'S:  {s_batch}')
                debug_queue.append(f'A:  {a_batch}')
                debug_queue.append(f'Pred vs Obs Q(s, a):  ')
                for p, o in zip(critic.predict_on_batch([s_batch, a_batch]), s_predicted_val):
                    debug_queue.append(f'{p}  {o}  {p-o}')

            # Update TF actor weights
            err_actor = actor_train_func(s_batch, grad_tf)
            err_critic = critic_train_func([s_batch, a_batch], s_observed_val)

            print(f'E: {episode}  S: {step}  MSE: {err_critic}   Mean Diff: {np.mean(s_predicted_val - s_observed_val)}   Buffer:{len(memory)}')
            if DEBUG:
                debug_queue.append(f'Critic Training loss:  {err_critic}')

            if err_critic > 1000:
                print('uh oh!')
                for p, o in zip(critic.predict_on_batch([s_batch, a_batch]), s_predicted_val):
                    print(f'{p}  {o}  {p-o}')


            # Update target models
            # Soft model update
            w_t = critic_target.get_weights()
            w_o = critic.get_weights()
            for layer in range(len(w_o)):
                w_t[layer] = tau * w_o[layer] + (1.0 - tau) * w_t[layer]
            critic_target.set_weights(w_t)

            w_t = actor_target.get_weights()
            w_o = actor.get_weights()
            for layer in range(len(w_o)):
                w_t[layer] = tau * w_o[layer] + (1.0 - tau) * w_t[layer]
            actor_target.set_weights(w_t)

            if DEBUG:
                debug_queue.append(f'Gradient Q wrt A:  {grad_tf}')
                debug_queue.append(f'Hidden Kernel:  {w_o[0]}')
                debug_queue.append(f'Hidden Bias:  {w_o[1]}')
                debug_queue.append(f'Output Kernel:  {w_o[2]}')
                debug_queue.append(f'Output Bias:  {w_o[3]}')

            if done:
                print(f'{episode}:  {episode_return}') # Print terminal state number
                s = env.reset()
                break
            else:
                s = s_prime

except Exception as e:
    logging.exception(e)

env.render()


for message in debug_queue:
    logging.debug(message)

pass