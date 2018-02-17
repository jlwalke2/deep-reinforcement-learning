import os
import gym
import numpy as np
import unittest
from PIL import Image
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from ..memories import TrajectoryMemory

class Test(unittest.TestCase):

    # Taken from: https://github.com/tensorflow/tensorflow/issues/6322
    def images_to_sprite(self, data):
        """Creates the sprite image along with any necessary padding

        Args:
          data: NxHxW[x3] tensor containing the images.

        Returns:
          data: Properly shaped HxWx3 image with any necessary padding.
        """
        if len(data.shape) == 3:
            data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
        data = data.astype(np.float32)
        min = np.min(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
        max = np.max(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)
        # Inverting the colors seems to look better for MNIST
        # data = 1 - data

        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, 0),
                   (0, 0)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant',
                      constant_values=0)
        # Tile the individual thumbnails into an image.
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                               + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        data = (data * 255).astype(np.uint8)
        return data

    def random_sample(self, env, sample_size, frame_skip:int =4, thumbnail_size=(32,32), sprite_file: str =None):
        memory = TrajectoryMemory(maxlen=sample_size)
        thumbnails = []
        s = env.reset()

        for step in range(sample_size * frame_skip):
            s = np.asarray(s)

            if isinstance(env.action_space, gym.spaces.Discrete):
                a = env.action_space.sample()
            else:
                raise TypeError(f'Action selection for action space of type {type(env.action_space)} is not defined.')

            s_prime, r, episode_done, _ = env.step(a)

            if step % frame_skip == 0:
                memory.append((s, a, r, s_prime, episode_done))

                if sprite_file is not None:
                    image = Image.fromarray(env.render(mode='rgb_array'))
                    thumbnails.append(np.asarray(image.resize(thumbnail_size)))

            if episode_done:
                s = env.reset()
            else:
                s = s_prime

        if sprite_file:
            sprite = self.images_to_sprite(np.asarray(thumbnails))
            Image.fromarray(sprite).save(sprite_file)

        return memory.sample()


    def test_embeddings(self):
        # Randomly sample trajectories from the environment

        dir = os.path.join(os.getcwd(), 'embeddings')
        ckpt_file = os.path.join(dir, 'test.ckpt')
        metadata_file = os.path.join(dir, 'metadata.tsv')
        sprite_file = os.path.join(dir, 'thumbnails.png')

        if os.path.exists(dir):
            os.remove(dir)
        os.mkdir(dir)

        thumbnail_size = (50,50)

        states, actions, rewards, s_primes, episode_done = self.random_sample(gym.make('LunarLander-v2'), 1000,
                                                                              thumbnail_size=thumbnail_size,
                                                                              sprite_file=sprite_file)

        with open(metadata_file, 'w') as f:
            f.write('Action\tReward\n')
            for i in range(actions.shape[0]):
                f.write(f'{np.asscalar(actions[i])}\t{np.asscalar(rewards[i])}\n')

        tf_states = tf.Variable(states, name='States')

        with tf.Session() as sess:
            saver = tf.train.Saver([tf_states])
            sess.run(tf_states.initializer)
            saver.save(sess, ckpt_file)

            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = tf_states.name
            embedding.sprite.image_path = sprite_file
            embedding.sprite.single_image_dim.extend(list(thumbnail_size))
            embedding.metadata_path = metadata_file
            projector.visualize_embeddings(tf.summary.FileWriter(dir), config)

