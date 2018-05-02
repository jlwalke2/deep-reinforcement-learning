import os
import gym
import numpy as np
import unittest
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


class Test(unittest.TestCase):

    def test_embeddings(self):

        dir = os.path.join(os.getcwd(), 'embeddings')
        ckpt_file = os.path.join(dir, 'test.ckpt')
        metadata_file = os.path.join(dir, 'metadata.tsv')
        sprite_file = os.path.join(dir, 'thumbnails.png')

        if os.path.exists(dir):
            os.remove(dir)
        os.mkdir(dir)

        thumbnail_size = (50,50)

        from ..utils.misc import RandomSample
        sample = RandomSample(gym.make('LunarLander-v2'))
        sample.run(sample_size=1000, thumbnail_size=thumbnail_size)

        with open(metadata_file, 'w') as f:
            f.write('Action\tReward\n')
            for i in range(sample.actions.shape[0]):
                f.write(f'{np.asscalar(sample.actions[i])}\t{np.asscalar(sample.rewards[i])}\n')

        tf_states = tf.Variable(sample.states, name='States')

        sample.sprite.save(sprite_file)

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

