try:
    import tensorflow as tf
    from tensorflow.contrib.tensorboard.plugins import projector
except ImportError:
    pass

import numpy as np
import keras.callbacks
import os.path
from PIL import Image


class SaveImageCallback():
    def __init__(self, env, frequency, path: str ='', **kwargs: 'Passed to PIL.Image.save.'):
        if not hasattr(env, 'metadata'):
            raise ValueError(f'Environment {env} does not have a `.metadata` attribute.')

        if 'rgb_array' not in env.metadata.get('render.modes', []):
            raise ValueError(f'Render mode `rgb_array` not found in {env.metadata["render.modes"]}.')

        self._env = env
        self.path = path
        self._index = 0
        self._image_args = kwargs

    def on_step_start(self, **kwargs):
        assert 'episode' in kwargs
        assert 'step' in kwargs

        data = self._env.render(mode='rgb_array')
        Image.fromarray(data).save(os.path.join(self.path, f'e{kwargs["episode"]}_s{kwargs["step"]}.png'), **self._image_args)


class TensorBoardCallback(keras.callbacks.TensorBoard):

    def __init__(self, *args, **kwargs):
        super(TensorBoardCallback, self).__init__(*args, **kwargs)

        self.scalars = []

    def on_execution_start(self, **kwargs):
        super(TensorBoardCallback, self).on_train_begin()

    def write_metadata(self, metadata: dict, filename: str):
        with open(filename, 'w') as f:
            # Write header row
            keys = list(metadata.keys())
            f.write('\t'.join(keys) + '\n')

            for i in range(metadata[keys[0]].shape[0]):
                row = '\t'.join([str(np.asscalar(metadata[k][i])) for k in keys]) + '\n'
                f.write(row)

    def add_embeddings(self, inputs, metadata: dict ={}, sprites: dict ={}):
        """

        :param inputs:
        :param metadata: tf.Variable.name: {Label: values}
        :param sprites:  tf.Variable.name: (image, thumbnail dimensions)
        :return:
        """
        assert isinstance(metadata, dict)
        assert isinstance(sprites, dict)

        inputs = list(inputs)
        self.saver = tf.train.Saver(inputs)

        config = projector.ProjectorConfig()
        for input in inputs:
            assert isinstance(input, tf.Variable)
            self.sess.run(input.initializer)
            embedding = config.embeddings.add()
            embedding.tensor_name = input.name
            name = input.name.split(':')[0]  # Colons mess up file names

            if input.name in metadata.keys():
                embedding.metadata_path = os.path.join(os.path.realpath(self.log_dir), f'metadata_{name}_0.tsv')
                self.write_metadata(metadata[input.name], embedding.metadata_path)

            if input.name in sprites.keys():
                assert len(sprites[input.name]) == 2, 'Must provide a tuple of size 2 containing the Image and thumbnail dimensions.'
                embedding.sprite.image_path = os.path.join(os.path.realpath(self.log_dir), f'sprite_{name}.png')

                t0 = sprites[input.name][0]
                t0.save(embedding.sprite.image_path)
                embedding.sprite.single_image_dim.extend(list(sprites[input.name][1]))
                sprites[input.name][0].save(embedding.sprite.image_path)

        projector.visualize_embeddings(self.writer, config)

        self.saver.save(self.sess, os.path.join(os.path.realpath(self.log_dir), 'embeddings'))



    def on_episode_end(self, **kwargs):
        episode = kwargs.get('episode', None)
        assert episode is not None, 'Key `episode` not found in kwargs.'

        if not self.validation_data and self.histogram_freq:
            raise ValueError('If printing histograms, validation_data must be '
                             'provided, and cannot be a generator.')

        if self.validation_data and self.histogram_freq:
            if episode % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)
                assert len(val_data) == len(tensors)

                feed_dict = dict(zip(tensors, val_data))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, episode)

        for name, value in kwargs.items():
            if name not in self.scalars:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, episode)
        self.writer.flush()


    def on_execution_end(self, **kwargs):
        super().on_train_end() # Let Keras callback perform cleanup.

    def on_train_end(self, **kwargs):
        # Keras parent class defiles on_train_end but it performs final cleanup and is intended to be called
        # when execution is terminating.  Override and do nothing.
        pass