import tensorflow as tf
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


        self.writer.flush()


    def on_execution_end(self):
        self.on_train_end()

    def on_train_end(self, _):
        self.writer.close()