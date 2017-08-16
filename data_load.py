from hyperparams import Hyperparams as hp
import numpy as np
from prepro import *
import tensorflow as tf
from utils import *
from sg_queue import producer_func


def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        texts, sound_files = load_train_data() # byte, string

        # calc total batch count
        num_batch = len(texts) // hp.batch_size

        # Convert to tensor
        texts = tf.convert_to_tensor(texts)
        sound_files = tf.convert_to_tensor(sound_files)

        # Create Queues
        text, sound_file = tf.train.slice_input_producer([texts, sound_files], shuffle=True)

        @producer_func
        def get_text_and_spectrograms(_inputs):
            '''From `_inputs`, which has been fetched from slice queues,
               makes text, spectrogram, and magnitude,
               then enqueue them again.
            '''
            _text, _sound_file = _inputs

            # Processing
            _text = np.fromstring(_text, np.int32) # byte to int
            _spectrogram, _magnitude = get_spectrograms(_sound_file)

            _spectrogram = reduce_frames(_spectrogram, hp.win_length//hp.hop_length, hp.r)
            _magnitude = reduce_frames(_magnitude, hp.win_length//hp.hop_length, hp.r)

            return _text, _spectrogram, _magnitude

        # Decode sound file
        x, y, z = get_text_and_spectrograms(inputs=[text, sound_file],
                                            dtypes=[tf.int32, tf.float32, tf.float32],
                                            capacity=128,
                                            num_threads=32)

        # create batch queues
        x, y, z = tf.train.batch([x, y, z],
                                shapes=[(None,), (None, hp.n_mels*hp.r), (None, (1+hp.n_fft//2)*hp.r)],
                                num_threads=32,
                                batch_size=hp.batch_size,
                                capacity=hp.batch_size*32,
                                dynamic_pad=True)

        if hp.use_log_magnitude:
            z = tf.log(z+1e-10)

    return x, y, z, num_batch
