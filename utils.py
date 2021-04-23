import tensorflow as tf
import numpy as np
from glob import glob
from random import randint
from librosa import load
noise_files = glob("data/noise/*")

def get_audio(path):
    audio, _ = tf.audio.decode_wav(tf.io.read_file(path), 1)
    return audio


def add_noise_to_clean_audio(clean_audio, noise_signal):
    """Adds noise to an audio sample"""
    if len(clean_audio) >= len(noise_signal):
        while len(clean_audio) >= len(noise_signal):
            noise_signal = np.append(noise_signal, noise_signal)

    ## Extract a noise segment from a random location in the noise file
    ind = np.random.randint(0, noise_signal.size - clean_audio.size)

    noise_segment = noise_signal[ind: ind + clean_audio.size]

    speech_power = np.sum(clean_audio ** 2)
    noise_power = np.sum(noise_segment ** 2)
    noisy_audio = clean_audio + np.sqrt(speech_power / noise_power) * noise_segment
    return noisy_audio


def get_random_noisy_audio(clean_file, sample_rate=16000):
    clean_audio, _ = load(clean_file, sr=sample_rate)

    random_ix = randint(0, len(noise_files) - 1)
    noise_file = noise_files[random_ix]
    noise_audio, _ = load(noise_file, sr=sample_rate)

    noisy_audio = add_noise_to_clean_audio(clean_audio=clean_audio, noise_signal=noise_audio)

    return noisy_audio, clean_audio


def inference_preprocess(path, batching_size=12000):
    audio = get_audio(path)
    audio_len = audio.shape[0]
    batches = []
    for i in range(0, audio_len - batching_size, batching_size):
        batches.append(audio[i:i + batching_size])

    batches.append(audio[-batching_size:])
    diff = audio_len - (i + batching_size)  # Calculation of length of remaining waveform
    return tf.stack(batches), diff


def predict(path, model):
    test_data, diff = inference_preprocess(path)
    predictions = model.predict(test_data)
    final_op = tf.reshape(predictions[:-1], (
    (predictions.shape[0] - 1) * predictions.shape[1], 1))  # Reshape the array to get complete frames
    final_op = tf.concat((final_op, predictions[-1][-diff:]), axis=0)  # Concat last, incomplete frame to the rest
    return final_op


def get_dataset(x_train, y_train):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(100).batch(64, drop_remainder=True)
    return dataset