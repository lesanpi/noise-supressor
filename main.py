import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError
from librosa import load
from tqdm import tqdm
from scipy.io.wavfile import write
from Denoiser import build_model
import datetime
import time
from glob import glob
from utils import *
from hparams import *

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

model = build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)


def wave_loss(wave_output, wave_target):
    loss = tf.reduce_mean(tf.abs(wave_target - wave_output))
    return loss


def spec_loss(wave_output, wave_target):
    wave_output = tf.squeeze(wave_output)
    wave_target = tf.squeeze(wave_target)

    spec_output = tf.signal.stft(wave_output, frame_length=255, frame_step=128)
    spec_output = tf.abs(spec_output)

    spec_target = tf.signal.stft(wave_target, frame_length=255, frame_step=128)
    spec_target = tf.abs(spec_target)

    loss = tf.reduce_mean(tf.abs(spec_target - spec_output))
    return loss


def denoiser_loss(wave_output, wave_target):
    w_loss = wave_loss(wave_output=wave_output, wave_target=wave_target)
    s_loss = spec_loss(wave_output=wave_output, wave_target=wave_target)

    total_loss = w_loss + (SPECT_LOSS_FACTOR * s_loss)
    return total_loss, w_loss, s_loss


@tf.function
def train_step(wave_input, wave_target, epoch):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        wave_output = model(wave_input, training=True)
        d_loss, w_loss, s_loss = denoiser_loss(wave_output=wave_output, wave_target=wave_target)

    gradients = tape.gradient(d_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('denoiser_total_loss', d_loss, step=epoch)
        tf.summary.scalar('wave_loss', w_loss, step=epoch)
        tf.summary.scalar('spec_loss', s_loss, step=epoch)

    return d_loss, w_loss, s_loss


def fit(train_ds, epochs):
    for epoch in range(epochs):
        start = time.time()

        print('EPOCH: {}'.format(epoch + 1))
        # Train
        denoiser_l, wave_l, spect_loss = 0, 0, 0
        for n, (wave_input, wave_target) in train_ds.enumerate():
            print('.', end='')
            if (n + 1) % 100 == 0:
                print()
            d_loss, w_loss, s_loss = train_step(wave_input, wave_target, epoch)
            denoiser_l, wave_l, spect_loss = d_loss, w_loss, s_loss
        print()
        print('Denoiser Loss: {}, Wave Loss: {}, Spect Loss: {}'.format(denoiser_l, wave_l, spect_loss))
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))


with tf.device('/CPU:0'):
    # Datashare Dataset
    clean_sounds = glob("data/clean_testset_wav/*")
    noisy_sounds = glob("data/noisy_testset_wav/*")

    clean_sounds_list, _ = tf.audio.decode_wav(tf.io.read_file(clean_sounds[0]), desired_channels=1)
    noisy_sounds_list, _ = tf.audio.decode_wav(tf.io.read_file(noisy_sounds[0]), desired_channels=1)

    if DATASHARE_FILES:
        for i in tqdm(clean_sounds[1:]):
            so, _ = tf.audio.decode_wav(tf.io.read_file(i), desired_channels=1)
            clean_sounds_list = tf.concat((clean_sounds_list, so), 0)

        for i in tqdm(noisy_sounds[1:]):
            so, _ = tf.audio.decode_wav(tf.io.read_file(i), desired_channels=1)
            noisy_sounds_list = tf.concat((noisy_sounds_list, so), 0)


if FLAC_FILES:
    # Speechlib Dataset + UrbanSound8k
    clean_audiofiles = glob("data/clean/*")
    noisy_audio, clean_audio = get_random_noisy_audio(clean_file=clean_audiofiles[0])
    noisy_audio = np.expand_dims(noisy_audio, -1)
    clean_audio = np.expand_dims(clean_audio, -1)

    for clean_file in tqdm(clean_audiofiles[1:]):
        noisy_audio, clean_audio = get_random_noisy_audio(clean_file=clean_file)

        noisy_audio = np.expand_dims(noisy_audio, -1)
        clean_audio = np.expand_dims(clean_audio, -1)

        noisy_sounds_list = tf.concat((noisy_sounds_list, noisy_audio), 0)
        clean_sounds_list = tf.concat((clean_sounds_list, clean_audio), 0)

print(clean_sounds_list.shape)  # 218086741, with flac audios
print(noisy_sounds_list.shape)

with tf.device('/CPU:0'):

    clean_train, noisy_train = [], []
    for i in tqdm(range(0, clean_sounds_list.shape[0] - BATCHING_SIZE, BATCHING_SIZE)):
        clean_train.append(clean_sounds_list[i:i + BATCHING_SIZE])
        noisy_train.append(noisy_sounds_list[i:i + BATCHING_SIZE])

    clean_train = tf.stack(clean_train)
    noisy_train = tf.stack(noisy_train)

    print("Noisy Train Shape:", noisy_train.shape, "Len:", len(noisy_train))
    # (18173, 12000, 1) with flac files


train_dataset = get_dataset(noisy_train[:], clean_train[:])
fit(train_dataset, epochs=EPOCHS)
