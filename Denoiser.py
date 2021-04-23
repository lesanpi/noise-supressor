import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv1D, Conv1DTranspose, Concatenate

def build_model(batching_size=12000):

    inp = Input(shape=(batching_size, 1))
    c1 = Conv1D(2, 32, 2, 'same', activation='relu')(inp)
    c2 = Conv1D(4, 32, 2, 'same', activation='relu')(c1)
    c3 = Conv1D(8, 32, 2, 'same', activation='relu')(c2)
    c4 = Conv1D(16, 32, 2, 'same', activation='relu')(c3)
    c5 = Conv1D(32, 32, 2, 'same', activation='relu')(c4)

    dc1 = Conv1DTranspose(32, 32, 1, padding='same')(c5)
    conc = Concatenate()([c5, dc1])
    dc2 = Conv1DTranspose(16, 32, 2, padding='same')(conc)
    conc = Concatenate()([c4, dc2])
    dc3 = Conv1DTranspose(8, 32, 2, padding='same')(conc)
    conc = Concatenate()([c3, dc3])
    dc4 = Conv1DTranspose(4, 32, 2, padding='same')(conc)
    conc = Concatenate()([c2, dc4])
    dc5 = Conv1DTranspose(2, 32, 2, padding='same')(conc)
    conc = Concatenate()([c1, dc5])
    dc6 = Conv1DTranspose(1, 32, 2, padding='same')(conc)
    conc = Concatenate()([inp, dc6])
    dc7 = Conv1DTranspose(1, 32, 1, padding='same', activation='linear')(conc)

    model = tf.keras.models.Model(inp, dc7)
    # model.compile(optimizer=tf.keras.optimizers.Adam(0.002), loss=tf.keras.losses.MeanAbsoluteError())

    return model
