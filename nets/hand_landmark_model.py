import tensorflow as tf

INPUT_SHAPE = (256, 256, 3)


def landmark_block(block_in, filters, max_pooling=False):
    if max_pooling:
        convolution = tf.keras.layers.Conv2D(filters=filters, kernel_size=(2, 2), strides=2, padding='same')(block_in)
        activation = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None)(
            convolution)
        d_convolution = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same')(activation)
        convolution = tf.keras.layers.Conv2D(filters=filters * 2, kernel_size=(1, 1), strides=1, padding='same')(
            d_convolution)

        max_pooling = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(block_in)
        input_channel = block_in.shape[-1]
        output_channel = convolution.shape[-1]
        if input_channel != output_channel:
            channel_pad = tf.keras.backend.concatenate([max_pooling, tf.zeros_like(max_pooling)], axis=-1)
            add = tf.keras.layers.add([convolution, channel_pad])
        else:
            add = tf.keras.layers.add([convolution, max_pooling])
    else:
        convolution = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1, padding='same')(block_in)
        activation = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None)(
            convolution)
        d_convolution = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same')(activation)
        convolution = tf.keras.layers.Conv2D(filters=filters * 2, kernel_size=(1, 1), strides=1, padding='same')(
            d_convolution)
        add = tf.keras.layers.add([convolution, block_in])

    block_out = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None)(add)
    return block_out


def build_hand_landmark_model(only_landmark=True):
    # size of max pooling is unknown
    x_in = tf.keras.layers.Input(shape=INPUT_SHAPE)
    convolution = tf.keras.layers.Conv2D(filters=16,
                                         kernel_size=(3, 3),
                                         strides=2,
                                         padding='same')(x_in)
    landmark_block_128 = tf.keras.layers.PReLU(alpha_initializer='zeros',
                                               alpha_regularizer=None,
                                               alpha_constraint=None)(convolution)
    for i in range(8):
        landmark_block_128 = landmark_block(landmark_block_128, filters=8)
    landmark_block_64 = landmark_block(landmark_block_128, filters=16, max_pooling=True)
    for i in range(8):
        landmark_block_64 = landmark_block(landmark_block_64, filters=16)
    landmark_block_32 = landmark_block(landmark_block_64, filters=32, max_pooling=True)
    for i in range(8):
        landmark_block_32 = landmark_block(landmark_block_32, filters=32)
    convolution = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), strides=2, padding='same')(landmark_block_32)
    activation = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None)(
        convolution)
    d_convolution = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same')(activation)
    convolution = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=1, padding='same')(d_convolution)
    max_pooling = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(landmark_block_32)
    channel_pad = tf.keras.backend.concatenate([max_pooling,
                                                tf.zeros_like(max_pooling),
                                                tf.zeros_like(max_pooling),
                                                tf.zeros_like(max_pooling)],
                                               axis=-1)
    add = tf.keras.layers.add([convolution, channel_pad])
    landmark_block_16 = tf.keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None)(
        add)
    for i in range(8):
        landmark_block_16 = landmark_block(landmark_block_16, filters=128)
    landmark_block_8 = landmark_block(landmark_block_16, filters=128, max_pooling=True)
    for i in range(8):
        landmark_block_8 = landmark_block(landmark_block_8, filters=128)
    landmark_block_4 = landmark_block(landmark_block_8, filters=128, max_pooling=True)
    for i in range(8):
        landmark_block_4 = landmark_block(landmark_block_4, filters=128)
    landmark_block_2 = landmark_block(landmark_block_4, filters=128, max_pooling=True)
    for i in range(8):
        landmark_block_2 = landmark_block(landmark_block_2, filters=128)
    hand_landmark = tf.keras.layers.Conv2D(filters=42, kernel_size=(2, 2), strides=2, padding='same')(landmark_block_2)
    hand_landmark = tf.keras.layers.Reshape((42,))(hand_landmark)
    if only_landmark:
        x_out = hand_landmark
    else:
        hand_presence = tf.keras.layers.Conv2D(filters=1, kernel_size=(2, 2), strides=2, padding='same')(landmark_block_2)
        hand_presence = tf.keras.layers.Activation('sigmoid')(hand_presence)
        hand_presence = tf.keras.layers.Reshape((1,))(hand_presence)
        x_out = tf.concat([hand_presence, hand_landmark], axis=-1)

    model = tf.keras.Model(inputs=x_in, outputs=x_out)
    return model
