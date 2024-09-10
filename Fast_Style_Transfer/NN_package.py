import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

def stylelizer(style_shape, batch_size):
    X_content = tf.keras.Input(name = 'X_Content', shape = style_shape, batch_size = batch_size)
    image = X_content / 255.0
    # Block 
    conv1 = Conv2D(32, (3, 3), strides = (1,1), activation='relu', padding='same', name='conv1')(image)
    conv2 = Conv2D(64, (3, 3), strides = (2,2), activation='relu', padding='same', name='conv2')(conv1)
    conv3 = Conv2D(128, (3, 3), strides = (2,2), activation='relu', padding='same', name='conv3')(conv2)
    resid1 = residual_block(conv3, 3)
    resid2 = residual_block(resid1, 3)
    resid3 = residual_block(resid2, 3)
    resid4 = residual_block(resid3, 3)
    resid5 = residual_block(resid4, 3)
    conv_t1 = Conv2DTranspose(64, (3, 3), strides = (2,2), activation='relu', padding='same', name='conv_t1')(resid5)
    conv_t2 = Conv2DTranspose(32, (3, 3), strides = (2,2), activation='relu', padding='same', name='conv_t2')(conv_t1)
    conv_t3 = Conv2D(3, (9, 9), strides = (1,1), padding='same', name='conv_t3')(conv_t2)
    preds = tf.nn.tanh(conv_t3)
    output = image + preds
    outputs = tf.nn.tanh(output) * 127.5 + 255./2
    model = tf.keras.Model(X_content, outputs)
    return model

def residual_block(net, filter_size = 3):
    temp = Conv2D(128, (filter_size, filter_size), strides = (1,1), activation='relu', padding='same')(net)
    return net + Conv2D(128, (filter_size, filter_size), strides = (1,1), padding='same')(temp)

