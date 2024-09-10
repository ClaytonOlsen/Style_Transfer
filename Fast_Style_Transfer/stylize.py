import tensorflow as tf
from tensorflow.keras.layers import Input
from NN_package import stylelizer

def ffwd(content, network_path):
    img_placeholder = tf.keras.Input(name = 'X_Content', shape = content.shape)

    network = stylelizer(img_placeholder)
    saver = tf.train.Checkpoint(network)

    ckpt = tf.train.get_checkpoint_state(network_path)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise Exception("No checkpoint found...")

    prediction = sess.run(network, feed_dict={img_placeholder:content})
    return prediction[0]