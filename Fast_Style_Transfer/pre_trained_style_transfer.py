import tensorflow as tf
#from Fast_style_transfer.utils import load_image
import NN_package as nn_package
import numpy as np
from functools import reduce
from keras.applications import imagenet_utils
import vgg_model as vgg
from sys import stdout
import preprocess_util as utils




class LostFunctions:        
    def style_loss(style_layers, style_targets, style_weight, num_style_layers):
        style_outputs = style_layers
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                            for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers
        return style_loss

    def content_loss(content_layers, content_targets, content_weight, num_content_layers):
        content_outputs = content_layers
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]- content_targets[name])**2) 
                                for name in content_outputs.keys()])
        content_loss *= content_weight /num_content_layers
        return content_loss
   

    #Variational loss L1 loss
    def total_variation_loss(image, total_variation_weight):
        x_deltas, y_deltas = image[:, :, 1:, :] - image[:, :, :-1, :], image[:, 1:, :, :] - image[:, :-1, :, :]
        return total_variation_weight*(tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas)))

    #Variation denoising
    def high_pass_x_y(image):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
        return x_var, y_var

    

def loss_process(test, content_layers, style_layers, content_targets, style_targets, content_weight, style_weight, tv_weight, batch_size):
    
    num_style_layers = len(style_layers)
    style_loss = LostFunctions.style_loss(style_layers, style_targets, style_weight, num_style_layers)/batch_size

    #Content Loss
    num_content_layers = len(content_layers)
    content_loss = LostFunctions.content_loss(content_layers, content_targets, content_weight, num_content_layers)/batch_size

    #Total variation loss
    tv_loss = LostFunctions.total_variation_loss(test, tv_weight)/batch_size
    loss = content_loss  + style_loss + tv_loss
    return loss


def optimize_style_transfer(content_training_images, style_image, content_weight, style_weight, tv_weight,epochs = 2, print_iterations = 100, batch_size=4, save_path = '', learning_rate = 0.001, content_path= '', device = '/cpu:0'):
    #Content images for training
    with tf.device(device):
        mod = len(content_training_images) % batch_size
        if mod > 0:
            print("Train set has been trimmed slightly..")
            content_training_images = content_training_images[:-mod]
        
        
        batch_shape = (batch_size, 256, 256, 3)
        style_shape = tf.squeeze(style_image).shape
        
       
        model = nn_package.stylelizer(style_shape, batch_size)

        #Style content extractor
        content_layers = ['block5_conv2'] 
        style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']
        extractor = vgg.StyleContentModel(style_layers, content_layers)
        style_target = extractor(style_image)['style']
        opt = tf.keras.optimizers.Adam(learning_rate)
       
        for epoch in range(epochs):
            iterations = 0
            num_examples = len(content_training_images)
            print(iterations)
            while iterations * batch_size < num_examples:
                curr = iterations * batch_size
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_training_images[curr:step]):
                    content_image = utils.preprocess(utils.preprocess_image(utils.load_img(content_path+img_p),256))
                    X_batch[j] = content_image
                assert X_batch.shape[0] == batch_size
                content_targets = extractor(X_batch)['content']
                with tf.GradientTape() as tape:
                    test = model(X_batch, training = True)
                    loss_value = loss_process(test, extractor(test)['content'], extractor(test)['style'], content_targets, style_target, content_weight, style_weight, tv_weight, batch_size)
                grads = tape.gradient(loss_value, model.trainable_weights)
                opt.apply_gradients(zip(grads, model.trainable_weights))
                print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                iterations += 1
        return model




