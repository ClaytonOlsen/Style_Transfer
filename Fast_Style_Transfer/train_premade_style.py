
from argparse import ArgumentParser
import preprocess_util as util
import numpy as np
import tensorflow as tf
from pre_trained_style_transfer import optimize_style_transfer
import os
#DEFAULTS
DEFAUTLS = {
    'CONTENT_WEIGHT': 1e4,
    'STYLE_WEIGHT':1e-4,
    'TV_WEIGHT':30,
    'LEARNING_RATE': 1e-3,
    'NUM_EPOCHS':50,
    'BATCH_SIZE':1,
    'MODEL_PATH' : 'imagenet-vgg-verydeep-19.mat',
    'CHECKPOINT_ITERATIONS': 100,
    'SAVE_PATH':'./pre_made_style_transfer/model_weights',
    'CONTENT_PATH': './pre_made_style_transfer'
}



def train_style_model(style_path:str, content_path:str, save_path = DEFAUTLS['SAVE_PATH'], 
                        epochs = DEFAUTLS['NUM_EPOCHS'], batch_size = DEFAUTLS['BATCH_SIZE'], 
                        checkpoints = DEFAUTLS['CHECKPOINT_ITERATIONS'], model_path = DEFAUTLS['MODEL_PATH'], 
                        content_weight = DEFAUTLS['CONTENT_WEIGHT'], style_weight = DEFAUTLS['STYLE_WEIGHT'],
                        tv_weight = DEFAUTLS['TV_WEIGHT'], learning_rate = DEFAUTLS['LEARNING_RATE'], use_gpu = False):

    #Load and reshape style image
    style_image = util.load_img(style_path)
    #blurr style image a bit to ignore some content capture
    style_image = tf.nn.avg_pool(style_image, ksize = [3,3], strides = [1,1], padding = 'SAME')
    style_image = util.preprocess_image(style_image, 256)

    #Get the shape of the content images
    content_training_images = os.listdir(content_path)
    content_image = util.load_img(content_path + content_training_images[0])
    content_shape = tf.squeeze(util.preprocess_image(content_image, 256)).shape

    #checks if gpu is available
    device = '/gpu:0' if use_gpu else '/cpu:0'
    
    # Calls style transer model using the path to the VGG19 model, the style image, the shape of the content tensor (probably to match the input with the size of the content image), and the weights
    
    
    
    kwargs = {
        "epochs":epochs,
        "print_iterations":checkpoints,
        "batch_size":batch_size,
        "save_path":save_path,
        "learning_rate":learning_rate,
        "content_path":content_path,
        "device": device
    }

    args = [
        content_training_images,
        style_image,
        content_weight,
        style_weight,
        tv_weight
        ]
    model = optimize_style_transfer(*args, **kwargs)
    return model
    '''
    #Style_trasfer contains the losses and the .train() function that
    for losses, i, epoch in optimize_style_transfer(*args, **kwargs):
        style_loss, content_loss, tv_loss, loss = losses
        print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
        to_print = (style_loss, content_loss, tv_loss)
        print('style: %s, content:%s, tv: %s' % to_print)
    '''

     