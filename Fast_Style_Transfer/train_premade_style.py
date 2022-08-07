
from argparse import ArgumentParser
import preprocess_util as util
import numpy as np
import tensorflow as tf
from pre_trained_style_transfer import optimize_style_transfer, reoptimize_style_transfer
import os
#DEFAULTS
DEFAUTLS = {
    'CONTENT_WEIGHT': 1e3,
    'STYLE_WEIGHT':1e-1,
    'TV_WEIGHT':10,
    'LEARNING_RATE': 1e-4,
    'NUM_EPOCHS':100,
    'BATCH_SIZE':256,
    'MODEL_PATH' : 'imagenet-vgg-verydeep-19.mat',
    'CHECKPOINT_ITERATIONS': 25,
    'SAVE_PATH':'./pre_made_style_transfer/model_weights',
    'CONTENT_PATH': './pre_made_style_transfer'
}



def train_style_model(style_path:str, content_path:str, save_path = DEFAUTLS['SAVE_PATH'], 
                        epochs = DEFAUTLS['NUM_EPOCHS'], batch_size = DEFAUTLS['BATCH_SIZE'], 
                        checkpoints = DEFAUTLS['CHECKPOINT_ITERATIONS'], model_path = DEFAUTLS['MODEL_PATH'], 
                        content_weight = DEFAUTLS['CONTENT_WEIGHT'], style_weight = DEFAUTLS['STYLE_WEIGHT'],
                        tv_weight = DEFAUTLS['TV_WEIGHT'], learning_rate = DEFAUTLS['LEARNING_RATE'], use_gpu = True):

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
    print(device)
    
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
    model.compile(optimizer='adam')
    return model


def retrain_style_model(style_path:str, content_path:str, partly_trained:str, save_path = DEFAUTLS['SAVE_PATH'], 
                    epochs = DEFAUTLS['NUM_EPOCHS'], batch_size = DEFAUTLS['BATCH_SIZE'], 
                    checkpoints = DEFAUTLS['CHECKPOINT_ITERATIONS'], model_path = DEFAUTLS['MODEL_PATH'], 
                    content_weight = DEFAUTLS['CONTENT_WEIGHT'], style_weight = DEFAUTLS['STYLE_WEIGHT'],
                    tv_weight = DEFAUTLS['TV_WEIGHT'], learning_rate = DEFAUTLS['LEARNING_RATE'], use_gpu = True):

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
    print(device)
    
    # Calls style transer model using the path to the VGG19 model, the style image, the shape of the content tensor (probably to match the input with the size of the content image), and the weights
    
    
    
    kwargs = {
        "epochs":epochs,
        "checkpoint_iterations":checkpoints,
        "batch_size":batch_size,
        "save_path":save_path,
        "learning_rate":learning_rate,
        "content_path":content_path,
        "device": device,
        "partly_trained":partly_trained
    }

    args = [
        content_training_images,
        style_image,
        content_weight,
        style_weight,
        tv_weight
        ]
    model = reoptimize_style_transfer(*args, **kwargs)
    model.compile(optimizer='adam')
    return model