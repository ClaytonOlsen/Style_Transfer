import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense
from tensorflow.keras import Model
from keras.utils.data_utils import get_file
from utility_functions.preprocessing_utility import *
from keras.applications import imagenet_utils
from tensorflow.keras import backend
import IPython.display as display


# Image net vgg19 model
def VGG19_style_bottleneck(input_shape=None, input_tensor=None, include_top = True, classifier_activation='softmax', weights='imagenet'):
    input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=224,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    img_input = Input(shape = input_shape)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Fully connected layers  
    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = Dense(1000, activation=classifier_activation, name='predictions')(x)
   
    
    model = Model(inputs= img_input, outputs =x, name='vgg19_copy')
    if weights == 'imagenet':
        if include_top:
            WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                       'keras-applications/vgg19/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
          file_hash='253f8cb515780f3b799900260a226db6')
    model.load_weights(weights_path)
    return model

#The Style/Content layer extractor
def model_layer_extraction(layer_names):
  vgg = VGG19_style_bottleneck(include_top=False, weights='imagenet')
  vgg.trainable = False
  outputs = [vgg.get_layer(name).output for name in layer_names]
  model = tf.keras.Model([vgg.input], outputs)
  return model


#GRAM MATRIX
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

#Uses layer extractor to find style and content layers to make a new model
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = model_layer_extraction(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        #tf.keras.applications.vgg19.preprocess_input/imagenet_utls.preprocess_input returns the images converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling.
        preprocessed_input = imagenet_utils.preprocess_input(inputs, data_format=None, mode='caffe')
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                        outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                        for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}



#Loss function calculation
def style_content_loss(outputs, style_targets, style_weight, num_style_layers, content_targets, content_weight, num_content_layers):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

#Variation denoising
def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    return x_var, y_var

#Variational loss L1 loss
def total_variation_loss(outputs, total_variation_weight):
    x_deltas, y_deltas = high_pass_x_y(outputs)
    return total_variation_weight*(tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas)))

def total_loss_function(image, outputs, constants):
    style_targets, style_weight, num_style_layers, content_targets, content_weight, num_content_layers, total_variation_weight = constants
    sty_con_loss = style_content_loss(outputs, style_targets, style_weight, num_style_layers, content_targets, content_weight, num_content_layers)
    total_var_loss = total_variation_weight*tf.image.total_variation(image)
    return sty_con_loss + total_var_loss

#Train function
@tf.function()
def train_step(image, opt, extractor, constants):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = total_loss_function(image, outputs, constants)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    return loss



#Full function
def image_style_transfer(content_image_path, style_image_path, epochs = 500):
    content_image = load_img(content_image_path)
    style_image = load_img(style_image_path)
    style_image = tf.nn.avg_pool(style_image, ksize = [3,3], strides = [1,1], padding = 'SAME')
    content_layers = ['block5_conv2'] 

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    extractor = StyleContentModel(style_layers, content_layers)
    #results = extractor(tf.constant(content_image))
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    opt = tf.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-8)
    style_weight=1e-2
    content_weight=1e4
    total_variation_weight = 10
    constants = [style_targets, style_weight, num_style_layers, content_targets, content_weight, num_content_layers, total_variation_weight]
    image = tf.Variable(content_image)
    

    total_loss = []
    for n in range(epochs):
        total_loss.append(train_step(image, opt, extractor, constants))
        if n%100 == 0:
            display.clear_output(wait=True)
            print(f'Current Loss: {total_loss[n]}')
            print(f'Step: {n}')
            display.display(tensor_to_image(image))


    return image, total_loss
