# Style_Transfer


In this repository are scripts to build and run two types of neural style transfer models. 

The first uses an optimization technique for two images, a content image and a style reference image, to blend them together to make a new images that make the content image look like it was painted in the style of the style reference image. This is implemented by optimizing an output image to match the content statistics if the content image and the style statistics of the style reference image which are extracted from the images using the intermediatary layers of Convolution Neural Network. The vgg19 model, used in this example, is trained to understand an input image and generalize its invariances and defining features within classes, ignoring potential background noise. Between feeding the input image and receiving the classification label, the model serves as a complex feature extractor that we can strip apart to understand the content and style features of an image. So, with a model that can extract the contetn and feature of an image, we extract the content features from the content image, the style features from the style reference image, and adjust an output image with gradent descent that minimizes a loss function that tries to minimize the loss for the content features and style features. The use of the nueral style transfer model was motivated by [this article](https://www.tensorflow.org/tutorials/generative/style_transfer).


Content Image             | Style Image             |  Output Image
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://storage.googleapis.com/khanhlvg-public.appspot.com/arbitrary-style-transfer/belfry-2611573_1280.jpg" width="325" />   | <img src="Style_Transfer_Individual/style_images/zelda.jpg" width="325" />  |  <img src="Style_Transfer_Individual/finished_transfers/castle+zelda.png" width="325" />
<img src="https://storage.googleapis.com/khanhlvg-public.appspot.com/arbitrary-style-transfer/belfry-2611573_1280.jpg" width="325" />   | <img src="Style_Transfer_Individual/style_images/8bit.jpg" width="325" />  |  <img src="Style_Transfer_Individual/finished_transfers/castle_8bit.png" width="325" />


The second is a Fast Style Transfer algorithm we train a model using the style_content loss extracted using the vgg19 model to train a feed-forward network for image style transfer using a single style reference image. The trained model can quickly style transfer images or video rather than run the entire optimization for each use case. Implemented with tensorflow 2.0 and trained on the Microsoft Coco 2017 dataset which is a large-scale object detection, segmentation, and captioning dataset. The results are not ideal and  would benefit from longer training periods and a better gpu that would allow for larger batches and faster learning.

Model trained on Starry Night -> Changing the 8bit photo -> Style transfered photo
<img src="Fast_Style_Transfer//fast_style_transer.png" width="800" /> 
