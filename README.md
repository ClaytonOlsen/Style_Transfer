# Style_Transfer


In this repository are scripts to build and run two types of neural style transfer models. 

The first uses an optimization technique for two images, a content image and a style reference image, to blend them together to make a new images that make the content image look like it was painted in the style of the style reference image. This is implemented by optimizing the output image to match the content statistics if the content image and the style statistics of the style reference image which are extracted from the images using a CNN. The use of the nueral style transfer model was motivated by https://www.tensorflow.org/tutorials/generative/style_transfer


