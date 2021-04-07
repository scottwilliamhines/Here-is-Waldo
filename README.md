# Here's Waldo

### The Problem

Waldo needs to be found. No one can get a close up photo of him and he is always in very crowded areas. He has been spotted in all time periods and among all species, both real and fictional. How do you search an area with that broad of a scope? My answer to this problem is to use object detection. Through a transfer learning process I will use pre-trained models from Tensorflow and retrain them towards the task of findind the ever-elusive Waldo. 

### The Data

I found a data set of Where's Waldo images on Kaggle: [Where's Waldo](https://www.kaggle.com/residentmario/wheres-waldo). It is a collection of full Where's Waldo puzzles along with cropped sections of those Where's Waldo puzzles split into the categories of "Waldo" and "Not Waldo." Those cropped portions of the total image come in the sizes of 268X268 pixels, 128X128 pixels and 64X64 pixels. For image detection I am only really concerned with the images that contain Waldo in them. So I took every image that was classified as having Waldo and made that into my data set. 

### Cleaning and Labelling
Tensorflow's Object Detection API relies on the trainer to tell it where the objects in question are with in the images that are fed into it. Therefore we have to take the painstaking task of labelling every image. To do this I used a tool called labelimg found here: [labelimg](https://github.com/tzutalin/labelImg). This utility allows you to label the image with the location or locations of the objects you wish to train your model on. The output is a .xml file relating to your image that has the following information in it:

- Image Height, Width and Depth

- Classifications for the bounding boxes (in our case there is only one: waldo)

- x_min, y_min, x_max and y_max for each bounding box

![Labelimg Usage:](https://github.com/scottwilliamhines/here_is_waldo/blob/main/visuals/labelimg_example.gif)

This is important because we will feed this information along with the location information of the original images into our Tensorflow Object Detection API by converting all of that data into a TFRecord file. 

### Creating a label map and generating the TFrecord

Before we continue onto the next step we have to create a label map, which is essentially the a .txt document with a specific format that lets our API know what to label the inference boxes that it will eventually draw. For my case since I am only training on one object it looks like this:

<code>
    item {
    
    id: 1
    
    name: 'waldo'
    
    } 
</code>

The last step is feeding all of the data above into the `generate_tfrecord.py` file. There is example usage both in my [Here is Waldo Notebook](https://github.com/scottwilliamhines/here_is_waldo/blob/main/tensorflow/here_is_waldo.ipynb) as well as inside the script itself. 
(Code Source: `generate_tfrecord.py` came from [Tensorflow's Object Detection API Tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#training-the-model))

That is it for the data prep!

### The Fun Stuff

Now we get to pick our model and try to detect Waldo!

I decided to re-train multiple models to see how they would compare. Here are the models that I tried 

**ssd_resnet50_v1_fpn_640x640_coco17_tpu:**
A Resnet architecture helps deeper nueral networks perform at least as well or better than their shallower counterparts. I was intrigued by the possiblities of more layers and therefore more image manipulations in object detection. This model produced decent results and I was encouraged, but I thought that I could do better. Here are some example detections from the resnet model:

![ssd_resnet50 on a small resolution image:](https://github.com/scottwilliamhines/here_is_waldo/blob/main/tensorflow/workspace/exported_images/ssd_resnet_v1/waldo1_predict.png?raw=true =100x)

![ssd_resnet50 on a high resolution image:](https://github.com/scottwilliamhines/here_is_waldo/blob/main/tensorflow/workspace/exported_images/ssd_resnet_v1/waldo108_predict.png?raw=true =100x)

![ssd_resnet50 on a completely different style of Waldo image from what the model was trained on:](https://github.com/scottwilliamhines/here_is_waldo/blob/main/tensorflow/workspace/exported_images/ssd_resnet_v1/waldo_predict.png?raw=true =100x)


This being the first model I tried, I wanted to see where I could get with some other CNNs. 

**efficientdet_d0_coco17_tpu:**

EfficientDet is a model developed by the Google Brain team and is their flavor of a Convolutional Neural Network. It has been pre-trained on the ImageNet data set and is fairly usable right out of the box. The main benefit of using EfficientDet is that it is significantly cheaper on computational resources than many of it's counterparts without sacrificing the accuracy of the model. 

<mark>TODO: add efficientdet speed graph</mark>

This model did produce really good results right out of the gate with the smaller Waldo crops, but had a lot of trouble with the larger Waldo images.

This was the total loss over the course of training:
<p><img src= 'https://github.com/scottwilliamhines/here_is_waldo/blob/main/visuals/efficientdet_d0_v1_tensorboard_visuals/Loss_total_loss.svg'></p>

Here are some of the detections that it made:

<mark>TODO: add waldo image example from efficientdet</mark>

**faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8:**

The third and final model that I trained was an older model and more computationally expensive. The faster_rcnn architechture is all about regionalizing the image and then making classifications on each of those regions. For detection problems like mine it also includes a bounding box around the regions that it classifies. This model utilizes inception_resnet_V2 as well which, as I understand it, runs the image through multiple sizes of convolutions at each layer to help determine where in the image the usable features are and help in the regionalization of the model. 

<mark>TODO: add waldo image example from faster_rcnn</mark>

### Conclusion 









