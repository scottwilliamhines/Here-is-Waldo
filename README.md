# Here's Waldo

***

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
A Resnet architecture helps deeper nueral networks perform at least as well or better than their shallower counterparts. I was intrigued by the possiblities of more layers and therefore more image manipulations in object detection. This model was not exceedingly fast, training at about 1.26 steps per second. It did, however, produce some encouraging results given that it was the first model that I ran. Here are some example detections from the resnet model:

**ssd_resnet50 on a small resolution image**

<img src="https://github.com/scottwilliamhines/here_is_waldo/blob/main/tensorflow/workspace/exported_images/ssd_resnet_v1/waldo1_predict.png?raw=true"
     alt = "ssd_resnet50 on a small resolution image" 
     width = "200"/>

**ssd_resnet50 on a high resolution image**

 <img src="https://github.com/scottwilliamhines/here_is_waldo/blob/main/tensorflow/workspace/exported_images/ssd_resnet_v1/waldo108_predict.png?raw=true"
     alt = "ssd_resnet50 on a high resolution image" 
     width = "800"/>
     
**ssd_resnet50 on a completely different style of Waldo image from what the model was trained on**
 
 <img src="https://github.com/scottwilliamhines/here_is_waldo/blob/main/tensorflow/workspace/exported_images/ssd_resnet_v1/waldo_predict.png?raw=true"
     alt = "ssd_resnet50 on a completely different style of Waldo image from what the model was trained on" 
     width = "500"/>


This being the first model I tried, I wanted to see where I could get with some other CNNs. 

* * *

**faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8:**

The second model that I trained was an older model and more computationally expensive. The faster_rcnn architechture is all about regionalizing the image and then making classifications on each of those regions. For detection problems like mine it also includes a bounding box around the regions that it classifies. This model utilizes inception_resnet_V2 as well which, as I understand it, runs the image through multiple sizes of convolutions at each layer to help determine where in the image the usable features are and help in the regionalization of the model. I really felt that this model would do well, but it ultimately under performed. It was by far the slowest model that I trained running at about .001 steps per second and had a tough time with the larger image and the different style of Waldo.

**Faster RCNN on a small resolution image**

<img src="https://github.com/scottwilliamhines/here_is_waldo/blob/main/tensorflow/workspace/exported_images/rcnn_inception_resnet_v2/waldo1_rcnn.png?raw=true"
     alt = "Faster RCNN on a small resolution image" 
     width = "200"/>

**Faster RCNN on a high resolution image**

 <img src="https://github.com/scottwilliamhines/here_is_waldo/blob/main/tensorflow/workspace/exported_images/rcnn_inception_resnet_v2/waldo108_rcnn.png?raw=true"
     alt = "Faster RCNN on a high resolution image" 
     width = "800"/>
     
**Faster RCNN on a completely different style of Waldo image from what the model was trained on**
 
 <img src="https://github.com/scottwilliamhines/here_is_waldo/blob/main/tensorflow/workspace/exported_images/rcnn_inception_resnet_v2/waldo_rcnn.png?raw=true"
     alt = "Faster RCNN on a completely different style of Waldo image from what the model was trained on" 
     width = "500"/>

* * *

**efficientdet_d0_coco17_tpu:**

EfficientDet is a model developed by the Google Brain team and is their flavor of a Convolutional Neural Network. It has been pre-trained on the ImageNet data set and is fairly usable right out of the box. The main benefit of using EfficientDet is that it is significantly cheaper on computational resources than many of it's counterparts without sacrificing the accuracy of the model. You can see in the image below a comparison of the resource usage in Billions of Flops against the accuracy of each model training on the COCO17 dataset. 

<img src= "https://github.com/scottwilliamhines/here_is_waldo/blob/main/visuals/efficientdet_speed.png?raw=true"
     width = "500"
     />
 
 ###### **Image credit: [learnopencv.com](https://learnopencv.com/efficientnet-theory-code/)**

This model did produce really good results right out of the gate with the smaller Waldo crops, but had a lot of trouble with the larger Waldo images. It performed best of all the models that I worked on with an out-of-context Waldo image. It was also by far the fastest model to train. It trained at roughly 5.75 steps per minute. 

This was the total loss over the course of training:

<p><img src= 'https://github.com/scottwilliamhines/here_is_waldo/blob/main/visuals/efficientdet_d0_v1_tensorboard_visuals/Loss_total_loss.svg'
        width = "500"
        ></p>

Here are some of the detections that it made:

**EfficientDet on a small resolution image**

<img src="https://github.com/scottwilliamhines/here_is_waldo/blob/main/tensorflow/workspace/exported_images/efficientdet_d0_v1/waldo1_efficientdet.png?raw=true"
     alt = "EfficientDet on a small resolution image" 
     width = "200"/>

**EfficientDet on a high resolution image**

 <img src="https://github.com/scottwilliamhines/here_is_waldo/blob/main/tensorflow/workspace/exported_images/efficientdet_d0_v1/waldo108_efficientdet.png?raw=true"
     alt = "EfficientDet on a high resolution image" 
     width = "800"/>
     
**EfficientDet on a completely different style of Waldo image from what the model was trained on**
 
 <img src="https://github.com/scottwilliamhines/here_is_waldo/blob/main/tensorflow/workspace/exported_images/efficientdet_d0_v1/waldo_efficientdet.png?raw=true"
     alt = "EfficientDet on a completely different style of Waldo image from what the model was trained on" 
     width = "500"/>

* * * 

### Conclusion 

The best performing model was the Efficientdet model. It was not by any means perfect, but it did really well on the smaller resolution images and it also was able to better predict a different style of Waldo image than any of the other models. My ultimate goal is to get this model working with live video capture and so I will need the model to work quickly on the images that are fed into it. As a goal for the future I would like to work on some of the more complex architextures within the Efficientdet family. I would also like to add in a bit more image processing and expand my dataset a bit. This was an amazingly fun project to work on and I fully intend to continue to hone this model. 

Waldo will be found!

<img src="https://github.com/scottwilliamhines/here_is_waldo/blob/main/tensorflow/workspace/exported_images/efficientdet_d0_v1/training_gif.gif?raw=true"
     alt = "Finding Waldo Gif" 
     width = "200"/>





