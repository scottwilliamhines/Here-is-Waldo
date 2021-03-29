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

<mark>TODO: Insert gif of labelimg example**</mark>

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

