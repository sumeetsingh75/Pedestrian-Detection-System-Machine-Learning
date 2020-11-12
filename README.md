# Pedestrian-Detection-System-Machine-Learning
# Introduction

The project aims at proving a software based solution for pedestrian Detection on roads using Machine Learning Techniques and Computer Vision. Advancement in Autonomous driving also demands more accurate and precise safety features to be in place, for the safety of both drivers and pedestrians. The objective of this project is to enhance the pedestrian safety at crosswalks as crosswalks are main hot spots where most of the accidents occur involving pedestrians.

The system is expected to take the video feed from a camera at the crosswalk as input. Computer vision and Machine Learning approaches will process the video feed and detect the presence of pedestrians on the crosswalk. In addition, the system will allow the user to defining the region of interest in scenes to speed up the detection process and count number of pedestrians in the scene.

# MobileNet Single Shot Detector
Certain Object Detection models like R-CNN family works great for Object Detection problems, but cannot be used in real-time scenarios. While, these models are known for accuracy, but lag in terms of speed. R-CNNs use selective search algorithms to get bounding box proposals and then compute CNN features in proposed regions. This two-step procedure makes R-CNN family unsuitable for real-time applications.

Single Shot MultiBox detector is originally designed over Feed Forward Convolution Network and generates fixed size anchor boxes and uses Non-Maximum suppression to produce final detections. SSD is a backbone of various object detection models and originally designed for real-time object detection tasks. SSDs aim to improve the speed to object detection by eliminating the need for region proposal as in R-CNN family and detect and classify multiple objects in single-shot or one pass within the image.

In SSDs various feature map extractors can be used. Advantage of using MobileNet feature extractor with SSD framework is that, it is specifically created for mobile devices. The basic idea behind MobileNet is that the convolution layers that are important for computer vision tasks can be replaced by depth wise separable convolutions that reduces the number of operations drastically. MobileNets uses 3x3 depthwise separable convolutions that reduces computation time by 8 to 9 times than standard convolutions without much reduction in accuracy. MobileNet SSD is initially trained on COCO dataset and fine-tuned on PASCAL VOC.


# Transfer Learning
Transfer Learning is a Machine Learning approach in which knowledge gathered from one task is used in another task. This technique is widely used in the Computer Vision domain where it is quite difficult to create and train models from scratch considering huge the size of dataset and computing resources required to train the model. MobileNet SSD provided by TensorFlow API is used as base model and trained over new dataset by employing Transfer Learning techniques to customize the model as per
the requirement.The expectation is to train the MobileNet SSD model on an extensive dataset specifically designed for applications involving pedestrian.

# Machine Learning Training Pipeline
## Dataset Preprocessing
[Joint Attention in Autonomous Driving (JAAD)](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) dataset consists of 346 HD mp4 video clips 5-10 seconds long, recorded from an onboard camera at 30 fps. The dataset focuses on pedestrian behaviors at a crosswalk and provides spatial annotation information of pedestrians in XML format.
1. **Data Filtering:** Dataset is recorded from over 240 hours of driving. A subset of videos from the entire dataset is manually selected for desired scenarios. 

2. **Frames Generation and resizing:** Individual video is converted into respective frames using OpenCV. Frame size and corresponding Bounding Box are scaled down from 1920 X 1080 to 960 X 540 by a factor of 0.5 to speed up the training process.

3. **Removed Non-Meaningful Data:** Frames with no pedestrians are discarded during frame generation using a python script to ensure NO TRUE NEGATIVES in the dataset. Since the video is recorded at 30fps, so selected every 8th frame of video to avoid the redundant data. Also in annotation files, a specific annotation referred to as ‘people’ is ignored. In the original dataset, a group of persons is annotated as ‘people’ when an individual person cannot be annotated in the group.

4. **XML to CSV conversion for annotations:** TensorFlow training pipeline requires annotation to be present in CSV format and that too a common CSV file for the entire dataset. But originally dataset contains annotations in JAAD specific XML format for an individual video. Hence XML file is parsed to extract required annotations and generate a common CSV file.

5. **Train/Test Split:** Train and Test data is split in 80:20 proportion. Train dataset contains 1136 images with 5644 bounding boxes and Test dataset contains 238 images with 1213 bounding boxes

6. **TRF Records Generation:** Finally, Train and Test Data is converted into respective *.records files using python script.

## Loss Function

Object Detection Algorithm outputs two components: a bounding box depicting the position of an object in an image and a confidence score associated with the class probabilities of the identified object. Hence, the Loss function of trainer consists of two components:
- Confidence Loss (ζcls) that depicts the goodness of the classification
- Localization loss (ζloc) for bounding box offset.
```bash
Total Loss ζ = (ζ𝑐𝑙𝑠+ αζ𝑙𝑜𝑐)/𝑁
Where 𝑁 : number of matched bounding boxes.
α: balances the weights between two losses.
```

If 𝑁=0, loss is taken as 0. The localisation loss is a smooth L1 loss between ground truth and predicted bounding box and confidence loss is the SoftMax loss over multiple classes confidences.

## Training Environment
Model training is done over Google Colab that offers free GPU services to train machine learning models. Training data is uploaded in the google drive and google drive is mounted with Google Colab. During training, the checkpoints of the trained model are stored in google drive. Even if the session gets expire, the latest checkpoint is reloaded in the training pipeline and training is resumed. Model Training took ~9 hours over the cloud with average compute time/step of 0.5 seconds.

## Model Training
The model is trained for 80k steps. Training graphs are plotted in Tensor Board and carefully monitored during the training process. Total loss is the combination of both localization and classification loss. Due to Transfer Learning, the initial loss dropped significantly from 15 to 3 in just a few steps. Training is stopped when loss function stopped decreasing further.


# Output of Detection Module
Implementation of Pedestrian Detection Module, allows user to select the region of interest in the frame. The purpose of selecting a region in the frame is to give meaningful results to the clients. Admin can draw a region of interest in the frame, more likely to be area on the crosswalk only and pedestrian detection will occur only on that area. It will help ignoring the pedestrians where are in the view of camera, but not of interest of the use case. The output of pedestrian detection module will be number of pedestrians detected in the specified area and the detected pedestrians

## Note

This is just a small part of the actual project. Actual project involves an extensive research on the different Object Detection models and datasets that are suitable for the requirement. Another important aspect of the project that is covered is the inference time and model's accuracy. A Deep analysis of model's performance in different weather and lighting conditions and effect on the accuracy due to different camera angles is also done.
