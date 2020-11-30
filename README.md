# Deep-sea classifier
This project represents several attempts to build an accurate classifier of deep-sea video footage as "interesting" or not.
Because a big part of the deep-sea video footage is plain dark image without any interest, this classifier allows to mark the frames on which something interesting actually happens, for example a fish swims by, a plant is seen, or a litter is detected.
This can save a lot of human labor for those who have to analyze tens of hours of deep-sea expedition footage to find anything of interest.

# Data set
A data set used for training and testing the classifier consisted of 40000 frames extracted from the video footage of Ocean Exploration
Trust’s Nautilus Exploration Program Cruise NA095 and NA101 which took place in Cascadia Margin and Papah¯anaumoku¯akea Marine National Monument respectively.

**The data set along with everything in this repository can be downloaded from [here](https://1drv.ms/u/s!AsQgIUPpz0Thl0ByAukM0EukzqsD?e=iyDrr4)**

# Approach

## Simple Convolutional Neural Network
First, a simple Convolutional Neural Network (CNN) was constructed as CNNs are the most common approach used for image classification.
A CNN has an input layer, then 3 convolutional layers, 2 dense layers and finally the output.
Unfortunatelly, it showed quite bad results most likely due to lack of training data bacause the network contained over 67 million trainable parameters thus it needed millions of training images to reach an acceptable performance while in reality the data set consisted of only 40000 images.
To be more precise, after an extensive training on the data set, the CNN showed an accuracy of only around 50% which as good as random guessing.

## Transfer learning
In order to combat the lack of training data, a transer learining technique was applied. In simple terms, transfer learning means that a pre-trained model is taken and only the last layer is retrained with regard to the current dataset.
Several most popular pre-trained models were tested in order to find which one would generalize the best for classifying the deep-sea footage with such small training data set, namely MobileNetV2, DenseNet, InceptionV3, Xception, InceptionResNetV2, and VGG19.

# Results
Listed below are the models that were tested and their classification accuracy
- Simple Convolutional Neural Network: 50% accuracy
- MobileNetV2: 45% accuracy
- DenseNet: 92% accuracy
- InceptionV3: 63% accuracy
- Xception: 93% accuracy
- InceptionResNetV2: 88% accuracy
- VGG19: 93% accuracy

As can be seen Xception and VGG19 performed the best. However, VGG19 took less trainig time to reach this accuracy, which allows to conclude that VGG19 is the best model out of the tested ones for the deep-sea footage classification as "interesting" or not.
