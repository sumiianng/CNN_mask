# Convolutional Neural Network
In the project, I construct a Convolutional Neural Network (CNN) for image recognition by using Masks Dataset. The dataset comes from Eden Social Welfare Foundation. The dataset contains images with more than one person, and some of people in the image wear masks. Then, the goal of the projuct is to classify the person's situation of wearing the masks.  
  
The file contains two parts. The first part is the images, and the second part is train.csv and test.csv which contain the person's information in the image including the location in the image and the the situaion of wearing mask (targets).  

[Here](https://drive.google.com/drive/folders/1TvGSz-YL2IXkPCMRY_MzlHAtjQzSmH_k?usp=sharing) is the dataset.

## Preprocessing
The raw data contains a images folder and train/test.csv. In the images folder, there are 684 images. Each image can be cut into many people's face images through the train/test.csv recording the location of people's face in the image.These people's face images are our training and testing data.
 
However, the people's face images size are not the same, so we have to do some preprocesses. We consider 3 ways to make our image size the same.

1. Resize：It's the most convenient way to solve the problem, but the objects in the image would be distorted.
2. Pad and Resize：Make the image to be square by padding 0 to the images, and then resize.
3. Crop and Resize：Make the image to be square by crop the images, and then resize.

Set the modify parameter in class Preprocess could select the way to make our image size the same. We try the 3 ways and choose the best.

## Training
Construct our network by using pytorch.Our CNN contains two parts. The first part is consist of convolution layers and maxpool layers. The second part is full connected part, and it contains linear layers and activation layers. The weights and biases in the layers are parameters which are auto setted by the network. However, the hyper-parameters including number of neurons in each layers, stride and padding in convolution layers should be set by ourselves.Change the stride and pad setting of convolution and maxpool layers in the network could make the different results. 

Then, set the hyper-parameters including batch size, learning rate.... Then, we use adam optimizer to train our model.In the training stage, we use the run_manager constructed by ourselves to record the data in each epoch and show the dataframe to know the loss and accuracy. It may heip us to analyze the results later.

## Learning curve
Use the data recorded in run manager, show the line plot to understand the change of loss and accuracy as the epoch increases including training and testing dataset.  

![image](https://github.com/sumiianng/Deep_learning/blob/main/CNN/results/Accuracy.png)
![image](https://github.com/sumiianng/Deep_learning/blob/main/CNN/results/Cross%20entropy.png)

Compare the learning curve with different hyper parameters.  

![image](https://github.com/sumiianng/Deep_learning/blob/main/CNN/results/Accuracy_train.png)
![image](https://github.com/sumiianng/Deep_learning/blob/main/CNN/results/Cross_entropy_train.png)

## Confusion matrix
Show the confusion matrix of the training and testing prediction results, and plot it. It makes us to know the model's prediction situations between the three type of classes.  

![image](https://github.com/sumiianng/Deep_learning/blob/main/CNN/results/Train%20confusion%20matrix.png)
![image](https://github.com/sumiianng/Deep_learning/blob/main/CNN/results/Test%20confusion%20matrix.png)


Furthermore, we show the accuracies in each ture classes. Example: In the 100 true 'good' data, the model predict 97 'good' and 3 not 'good'. Knowing which classes predict the worst, we can try to adjust the dataset or model to make the results better.

## Results
Randomly sample some images contain many people's face , and show the predict lable of them in the imges.  

![image](https://github.com/sumiianng/Deep_learning/blob/main/CNN/results/sample_img.png)
