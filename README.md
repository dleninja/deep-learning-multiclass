# Deep Learning for Multiclass Classification

There are many different tasks that can be performed using deep learning models, i.e., neural networks. One such task is classification or the categorization of data, e.g., images. In the case when there is only two categorization, we would refer to that as *Binary* classification. However, if there are more than one category, it is referred to as *Multi-class* classification. There is another term which is *Mutli-label* classification. This is when a data can be categorized as more than one category.

An example could be a dataset of the outdoors. In each image, there could be a person, a cat, car, etc. The images in that dataset can have multiple labels. Whereas a dataset with images with cat only, people only, and car only images, would be considered a multiclass dataset.

![There are three types of classification problems, binary, multiclass and multilabel classification.](/misc/types_of_classification_examples.png)

<sub> Example of three types of classification problems, binary, multiclass and multilabel classification. [Image source](https://www.mathworks.com/help/deeplearning/ug/multilabel-image-classification-using-deep-learning.html)</sub>

---

In traditional machine learning, the user would have to develop features as input into the machine learning model, e.g., support vector machine, neural network, etc. For example, the Iris Flower Dataset, the features of that dataset include the sepal length, sepal width, petal length and petal width. This process can be time consuming to both develop and extract from the image. Furthermore, it may exclude certain information that is intrinsic to the image. Deep learning, on the other hand, uses convolutional neural networks, that can automatically learn features from the image and use these features for classification. This concept of automatic feature learning helped contribute to the potential of deep learning in a wide range of applications.

To develop and customize your CNN for your specific task, one of the important parameters to change is the number of neurons in the final layer. The neurons in the final layer (the fully connected layer) is the output of your model. For example in the case of a multiclass classifier, that is predicting 100 different categories, the final layer should contain 100 neurons. Whereas if is binary classification, then there should only be 2 neurons.

![A convolutional neural network has two parts, the feature learning stage and the decision making stage, i.e., the classifier.](/misc/neural_network_diagram.png)

<sub> Example of a convolutional neural network. [Image source](https://www.run.ai/guides/deep-learning-for-computer-vision/deep-convolutional-neural-networks)</sub>

To train a classifier into one of the three types, it is important to specify the type of ground truth data.

For instance in the binary classification, the desired output could be a vector with two elements, where each element is equal to 1 or 0. If one element is equal to 1, the other is equal to 0.
> Example:
> 
> For x<sub>1</sub>: y<sub>1</sub> = [1 0]
> 
> For x<sub>2</sub>: y<sub>2</sub> = [0 1]

In the multiclass classification, the desired output could be a vector, where each element is equal to 1 or 0, and if one element is equal to 1, the remaining elements are equal to 0.
> Example:
> 
> For x<sub>1</sub>: y<sub>1</sub> = [1 0 0]
> 
> For x<sub>2</sub>: y<sub>2</sub> = [0 1 0]
> 
> For x<sub>3</sub>: y<sub>3</sub> = [0 0 1]

In the multilabel classification, the desired output could be a vector, where each element is equal to 1 or 0, and the remaining elements could also be 1 or 0 as well.
> Example:
> 
> For x<sub>1</sub>: y<sub>1</sub> = [1 0 0]
> 
> For x<sub>2</sub>: y<sub>2</sub> = [0 1 1]
> 
> For x<sub>3</sub>: y<sub>3</sub> = [1 0 1]

---
# Endnotes:

There are actually two ways to design a binary classifier using a CNN. One way, which is mentioned above with the number of output neurons being equal to 2. In that case the ground truth will be a vector of length 2, e.g., [1 0]. Typically the activation function would be a softmax function. However, it the number of neurons could be equal to 1, in this case the ground truth will be a single value, that can be equal to 1 or 0. The activation function would be a sigmoid function. 

---

In this repository, I will be showing an example of multiclass classification using a publicly available dataset. For example online datasets, they are available with this [link](https://imerit.net/blog/22-free-image-datasets-for-computer-vision-all-pbm/).
