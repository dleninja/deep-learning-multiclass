# Deep Learning for Multiclass Classification

There are many different tasks that can be performed using deep learning models, i.e., neural networks. One such task is classification or the categorization of data, e.g., images. In the case when there is only two categorization, we would refer to that as *Binary* classification. However, if there are more than one category, it is referred to as *Multi-class* classification. There is another term which is *Mutli-label* classification. This is when a data can be categorized as more than one category.

An example could be a dataset of the outdoors. In each image, there could be a person, a cat, car, etc. The images in that dataset can have multiple labels. Whereas a dataset with images with cat only, people only, and car only images, would be considered a multiclass dataset.

![There are three types of classification problems, binary, multiclass and multilabel classification.](/misc/types_of_classification_examples.png)

<sub> Example of three types of classification problems, binary, multiclass and multilabel classification. [Image source](https://www.mathworks.com/help/deeplearning/ug/multilabel-image-classification-using-deep-learning.html)</sub>

---

Deep learning uses convolutional neural networks, that can automatically learn features from the image and use these features for classification. To train a classifier into one of the three types, it is important to specify the type of ground truth data.

For instance in the binary classification, the desired output could be a vector with two elements, where each element is equal to 1 or 0.

In the multiclass classification, the desired output could be a vector, where each element is equal to 1 or 0, and if one element is equal to 1, the remaining elements are equal to 0. Example:


---


In this repository, I will be showing an example of multiclass classification using a publicly available dataset. For example online datasets, they are available with this [link](https://imerit.net/blog/22-free-image-datasets-for-computer-vision-all-pbm/).
