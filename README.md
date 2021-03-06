Neural Networks and Genetic Algorithms

_Using Python 2.7, Conda and Jupyter Notebook_

Trained myself using: 
1. https://www.youtube.com/watch?v=ZzWaow1Rvho&list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So [Intro to NN]
2. https://www.youtube.com/watch?v=cKxRvEZd3Mw [Intro to SciKit and TensorFlow]
3. https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1 [BP deep dive]
4. http://scikit-learn.org/stable/documentation.html [...]
5. http://neuralnetworksanddeeplearning.com [Book by Michael Nielsen]
6. https://www.youtube.com/watch?v=0e0z28wAWfg&t=287s [Guy explaining BP numerically]
**CLASSIFICATION**

# FLOWER

**custom data;** Contains practices from https://www.youtube.com/watch?v=ZzWaow1Rvho&list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So
![Flower_dataset](https://github.com/nativefairie/NN-GA/blob/master/Flower/Flowers.png)


Flowers.ipynb
--------------------------------
Using numpy and matplotlib, we will train the linearly separable network using raw **perceptron training rule**, having our weights(w1, w2, b) and bias randomly chosen. We apply sigmoid as our activation function, squeezing everything between 0 and 1.
Using the cost function for each attribute of our input we find how bad our computer is doing.
Deriving the cost (measure of the rate of change of cost), we are able to find the gradient descent and using generalized delta rule we adjustr w1, w2 and b. Our accuracy being higher now we have the ability to classify our input vector, if a flower is blue then it's input is smaller than .5, else it is red. This is called hard limit transfer function (threshold function)

Flowers_CLF.ipynb
--------------------------------
The example above, this time using SciKit.


# IRIS
**dataset from https://archive.ics.uci.edu/ml/datasets.html**, since included with SciKit, importing it likewise.

![Iris_dataset_plot](https://github.com/nativefairie/NN-GA/blob/master/Iris/Iris.png)

Iris.ipynb
--------------------------------
One class is linearly separable from the other two; the latter are not linearly separable from each other. So, we have to use a multilayer architecture, the perceptrons in the first layer are
used as preprocessors producing linearly separable vectors for the next layer. Solved using Tree CLF, KNN CLF, MLP CLF.
Noted accuracies for Tree 0,94; KNN 0,973; MLP 0,96

Iris_Custom_CLF.ipynb
--------------------------------
Writing a scrappy version of KNearestNeighbor, using Euclidean distance.


# MUSHROOM

Mushroom.ipynb
--------------------------------

**dataset from https://archive.ics.uci.edu/ml/datasets.html**, reading it with pandas as matrix.
With SciKit features have to be numerical variables. Mushroom dataset being composed of categorical variables requires encoding. Pandas contains get_dummies() which does everything in one go and provides appropriate column labels. Solved with MLP with 10 hidden layers, score obtained: 1.0



**REGRESSION**


# WINE QUALITY

Wine_Quality.ipynb
--------------------------------

**dataset from https://archive.ics.uci.edu/ml/datasets.html**.
MLPRegressor implements MLP that trains using BackPropagation with no activation function(step fn, 0 or 1). It uses the cost function as the loss function, and the output is a set of continuous values.
PCA is an unsupervised method, rather than attempting to predict the y values from the x values, the problem attempts to learn about the relationship between the x and y values.
This relationship is quantified by finding a list of the principal axes in the data, and using those axes to describe the dataset: explained variance and components. Visualizing them as vectors means seeing the principal axes of the data. So we are reducing the dimension of the data but keepinf the relationship between the data points. Uses: speeding compilation and data visualization.
https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html is awesome
for understanding Principal Component Analysis!

Used Pipeline which helped to apply PCA together with our our model, the MPLRegressor. Scaled the components, as asked by PCA.



**IMAGE RECOGNITION**


# HANDWRITTEN DIGITS

Optical_digits_recognition.ipynb
---------------------------------

**dataset from: https://archive.ics.uci.edu/ml/datasets.html**
![Handwritten_digits_dataset](https://github.com/nativefairie/NN-GA/blob/master/Handwritten_Digits_Classification/Handwritten.png)

Dataset is composed of normalized bitmaps of handwritten digits from a preprinted form, that consisted of 8x8 pixel images. So 64 columns, each row being a number.

Importing the two datasets from UCI with pandas and splitting data into test data and train data was the first step. Used a couple of models to test the accuracy but all turned out to have 0 errors. Played around a bit and made a pipeline with PCA and MPLClassifier.
What this means is that we used backpropagation algorithm to recurrsively change the activatation value of the previous layers in proportion to weights, increase the weights in proportion to activation value, adjust the bias..With PCA principal components were already chosen. Used 2 hidden layers and Stochastic Gradient Descent as solver, which means we found approximations of the cost function at each step for each minibatch.
64 comlumns of input datas were reduced with PCA to 2columns for data visualization. We can now think of the data as the layout of the digits on X,y axis.

Optical_digits_recognition_BP.ipynb
------------------------------------

**dataset from: http://yann.lecun.com/exdb/mnist/**

Solved using raw BP algorithm.



**ALGORITHMS**

# BACKPROPAGATION

BP.ipynb
---------

Guess what the hidden units should look like based on how the inputs look like and what the output should look like, using:

![BackPropagation](https://github.com/nativefairie/NN-GA/blob/master/BackPropagation/BP.png)

I just like the drawing:


<img src="https://github.com/nativefairie/NN-GA/blob/master/BackPropagation/BP2.png" width="350">


The Backpropagation algorithm is a supervised learning method for multilayer feed-forward networks.


# HANDWRITTEN DIGITS

MNIST_Digit_Dataset.ipynb
------------------------------------

MNIST Digit Dataset with raw BP algorthm as classifier.

Long story short:
https://www.youtube.com/watch?v=0e0z28wAWfg&t=287s - this guy here helped by highlighting the obvious!

Some jibberish notes of how i kept track of what I was doing can be found in the folder.

The whole point of view was though sustained by the formulas.

Refactoring with Stochastic Gradient Descent should improve performance.


