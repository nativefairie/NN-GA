Neural Networks and Genetic Algorithms

_Using Python 2.7, Conda and Jupyter Notebook_

Trained myself using: 
1. https://www.youtube.com/watch?v=ZzWaow1Rvho&list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So
2. https://www.youtube.com/watch?v=cKxRvEZd3Mw
3. https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1
4. http://scikit-learn.org/stable/documentation.html
5. http://neuralnetworksanddeeplearning.com [_Book by Michael Nielsen_]

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
**dataset from https://archive.ics.uci.edu/ml/datasets.html**, reading it with pandas as matrix.
With SciKit features have to be numerical variables. Mushroom dataset being composed of categorical variables requires encoding. Pandas contains get_dummies() which does everything in one go and provides appropriate column labels. Solved with MLP with 10 hidden layers, score obtained: 1.0

**REGRESSION**

# WINE QUALITY
**dataset from https://archive.ics.uci.edu/ml/datasets.html**.
MLPRegressor implements MLP that trains using BackPropagation with no activation function(step fn, 0 or 1). It uses the cost function as the loss function, and the output is a set of continuous values.
PCA is an unsupervised method, rather than attempting to predict the y values from the x values, the problem attempts to learn about the relationship between the x and y values.
This relationship is quantified by finding a list of the principal axes in the data, and using those axes to describe the dataset: explained variance and components. Visualizing them as vectors means seeing the principal axes of the data. So we are reducing the dimension of the data but keepinf the relationship between the data points. Uses: speeding compilation and data visualization.
https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html is awesome
for understanding Principal Component Analysis!

Used two new concepts, make_pipeline and PCA. Pipeline helped to apply PCA together with our classifier, the MPLRegressor. Should have scaled the components on the X axis but the accuracy whitout it was still 1.0

**IMAGE RECOGNITION**

# HANDWRITTEN DIGITS
**dataset from: 
1. https://archive.ics.uci.edu/ml/datasets.html
2. http://yann.lecun.com/exdb/mnist/**

**ALGORITHMS**

# BACKPROPAGATION

