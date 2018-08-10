Neural Networks and Genetic Algorithms

_Using Python 2.7, Conda and Jupyter Notebook_

**CLASSIFICATION**

# FLOWER

**custom data;** Contains practices from https://www.youtube.com/watch?v=ZzWaow1Rvho&list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So
![Flower_dataset](https://github.com/nativefairie/NN-GA/blob/master/Flower/Flowers.png)


Flowers.ipynb
--------------------------------
Using numpy and matplotlib, we will train the perceptron network, **perceptron training rule**, so that they can learn to solve this classification problem.
Our weights(w1, w2, b) and bias are randomly chosen.
Using the cost functions we find out how bad our computer is doing.
Deriving the cost (finding it's slope), we are able to adjust w1, w2 and b.

Flowers_CLF.ipynb
--------------------------------
The example above, this time using SciKit.


# IRIS
**dataset from https://archive.ics.uci.edu/ml/datasets.html**, since included with SciKit, importing it likewise.

![Iris_dataset_plot](https://github.com/nativefairie/NN-GA/blob/master/Iris/Iris.png)

Iris.ipynb
--------------------------------
Solved using Tree CLF, KNN CLF, MLP CLF.
Noted accuracies for Tree 0,94; KNN 0,973; MLP 0,96

Iris_Custom_CLF.ipynb
--------------------------------
Writing a scrappy version of KNearestNeighbor, using Euclidean distance.



# MUSHROOM
**dataset from https://archive.ics.uci.edu/ml/datasets.html**, reading it with pandas as matrix.
With SciKit features have to be numerical variables. Mushroom dataset being composed of categorical variables requires encoding. Pandas contains get_dummies() which does everything in one go and provides appropriate column labels. Solved with MLP, score obtained: 1.0

**REGRESSION**

#WINE QUALITY
MLPRegressor implements MLP that trains using BackPropagation with no activation function(step fn, 0 or 1). It uses the cost function as the loss function, and the output is a set of continuous values.

**IMAGE RECOGNITION**

#HANDWRITTEN DIGITS


**ALGORITHMS**

#BACKPROPAGATION

