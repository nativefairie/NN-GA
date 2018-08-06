# NN-GA
Neural Networks and Genetic Algorithms
_Using Python 2.7, Conda and Jupyter Notebook_




# FLOWERS

**custom datas;** Contains practices from https://www.youtube.com/watch?v=ZzWaow1Rvho&list=PLxt59R_fWVzT9bDxA76AHm3ig0Gg9S3So



Flowers.ipynb
--------------------------------
We use only numpy and matplotlib. 
Our weights(w1, w2, b) and bias are randomly chosen.
Using the cost functions we find out how bad our computer is doing.
Deriving the cost (finding it's slope), we are able to adjust w1, w2 and b.


Flowers_CLF.ipynb
--------------------------------
The example above, this time only using SciKit.
Testing with various classifiers. Highest accuracy was obtained with NN CLF.


Flowers_custom_CLF.ipynb
--------------------------------
Combines the first two examples. Writing a custom classifer similar to NN, 
training the network using the slope of the cost function.




# IRIS
**dataset from https://archive.ics.uci.edu/ml/datasets.html**, since included with SciKit, importing it likewise.



Iris.ipynb
--------------------------------
Solved using Tree CLF, KNN CLF, MLP CLF and NN CLF.


Iris_Custom_CLF.ipynb
--------------------------------
Writing a scrappy version of KNearestNeighbor, using Euclidean distance.

