# Data-Science
All about data science

We discuss topics related to data science, neural networks, data mining, advanced analytics, deep learning, reinforcement learning etc

# Neural Networks:

# Perceptron:
The Perceptron Algorithm:

1. Start with the all-zeroes weight vector w1 = 0, and initialize t to 1. Also let’s automatically scale all examples x to have (Euclidean) length 1, since this doesn’t affect which side of the plane they are on.
2. Given example x, predict positive iff wt·x > 0.
3. On a mistake, update as follows:

    • Mistake on positive: wt+1 ← wt + x.
    
    • Mistake on negative: wt+1 ← wt − x.
    
    t ← t + 1.
    
So, this seems reasonable. If we make a mistake on a positive x we get wt+1 ·x = (wt+x)·x = wt·x+1, and similarly if we make a mistake on a negative x we have wt+1 ·x = (wt−x)·x =wt·x − 1. So, in both cases we move closer (by 1) to the value we wanted.

# Dimensionality Reduction:

# Linear Discriminant Analysis and Quadratic Discriminant Analysis

Linear discriminant analysis (LDA), normal discriminant analysis (NDA), or discriminant function analysis is a generalization of Fisher's linear discriminant, a method used in statistics, pattern recognition and machine learning to find a linear combination of features that characterizes or separates two or more classes of objects or events. The resulting combination may be used as a linear classifier, or, more commonly, for dimensionality reduction before later classification.
https://en.wikipedia.org/wiki/Linear_discriminant_analysis

Quadratic discriminant analysis (QDA) is closely related to linear discriminant analysis (LDA), where it is assumed that the measurements from each class are normally distributed. Unlike LDA however, in QDA there is no assumption that the covariance of each of the classes is identical. When the normality assumption is true, the best possible test for the hypothesis that a given measurement is from a given class is the likelihood ratio test. 
https://en.wikipedia.org/wiki/Quadratic_classifier

# K Nearest Neighbors Classification Model:
the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. In k-NN classification, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.

I used IRIS dataset to classify flower into it's class, this algorithm uploaded above finds the best 'k' based on the accuracy scores by plotting accuracy vs k values for 112 classes. Reporting the accuracy scores by picking up the best 'k' obtained from the graph. By using 'timeit' function you can check the performance of the algorithm which just takes about 1.5 milliseconds as '%timeit a.predict(X_test,25)' in the IPython Console.


