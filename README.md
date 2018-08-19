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
    Quadratic discriminant analysis (QDA) is closely related to linear discriminant analysis (LDA), where it is assumed that the measurements from each class are normally distributed. Unlike LDA however, in QDA there is no assumption that the covariance of each of the classes is identical. When the normality assumption is true, the best possible test for the hypothesis that a given measurement is from a given class is the likelihood ratio test. Suppose there are only two groups, (so {\displaystyle y\in \{0,1\}}  y \in \{0,1 \} ), and the means of each class are defined to be {\displaystyle \mu _{y=0},\mu _{y=1}}  \mu_{y=0},\mu_{y=1}  and the covariances are defined as {\displaystyle \Sigma _{y=0},\Sigma _{y=1}}  \Sigma_{y=0}, \Sigma_{y=1} . Then the likelihood ratio will be given by

Likelihood ratio = {\displaystyle {\frac {{\sqrt {|2\pi \Sigma _{y=1}|}}^{-1}\exp \left(-{\frac {1}{2}}(x-\mu _{y=1})^{T}\Sigma _{y=1}^{-1}(x-\mu _{y=1})\right)}{{\sqrt {|2\pi \Sigma _{y=0}|}}^{-1}\exp \left(-{\frac {1}{2}}(x-\mu _{y=0})^{T}\Sigma _{y=0}^{-1}(x-\mu _{y=0})\right)}}<t} {\displaystyle {\frac {{\sqrt {|2\pi \Sigma _{y=1}|}}^{-1}\exp \left(-{\frac {1}{2}}(x-\mu _{y=1})^{T}\Sigma _{y=1}^{-1}(x-\mu _{y=1})\right)}{{\sqrt {|2\pi \Sigma _{y=0}|}}^{-1}\exp \left(-{\frac {1}{2}}(x-\mu _{y=0})^{T}\Sigma _{y=0}^{-1}(x-\mu _{y=0})\right)}}<t}
for some threshold {\displaystyle t} t. After some rearrangement, it can be shown that the resulting separating surface between the classes is a quadratic. The sample estimates of the mean vector and variance-covariance matrices will substitute the population quantities in this formula.
