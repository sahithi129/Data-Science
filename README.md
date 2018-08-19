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
