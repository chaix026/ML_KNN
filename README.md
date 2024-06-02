### What is K-Nearest Neighbors (KNN)?

K-Nearest Neighbors (KNN) is a simple, non-parametric, and lazy learning algorithm used for both classification and regression tasks. The basic idea behind KNN is to predict the value of a new data point based on the values of its k nearest neighbors in the feature space. For classification, the predicted class is typically the majority class among the k nearest neighbors. For regression, the predicted value is often the average of the values of the k nearest neighbors.

### Key Concepts of KNN:

1. **Non-Parametric**: KNN does not assume any underlying distribution for the data.
2. **Lazy Learning**: KNN does not learn a model from the training data; instead, it memorizes the training dataset and performs computations only when a query is made.
3. **Distance Metric**: The most common distance metric used in KNN is the Euclidean distance, though others like Manhattan or Minkowski can also be used.

### Steps for Implementing KNN:

1. **Choose the number of neighbors (k)**.
2. **Calculate the distance between the query point and all the training samples**.
3. **Sort the distances and determine the k nearest neighbors**.
4. **For classification, use majority voting among the k neighbors. For regression, calculate the average value of the k neighbors**.
