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

### Intermediate Example: KNN for Classification

We'll use the Iris dataset, which is a classic dataset in machine learning. It contains 150 samples of iris flowers, each with four features (sepal length, sepal width, petal length, petal width) and a target variable representing the flower species.

#### Step-by-Step Implementation:

1. **Install necessary libraries**:

   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

2. **Import Libraries**:

   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
   ```

3. **Load and Prepare the Data**:

   ```python
   # Load the Iris dataset
   iris = load_iris()
   X = iris.data
   y = iris.target

   # Split the data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

   # Standardize the features
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)
   ```

4. **Train the KNN Model**:

   ```python
   # Create a KNN classifier with k=5
   k = 5
   knn = KNeighborsClassifier(n_neighbors=k)

   # Train the model
   knn.fit(X_train, y_train)
   ```

5. **Make Predictions and Evaluate the Model**:

   ```python
   # Make predictions on the test set
   y_pred = knn.predict(X_test)

   # Evaluate the model
   accuracy = accuracy_score(y_test, y_pred)
   print(f'Accuracy: {accuracy:.2f}')

   # Print the confusion matrix
   cm = confusion_matrix(y_test, y_pred)
   print('Confusion Matrix:\n', cm)

   # Print the classification report
   cr = classification_report(y_test, y_pred, target_names=iris.target_names)
   print('Classification Report:\n', cr)
   ```

6. **Visualize the Decision Boundary** (for 2D):

   ```python
   # Reduce the dataset to 2D for visualization (using only the first two features)
   X_train_2d = X_train[:, :2]
   X_test_2d = X_test[:, :2]

   knn_2d = KNeighborsClassifier(n_neighbors=k)
   knn_2d.fit(X_train_2d, y_train)

   # Create a mesh to plot the decision boundary
   x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
   y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
   xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))

   # Predict the label for each point in the mesh
   Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
   Z = Z.reshape(xx.shape)

   # Plot the decision boundary
   plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

   # Plot the training points
   plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.coolwarm)
   plt.xlabel('Sepal length')
   plt.ylabel('Sepal width')
   plt.title('KNN Decision Boundary (k=5)')
   plt.show()
   ```

### Complete Code:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a KNN classifier with k=5
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)

# Print the classification report
cr = classification_report(y_test, y_pred, target_names=iris.target_names)
print('Classification Report:\n', cr)

# Reduce the dataset to 2D for visualization (using only the first two features)
X_train_2d = X_train[:, :2]
X_test_2d = X_test[:, :2]

knn_2d = KNeighborsClassifier(n_neighbors=k)
knn_2d.fit(X_train_2d, y_train)

# Create a mesh to plot the decision boundary
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict the label for each point in the mesh
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

# Plot the training points
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('KNN Decision Boundary (k=5)')
plt.show()
```

This code demonstrates how to implement and evaluate a KNN classifier in Python using the `scikit-learn` library. It also shows how to visualize the decision boundary for a 2D subset of the Iris dataset. Adjust the parameters and features as needed for your specific use case.
