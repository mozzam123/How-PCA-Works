# Principal Component Analysis (PCA) in Machine Learning

## Overview

This repository demonstrates how **Principal Component Analysis (PCA)** works in the context of machine learning. PCA is a dimensionality reduction technique used to simplify large datasets by reducing the number of variables, retaining the most important information while discarding less significant features. It is commonly used to speed up machine learning algorithms, reduce overfitting, and visualize high-dimensional data.

This implementation uses the **Digits dataset** from `sklearn`, which consists of 8x8 pixel images of handwritten digits (0-9). The goal is to reduce the dataset's dimensionality while maintaining the variability of the data and then build a machine learning model using PCA components.

## Steps Explained

### 1. Load Dataset

We start by loading the Digits dataset, which is a set of 8x8 pixel images of digits and their corresponding labels.

```python
from sklearn.datasets import load_digits
dataset = load_digits()
dataset.keys()
```
### 2. Data Exploration

We explore the dataset to understand its structure, including the number of features and the shape of the data.

```python
dataset.data.shape
dataset.data[0]
dataset.data[0].reshape(8,8)
```
### 3. Visualize the Data

Using `matplotlib`, we visualize one of the images from the dataset to get a better understanding of what the data looks like.

```python
from matplotlib import pyplot as plt
plt.gray()
plt.matshow(dataset.data[4].reshape(8,8))
```
### 4. Prepare the Data

The dataset is converted into a Pandas DataFrame for easier manipulation and understanding. We also show the first few rows of the data.

```python
import pandas as pd
df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
df.head()
```
### 5. Standardize the Data

PCA is affected by the scale of the data, so we standardize the data to have a mean of 0 and a standard deviation of 1. This ensures that each feature contributes equally to the analysis.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset.data)
```

### 6. Apply PCA

Next, we apply PCA to the scaled data. We reduce the dimensionality of the dataset to two principal components for easier visualization.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
```

### 7. Visualize the PCA Result

Once we apply PCA, we can visualize the transformed data in a 2D space. This helps us understand how the data has been reduced while retaining important information.

```python
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=dataset.target, cmap='plasma')
plt.colorbar()
plt.title("PCA of Digits Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

### 8. Explained Variance

The explained variance tells us how much information (variance) is retained by each principal component. We can print the explained variance ratio to see how much of the dataset's variability is captured by the first two components.

```python
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance by the 2 components: {explained_variance}")
```
### 9. Train a Classifier with PCA Components

Now that we have reduced the dimensionality, we can train a machine learning model, such as a **Support Vector Machine (SVM)**, using the PCA components instead of the original dataset.

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(pca_result, dataset.target, test_size=0.3, random_state=42)

# Train the classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 10. Compare with Original Dataset

For comparison, we can train the same classifier with the original dataset (before applying PCA) to see how much the performance improves or worsens with dimensionality reduction.

```python
# Split the original data
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(scaled_data, dataset.target, test_size=0.3, random_state=42)

# Train the classifier on original data
clf_original = SVC(kernel='linear')
clf_original.fit(X_train_original, y_train_original)

# Evaluate the classifier
y_pred_original = clf_original.predict(X_test_original)
print("Classification Report (Original Data):")
print(classification_report(y_test_original, y_pred_original))
```
### 11. Conclusion

- **PCA** effectively reduces the dimensionality of the dataset, making it easier to visualize and process.
- Training a classifier with the PCA components results in faster computations and reduced complexity.
- A trade-off exists between data compression (dimensionality reduction) and performance, where the loss of some data might affect accuracy.
- **PCA** is particularly useful for high-dimensional datasets and can speed up the training of machine learning models when used appropriately.

### References

- [Principal Component Analysis (PCA) Documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
