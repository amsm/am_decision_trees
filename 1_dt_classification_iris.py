import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Load dataset
iris = load_iris()
X = iris.data # samples
y = iris.target # labels

# Train Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier = dt_classifier.fit(
    X, # samples
    y # labels
)

# Plot the decision tree
plt.figure(
    figsize=(20,10)
)

tree.plot_tree(
    dt_classifier,
    feature_names=iris.feature_names,
    class_names=iris.target_names.tolist(),  # Convert to list
    filled=True
)
plt.show()
