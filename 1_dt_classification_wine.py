import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier = dt_classifier.fit(
    X, # samples
    y # labels
)

# Plot the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(
    dt_classifier,
    feature_names=wine.feature_names,
    class_names=wine.target_names.tolist(),  # Convert to list
    filled=True,
    fontsize=13,
    # proportion=True
)
plt.show()
