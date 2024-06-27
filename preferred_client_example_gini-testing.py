import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree

# Example data
dict_dataset = {
    'Account age > 1': ['Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No'],
    'Violations > 0': ['No', 'Yes', 'No', 'No', 'No', 'No', 'No', 'Yes'],
    'Balance > $50.000': ['Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes'],
    'Labels': ['No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No']  # labels
}

# Convert the dataset to a DataFrame
df = pd.DataFrame(dict_dataset)
X = pd.get_dummies(df.drop(columns=['Labels']))
y = df['Labels'].apply(lambda x: 1 if x == 'Yes' else 0)

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Function to calculate Gini index
def gini(y):
    m = len(y)
    return 1 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

# Function to get the Gini index of the feature chosen at each split
def get_feature_ginis(tree, X):
    feature_ginis = []
    for i in range(tree.tree_.node_count):
        if tree.tree_.children_left[i] != tree.tree_.children_right[i]:  # If not a leaf
            feature_index = tree.tree_.feature[i]
            left_indices = X[:, feature_index] <= tree.tree_.threshold[i]
            right_indices = X[:, feature_index] > tree.tree_.threshold[i]
            left_gini = gini(y[left_indices])
            right_gini = gini(y[right_indices])
            feature_gini = (sum(left_indices) * left_gini + sum(right_indices) * right_gini) / len(y)
            feature_ginis.append((i, feature_gini))
    return feature_ginis

# Get Gini indices for the features chosen at each split
feature_ginis = get_feature_ginis(clf, X.values)

# Modify the tree visualization to include these Gini indices
tree_rules = export_text(clf, feature_names=list(X.columns))

print("Decision Tree Rules with Gini Indices:")
for i, gini_index in feature_ginis:
    print(f"Node {i}: Gini Index of Feature Chosen for Split = {gini_index:.4f}")

print("\nTree Structure:\n")
print(tree_rules)
