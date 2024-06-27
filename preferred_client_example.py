import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS_COL = 'Preferred'

# Example data
dict_dataset = {
    'Account age > 1': ['Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No'],
    'Violations > 0': ['No', 'Yes', 'No', 'No', 'No', 'No', 'No', 'Yes'],
    'Balance > $50.000': ['Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes'],
    LABELS_COL: ['No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No'] # labels
}

df4ds = pd.DataFrame(dict_dataset)

# Separate features and target
X = df4ds.drop(LABELS_COL, axis=1) # the samples
y = df4ds[LABELS_COL] # labels

# Convert categorical features to numerical values
X = X.apply(LabelEncoder().fit_transform)

# Convert target to numerical values
le = LabelEncoder() # designed for 1D arrays
y = le.fit_transform(y)

# Initialize the model
model = DecisionTreeClassifier()

# Fit the model
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Print predictions alongside actual labels and indices
results_df = pd.DataFrame({
    'Index': X.index,
    'Actual': le.inverse_transform(y),
    'Predicted': le.inverse_transform(predictions)
})
print(results_df)

# Calculate accuracy
accuracy = accuracy_score(y, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(
    model,
    feature_names=X.columns.tolist(),
    class_names=le.classes_.tolist(),
    filled=True
)
plt.show()
