import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

# Load dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Train Decision Tree Regressor
dt_regressor_california =\
    DecisionTreeRegressor(
        max_depth=3 # will take a lot of time if no max_depth is set, because this is a large dataset to plot!
    )
dt_regressor_california = dt_regressor_california.fit(X, y)

plt.figure(
    figsize=(20,10)
)

tree.plot_tree(
    dt_regressor_california,
    feature_names=housing.feature_names,
    filled=True,
    fontsize=13,
    proportion=True,
    # rounded=True
)
plt.show()
