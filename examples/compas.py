from corels import *
import numpy as np

# Train split proportion
train_proportion = 0.8

X, y, features, prediction = load_from_csv("data/compas.csv")
# Constructor parameters ensure maximum verbosity, and a maximum cardinality of 3
# makes CORELS search all rule antecedents with up to three features combined together
c = CorelsClassifier(max_card=1, n_iter=100000, verbosity=["progress"])

# Generate train and test sets
train_split = int(train_proportion * X.shape[0])

X_train = X[:train_split]
y_train = y[:train_split]

X_test = X[train_split:]
y_test = y[train_split:]


# Fit the model. Features is a list of the feature names
c.fit(X_train, y_train, features=features, prediction_name="recidivate")

# Score the model on the test set
a = c.score(X_test, y_test)

print("Test Accuracy: " + str(a))

# Print the rulelist
print(c.rl())
