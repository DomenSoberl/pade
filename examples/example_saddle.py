import numpy as np
import pade
from sklearn import tree
import matplotlib.pyplot as plt

# Generate random samples of the saddle function.
samples = 100
data = np.random.uniform(-1, 1, (samples, 2))
target = [x**2 - y**2 for [x, y] in data]

# Compute Pade values.
q_table = pade.pade(data, target)

# Translate Pade values to the readable Q-labels.
q_labels = pade.create_q_labels(q_table, ['x', 'y'])

# Print Pade results.
for (point, q_label) in zip(data, q_labels):
    print(point, q_label)

# Enumerate Q-labels so they can be used with scikit-learn.
classes, class_names = pade.enumerate_q_labels(q_labels)

# Build a qualitative model in the form of a decision tree.
classifier = tree.DecisionTreeClassifier(min_impurity_decrease=0.1)
model = classifier.fit(data, classes)

# Visualize the learned model.
tree.plot_tree(model, feature_names=['x', 'y'], class_names=class_names, filled=True)
plt.show()