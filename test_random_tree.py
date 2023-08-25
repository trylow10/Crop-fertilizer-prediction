import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left subtree (child)
        self.right = right  # Right subtree (child)
        self.value = value  # Class label (for leaf nodes)

class RandomForest:
    def __init__(self, n_trees=100, max_depth=5, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        if self.n_features is None:
            self.n_features = int(np.sqrt(n_features))

        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            subset_X = X[indices]
            subset_y = y[indices]
            tree = self._build_tree(subset_X, subset_y, depth=0)
            self.trees.append(tree)

    def get_classes(self):
        return self.classes

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionNode(value=leaf_value)

        feature_indices = np.random.choice(n_features, self.n_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feature_indices)
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return DecisionNode(feature_index=best_feature, threshold=best_threshold, left=left_tree, right=right_tree)

    def _best_split(self, X, y, feature_indices):
        best_gini = float('inf')
        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices
                if np.sum(left_indices) > 0 and np.sum(right_indices) > 0:
                    gini = self._gini(y[left_indices], y[right_indices])
                    if gini < best_gini:
                        best_gini = gini
                        best_feature = feature
                        best_threshold = threshold
        return best_feature, best_threshold

    def _gini(self, *groups):
        total_samples = sum(len(group) for group in groups)
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            class_counts = Counter(group)
            class_probabilities = [class_counts[c] / size for c in class_counts]
            gini -= sum(p ** 2 for p in class_probabilities)
        return gini

    def predict(self, X):
        predictions = [self._predict_sample(x) for x in X]
        return np.array(predictions)

    def _predict_sample(self, x):
        tree_predictions = [tree_predict(x, tree) for tree in self.trees]
        return Counter(tree_predictions).most_common(1)[0][0]

def tree_predict(x, tree):
    while tree.value is None:
        if x[tree.feature_index] < tree.threshold:
            tree = tree.left
        else:
            tree = tree.right
    return tree.value

