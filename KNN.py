import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.y_train = None
        self.X_train = None
        self.k = k

    def fit(self, x, y):
        self.X_train = x
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [euclideanDistance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(self.k)
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            y_list = self.y_train.tolist()
            indices = [y_list.index(most_common[i][0] for i in most_common)]
            indices.sort()
            return y_list[indices[0]]
        else:
            return most_common[0][0]


def euclideanDistance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2)**2))
    return distance
