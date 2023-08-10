import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from KNN import KNN

df = pd.read_csv("BankNote_Authentication.csv")

x = df[["variance", "skewness", "curtosis", "entropy"]]
y = df['class']

x_training_data, x_testing_data, y_training_data, y_testing_data = train_test_split(x,y, test_size=0.3, random_state=1234)
scaled_training = x_training_data.copy()
scaled_testing = x_testing_data.copy()
for col in ("variance", "skewness", "curtosis", "entropy"):
    scaled_testing[col] = (scaled_testing[col] - scaled_training[col].mean()) / scaled_testing[col].std()
# print("testing",scaled_testing)
for col in ("variance", "skewness", "curtosis", "entropy"):
    scaled_training[col] = (scaled_training[col] - scaled_training[col].mean()) / scaled_training[col].std()
# print("training",scaled_training)
X_training_data = np.array(x_training_data)
X_testing_data = np.array(x_testing_data)
Y_training_data = np.array(y_training_data)
Y_testing_data = np.array(y_testing_data)


k = 5
correctResult = 0
clf = KNN(k)
clf.fit(X_training_data, Y_training_data)
predictions = clf.predict(X_testing_data)

print("K value:", k)
for i in range(len(y_testing_data)):
    if predictions[i] == Y_testing_data[i]:
        correctResult = correctResult+1

print("Number of correctly classified instances:",correctResult)
print('Total number of instance:', len(Y_testing_data))
acc = correctResult / len(Y_testing_data) *100
print("Accuracy: ", acc)

