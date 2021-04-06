import pandas as pd

df = pd.read_csv("diamonds.csv", index_col=0)
# print(df["cut"].unique())
# print(df["color"].unique())
# print(df["clarity"].unique())

#print(df.isnull().any())

cut_to_num = {"Fair":1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
color_to_num = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
clarity_to_num = {"I3": 1, "I2": 2, "I1": 3, "SI2": 4, "SI1": 5, "VS2": 6, "VS1": 7, "VVS2": 8}

df["cut"] = df["cut"].map(cut_to_num)
df["color"] = df["color"].map(color_to_num)
df["clarity"] = df["clarity"].map(clarity_to_num)

import sklearn
import numpy as np
from sklearn import svm, preprocessing

df = sklearn.utils.shuffle(df)

X = df.drop("price", axis=1).values
X = preprocessing.scale(X)
y = df["price"].values

test_size = 200

X_train = X[:-test_size]
y_train = y[:-test_size]

X_test = X[-test_size:]
y_test = y[-test_size:]

# checking for any NaN
# print(np.isnan(X_train).any())
# print(np.isnan(y_train).any())
# print(np.isnan(X_test).any())
# print(np.isnan(y_test).any())

X_train[np.isnan(X_train)] = np.median(X_train[~np.isnan(X_train)])
X_test[np.isnan(X_test)] = np.median(X_test[~np.isnan(X_test)])

clf = svm.SVR(kernel="linear") #rbf
clf.fit(X_train, y_train)
print(f"{clf.score(X_test, y_test)*100}%")

for X,y in zip(X_test, y_test):
    print(f"Prediction: {clf.predict([X])[0]}, Actual: {y}")
