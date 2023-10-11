#task 1

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# reading csv file

df = pd.read_csv("C:\\Users\\Arbiya\\Downloads\\Iris.csv")     
df=pd.DataFrame(df) 
print(df)
print(df.head())     #returns first 5 entries

print(df.tail())

print(df.info())

print(df.nunique())

print(df.shape)

print(df.values)

print(df.isna().sum())

# import matplotlib.pyplot as plt

for i in df.columns:
    plt.scatter(df[str(i)], df['SepalLengthCm'], label=str(i))
    plt.xlabel(i)
    plt.ylabel("Sepal Length cm")
    plt.show()

# # Analysing data by pairplot


# # Create a pairwise scatter plot
sns.pairplot(df, hue='Species')

# # Display the plot
print(plt.show())

print(df['Species'].unique())

print(df['Species'].replace({'Iris-setosa':'1', 'Iris-versicolor':'2', 'Iris-virginica':'3'},inplace=True))
# converting categorical value into numerical value

from sklearn.model_selection import train_test_split
# splitting data into train data and Test data

X_train,X_test,y_train,y_test=train_test_split(df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']],df['Species'])

print(len(X_test))

print(len(X_train))

from sklearn.linear_model import LogisticRegression
# importing model for classification and prediction

lr=LogisticRegression()

print(lr.fit(X_train,y_train))

print(lr.predict(X_test))
# predicts the output
print(lr.score(X_test,y_test))
