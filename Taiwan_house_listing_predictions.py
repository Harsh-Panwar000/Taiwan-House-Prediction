import sklearn
from sklearn import linear_model
import pandas as pd
import numpy as np

#Read dataset and drop unnecessary columns

df = pd.read_csv("Real estate valuation data set.csv")
df = df[["X2 house age", "X3 distance to the nearest MRT station", "X4 number of convenience stores", "Y house price of unit area"]]

#Set what we are predicting
predict = "Y house price of unit area"

#Create the x and y arrays
x = np.array(df.drop([predict], 1))
y = np.array(df[predict])

#train the data while setting 5% aside for testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.05)

#apply the linear regression model
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

#print the accuracy of the predictions compared to actual
accuracy = linear.score(x_test, y_test)
print(accuracy)

#compare all the prediction values to the real listings
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
