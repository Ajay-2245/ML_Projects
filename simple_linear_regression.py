#Linear regression to predict GPA using SAT score.(dataset doesn't contain null values)
#objective is to verify manually created linear regression class methods to the one in sklearn library

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
class SimpleLinearRegression:

    def __init__(self, m, b):
        self.learning_rate = 0.00001
        self.m = m
        self.b = b
    def execute_batch_gradient_descent(self, x_train, y_train):
        n = self.learning_rate
        e1, e2 = 1e5, 1e5

        while abs(e1) > 0.00001 or abs(e2) > 0.00001:
            # e1 = dL/dm   e2 = dL/db
            e1, e2 = 0, 0
            for yi,xi in zip(y_train, x_train):
                ei = (yi - self.m * xi - self.b)
                e1 += -2 * (ei*xi)
                e2 += -2 * ei
            self.m-=(n*e1)
            self.b-=(n*e2)

    def execute_stochastic_gradient_descent(self, x_train, y_train):
        n = self.learning_rate
        # no.of epochs must be specified for gradient descent
        for i in range(0, 10000): #epochs = 10000
            for yi, xi in zip(y_train, x_train):
                # e1 = dL/dm   e2 = dL/db
                e1, e2 = 0, 0
                ei = (yi - self.m * xi - self.b)
                e1 += -2 * (ei * xi)
                e2 += -2 * ei
                self.m -= (n * e1)
                self.b -= (n * e2)
    def results(self):
        return {'slope':self.m, 'intercept':self.b}
    def predict(self, x_data):
        return self.m * x_data + self.b
    def r2_score(self, x_test, y_test):
        y_predictions = self.predict(x_test)
        rss, tss = 0, 0
        y_test_mean = y_test.mean()
        for yi, y_pred in zip(y_test, y_predictions):
            rss = rss + (yi-y_pred)**2
            tss = tss + (yi - y_test_mean)**2
        return 1 - (rss/tss)



# Read the csv file
df = pd.read_csv("./data.csv")

# standardize SAT column
df['standardized_SAT'] = (df['SAT'] - df['SAT'].mean()) / df['SAT'].std()

#defining training and test splits
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['standardized_SAT'], df['GPA'], test_size=0.25, random_state=42)

# Batch gradient decent results
linear_regression = SimpleLinearRegression(1, 1)
linear_regression.execute_batch_gradient_descent(X_train, y_train)
results = linear_regression.results()
print(results)
print(f' r2_score : {linear_regression.r2_score(X_test, y_test)}')

# training plot
plt.figure(figsize = (20, 8))
plt.subplot(2, 2, 1)
plt.title("Linear Regression using Batch gradient descent training phase")
plt.scatter(X_train, y_train, color = 'red')
training_pred = linear_regression.predict(X_train)
plt.plot(X_train, training_pred)

# predictions plot
plt.subplot(2, 2, 2)
plt.title("Linear Regression using Batch gradient descent predictions")
plt.scatter(X_test, y_test, color = 'red')
test_pred = linear_regression.predict(X_test)
plt.plot(X_test, test_pred)



# confirming the learnt parameters using sklearn
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(pd.DataFrame(X_train), pd.DataFrame(y_train))
print(model.coef_, model.intercept_)
print(f"R2_score_sklearn : {model.score(pd.DataFrame(X_test), pd.DataFrame(y_test))}")

#training plot
plt.subplot(2, 2, 3)
plt.title("Linear Regression using sklearn training phase")
plt.scatter(X_train, y_train, color = 'red')
training_pred2 = model.predict(pd.DataFrame(X_train))
plt.plot(X_train, training_pred2)

# predictions plot
plt.subplot(2, 2, 4)
plt.title("Linear Regression using sklearn predictions")
plt.scatter(X_test, y_test, color = 'red')
test_pred2 = model.predict(pd.DataFrame(X_test))
plt.plot(X_test, test_pred2)

plt.show()
