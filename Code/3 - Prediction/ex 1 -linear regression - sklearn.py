from sklearn import datasets, linear_model
import pandas as pd
import matplotlib.pyplot as plt


data = datasets.load_boston()

df = pd.DataFrame(data.data, columns=data.feature_names)

target = pd.DataFrame(data.target, columns=["MEDV"])

X = df.loc[:,"RM":"RM"]

y = target["MEDV"]

lm = linear_model.LinearRegression()

model = lm.fit(X, y)

predictions = lm.predict(X)

print("Linear regression score: %f " % lm.score(X, y))

print("Linear model coefficients: %f " % lm.coef_)

print("Linear model y intercept: %f " % lm.intercept_)

plt.scatter(X, y)
plt.plot(X, predictions, color='g')
plt.xlabel('RM: Average number of rooms per dwelling')
plt.ylabel('MEDV: Median value of owner-occupied homes ($1000s)')
plt.title('Simple Linear Regression - Without Constant')
plt.show()
plt.close()