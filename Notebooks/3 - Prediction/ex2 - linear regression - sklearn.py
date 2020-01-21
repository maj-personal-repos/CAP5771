from sklearn import datasets, linear_model
import pandas as pd

data = datasets.load_boston()

df = pd.DataFrame(data.data, columns=data.feature_names)

target = pd.DataFrame(data.target, columns=["MEDV"])

X = df

y = target["MEDV"]

print(X.head())

print(y)

lm = linear_model.LinearRegression()

model = lm.fit(X, y)

predictions = lm.predict(X)

print("Linear regression score: %f " % lm.score(X, y))

print("Linear model coefficients: " + str(lm.coef_))

print("Linear model y intercept: %f " % lm.intercept_)


