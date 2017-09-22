from sklearn import datasets, linear_model
import pandas as pd

data = datasets.load_boston()

df = pd.DataFrame(data.data, columns=data.feature_names)

target = pd.DataFrame(data.target, columns=["MEDV"])

X = df

y = target["MEDV"]

lm = linear_model.LinearRegression()
model = lm.fit(X, y)

predictions = lm.predict(X)

print(lm.score(X, y))

print(lm.coef_)

print(lm.intercept_)


