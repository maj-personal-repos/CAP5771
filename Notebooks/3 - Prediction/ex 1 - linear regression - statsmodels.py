from sklearn import datasets
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

data = datasets.load_boston()

print(data.DESCR)

print(data.feature_names)

print(data.target)

df = pd.DataFrame(data.data, columns=data.feature_names)

target = pd.DataFrame(data.target, columns=["MEDV"])

print(df.head())

print(target.head())

# without a constant, simple

X = df["RM"]
y = target["MEDV"]

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

print(model.summary())

plt.scatter(X, y)
plt.plot(X, predictions, color='g')
plt.xlabel('RM: Average number of rooms per dwelling')
plt.ylabel('MEDV: Median value of owner-occupied homes ($1000s)')
plt.title('Simple Linear Regression - Without Constant')
plt.show()
plt.close()

# with a constant

X_c = sm.add_constant(X)

model = sm.OLS(y, X_c).fit()
predictions = model.predict(X_c)

print(model.summary())

plt.scatter(X, y)
plt.plot(X, predictions, color='g')
plt.xlabel('RM: Average number of rooms per dwelling')
plt.ylabel('MEDV: Median value of owner-occupied homes ($1000s)')
plt.title('Simple Linear Regression - With constant')
plt.show()
plt.close()

# multivariate


X = df[["RM", "LSTAT"]]
y = target["MEDV"]

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

print(model.summary())

fig, ax = plt.subplots()
scat = ax.scatter(X["RM"], X["LSTAT"], c=y, s=200, marker='o')
fig.colorbar(scat, label='MEDV: Median value of owner-occupied homes ($1000)')
plt.xlabel("RM: Average number of rooms per dwelling")
plt.ylabel("LSTAT: % lower status of the population")
plt.title("Multivariate Linear Regression")
plt.show()
plt.close()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X["RM"], X["LSTAT"], y, c='g', marker='o')
ax.scatter(X["RM"], X["LSTAT"], predictions, c='r', marker='^')
ax.set_xlabel("RM: Average number of rooms per dwelling")
ax.set_ylabel("LSTAT: % lower status of the population")
ax.set_zlabel("MEDV: Median value of owner-occupied homes ($1000)")
plt.title("Multivariate Linear Regression")
plt.show()

