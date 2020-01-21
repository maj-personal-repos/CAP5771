from sklearn import svm
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style


def plot_hyperplane(linear_svm):
    w = linear_svm.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(4, 7)
    yy = a * xx - (linear_svm.intercept_[0]) / w[1]

    margin = 1 / np.sqrt(np.sum(linear_svm.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')


style.use('ggplot')

# create linear SVM
linear_svm = svm.SVC(kernel='linear')

# import source data
iris = load_iris()

# setup source dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df['species_codes'] = df['species'].cat.codes


# Linear SVM: Separable  - take only two classes from the iris dataset

# create training set
X_tr = df[df['is_train'] == True]
X_tr = pd.concat([X_tr[X_tr['species_codes'] == 0], X_tr[X_tr['species_codes'] == 1]])
y_tr = X_tr['species_codes']

# create testing set
X_tst = df[df['is_train'] == False]
X_tst = pd.concat([X_tst[X_tst['species_codes'] == 0], X_tst[X_tst['species_codes'] == 1]])
y_tst = X_tst['species_codes']


linear_svm.fit(X_tr[['sepal length (cm)', 'sepal width (cm)']].values, y_tr.values)

print(linear_svm.coef_)

print(linear_svm.intercept_)

plt.scatter(X_tst[X_tst['species_codes'] == 0]['sepal length (cm)'], X_tst[X_tst['species_codes'] == 0]['sepal width (cm)'],
            marker='o', c='green', s=64)
plt.scatter(X_tst[X_tst['species_codes'] == 1]['sepal length (cm)'], X_tst[X_tst['species_codes'] == 1]['sepal width (cm)'],
            marker='s', c='green', s=64)

plt.scatter(X_tr[X_tr['species_codes'] == 0]['sepal length (cm)'], X_tr[X_tr['species_codes'] == 0]['sepal width (cm)'],
            marker='o', c='blue', s=64)
plt.scatter(X_tr[X_tr['species_codes'] == 1]['sepal length (cm)'], X_tr[X_tr['species_codes'] == 1]['sepal width (cm)'],
            marker='s', c='blue', s=64)

plot_hyperplane(linear_svm)

plt.show()

print(linear_svm.score(X_tst[['sepal length (cm)', 'sepal width (cm)']].values, y_tst))

# Linear SVM: Nonseparable Case

# change the data a bit to create a non-separable case

y_tr.iloc[0] = 1
y_tr.iloc[len(y_tr)-1] = 0

# create linear SVM with larger slack boundaries
linear_svm_slack = svm.SVC(kernel='linear', C=0.05)

linear_svm_slack.fit(X_tr[['sepal length (cm)', 'sepal width (cm)']].values, y_tr.values)

plt.scatter(X_tr[X_tr['species_codes'] == 0]['sepal length (cm)'], X_tr[X_tr['species_codes'] == 0]['sepal width (cm)'],
            marker='o', c='blue', s=64)
plt.scatter(X_tr[X_tr['species_codes'] == 1]['sepal length (cm)'], X_tr[X_tr['species_codes'] == 1]['sepal width (cm)'],
            marker='s', c='green', s=64)

plot_hyperplane(linear_svm_slack)

plt.show()

print(linear_svm_slack.score(X_tr[['sepal length (cm)', 'sepal width (cm)']].values, y_tr))

# Nonlinear SVM

# first some helper functions

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, **params)

# change the data back
y_tr.iloc[0] = 0
y_tr.iloc[len(y_tr)-1] = 1

# create nonlinear svm
nonlinear_svm = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)

# fit the data
nonlinear_svm.fit(X_tr[['sepal length (cm)', 'sepal width (cm)']].values, y_tr)

# create meshgrid for plotting decision boundaries
X0, X1 = X_tr['sepal length (cm)'], X_tr['sepal width (cm)']
xx, yy = make_meshgrid(X0, X1)

plot_contours(nonlinear_svm, xx, yy, cmap=plt.cm.coolwarm, alpha=0.2)

plt.scatter(X_tr[X_tr['species_codes'] == 0]['sepal length (cm)'], X_tr[X_tr['species_codes'] == 0]['sepal width (cm)'],
            marker='o', c='blue', s=64)
plt.scatter(X_tr[X_tr['species_codes'] == 1]['sepal length (cm)'], X_tr[X_tr['species_codes'] == 1]['sepal width (cm)'],
            marker='s', c='green', s=64)

plt.show()

