import pandas as pd
from sklearn import datasets
import numpy as np


# frequencies

titanic_df = pd.read_csv('../data/titanic3.csv', na_values=['NA'])

# print(titanic_df.head())

char_cabin = titanic_df["cabin"].astype(str)    # Convert cabin to str

new_Cabin = np.array([cabin[0] for cabin in char_cabin]) # Take first letter

titanic_df["cabin"] = pd.Categorical(new_Cabin)  # Save the new cabin var

print(titanic_df.head())

# one way tables

print(pd.crosstab(index=titanic_df['survived'], columns='count'))


print(pd.crosstab(index=titanic_df["sex"], columns="count"))

cabin_tab = pd.crosstab(index=titanic_df["cabin"], columns="count").iloc[0:8]

print(cabin_tab)

print(cabin_tab/cabin_tab.sum())

# two way tables

survived_sex = pd.crosstab(index=titanic_df["survived"],
                           columns=titanic_df["sex"])

survived_sex.index= ["died", "survived"]

print(survived_sex)

survived_class = pd.crosstab(index=titanic_df["survived"], columns=titanic_df["pclass"])

survived_class.columns = ["class1", "class2", "class3"]
survived_class.index= ["died", "survived"]

print(survived_class)

survived_class = pd.crosstab(index=titanic_df["survived"], columns=titanic_df["pclass"], margins=True)

survived_class.columns = ["class1", "class2", "class3", "row total"]
survived_class.index = ["died", "survived", "col total"]

print(survived_class)

print(survived_class/survived_class.ix["col total", "row total"])

print(survived_class/survived_class.ix["col total"])

print(survived_class.div(survived_class["row total"], axis=0))

iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

print(df.head())

print(df.describe(percentiles=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9]))

print(df.median())

print(df.cov())

