import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# iris example

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# print(df.head())


train, test = df[df['is_train']==True], df[df['is_train']==False]

features = df.columns[:4]

rf = RandomForestClassifier(n_jobs=2, random_state=0)
y = pd.factorize(train['species'])[0]
rf.fit(train[features], y)

preds = iris.target_names[rf.predict(test[features])]

# confusion matrix
print(pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds']))

# feature importance
print(list(zip(train[features], rf.feature_importances_)))