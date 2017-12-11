import pandas as pd
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori, association_rules

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]


# encode the dataset into Onehot Transactions Dataframe
oht = OnehotTransactions()
oht_array = oht.fit(dataset).transform(dataset)
df = pd.DataFrame(oht_array, columns=oht.columns_)
# print(df.head())

frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
print(frequent_itemsets)

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print(rules)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.25)

rules["antecedant_len"] = rules["antecedants"].apply(lambda x: len(x))

print(rules[(rules['antecedant_len'] >= 2) & (rules['confidence'] > 0.75) &(rules['lift'] > 1.2)])

