import pandas as pd
from scipy import stats

data = pd.read_csv('../../data/brain_size.csv', sep=';', na_values='.')

print(data.head())

print(data.describe())

print(stats.ttest_1samp(data['VIQ'], 0))

print(stats.ttest_1samp(data['VIQ'], 112))

group_by_gender = data.groupby('Gender')

for gender, value in group_by_gender['VIQ']:
    print((gender, value.mean()))

print(group_by_gender.mean())

female_weight = data[data['Gender'] == 'Female']['Weight']

male_weight = data[data['Gender'] == 'Male']['Weight']

print(male_weight)

print(stats.ttest_ind(female_weight, male_weight, nan_policy='omit'))

