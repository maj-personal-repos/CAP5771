# TODO Comment

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

mu, sigma = 10, 2
s = np.random.normal(mu, sigma, 1000)
df = pd.DataFrame(s)

mean = df.mean()
variance = df.var()
bar_labels = ['Normal Distribution']
x_pos = list(range(len(bar_labels)))

f, pltarr = plt.subplots(2)

count, bins, ignored = pltarr[0].hist(s, 30, normed=True)
pltarr[0].plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2)), linewidth=2, color='r')

pltarr[1].bar([1], mean, yerr=variance, align='center', color='#FFC222', alpha=0.5)

# add a grid
pltarr[0].grid()
pltarr[1].grid()

# set height of the y-axis
max_y = max(zip(mean, variance)) # returns a tuple
pltarr[1].set_ylim([0, (max_y[0] + max_y[1]) * 1.1])

# set axes labels and title
pltarr[0].set_ylabel('Probability')
pltarr[0].set_title('Probability Mass Function: normal (gaussian) distribution, ')

pltarr[1].set_ylabel('Value')
pltarr[1].set_xticks([0, 1, 2])
pltarr[1].set_xticklabels(['', 'Normal Distribution', ''])
pltarr[1].set_title('Mean Scores For Each Distribution')

plt.show()