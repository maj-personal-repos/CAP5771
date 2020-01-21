import numpy as np
import pandas as pd

u, sigma = 1, 0.5
s = np.random.lognormal(u, sigma, 1000)
df = pd.DataFrame(s)

print(df.describe())

