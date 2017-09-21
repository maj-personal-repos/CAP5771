import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(-4.0, 4.0, 1000)

# gaussian
u = 0
sigma = 1

p_x_gaussian = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - u)**2 / (2 * sigma**2))

plt.plot(x, p_x_gaussian)
plt.show()
plt.close()

# chi-square

x = np.linspace(0.0, 5.0, 1000)

df = 1

p_x_chi_sq = stats.chi2.pdf(x, df)

plt.plot(x, p_x_chi_sq)
plt.show()
plt.close()
