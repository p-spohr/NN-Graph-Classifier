# https://matplotlib.org/stable/gallery/statistics/histogram_cumulative.html#sphx-glr-gallery-statistics-histogram-cumulative-py
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html

#%%

import numpy as np
import matplotlib.pyplot as plt
import random as random
import os

from lorem_ipsum_prep import lorumipsum as text

# %%

# mean and standard deviation
mu = 0
sigma = 1
n = 1000 

s = np.random.normal(mu, sigma, n)
print(s)

# %%

fig, ax = plt.subplots()


# %%

count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')

plt.show()

# %%

t = range(n)
s = np.random.normal(mu, sigma, n)

fig, ax = plt.subplots()

ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')

ax.grid()

plt.show()

# %%

s = np.random.normal(mu, sigma, n)

fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Normal Distribution')
plt.hist(s, 100, density=True)
plt.show()

# %%

np.random.seed(19680801)

mu = 200
sigma = 25
n_bins = 25
data = np.random.normal(mu, sigma, size=100)

fig = plt.figure(figsize=(9, 4), layout="constrained")
axs = fig.subplots(1, 2, sharex=True, sharey=True)

# Cumulative distributions.
axs[0].ecdf(data, label="CDF")
n, bins, patches = axs[0].hist(data, n_bins, density=True, histtype="step",
                               cumulative=True, label="Cumulative histogram")
x = np.linspace(data.min(), data.max())
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (x - mu))**2))
y = y.cumsum()
y /= y[-1]
axs[0].plot(x, y, "k--", linewidth=1.5, label="Theory")

# Complementary cumulative distributions.
axs[1].ecdf(data, complementary=True, label="CCDF")
axs[1].hist(data, bins=bins, density=True, histtype="step", cumulative=-1,
            label="Reversed cumulative histogram")
axs[1].plot(x, 1 - y, "k--", linewidth=1.5, label="Theory")

# Label the figure.
fig.suptitle("Cumulative distributions")
for ax in axs:
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("Annual rainfall (mm)")
    ax.set_ylabel("Probability of occurrence")
    ax.label_outer()

plt.show()
# %%


rng = np.random.default_rng(19680801)

# example data
mu = 106  # mean of distribution
sigma = 17  # standard deviation of distribution
x = rng.normal(loc=mu, scale=sigma, size=420)

num_bins = 42

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=True)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--')
ax.set_xlabel('Value')
ax.set_ylabel('Probability density')
ax.set_title('Histogram of normal distribution sample: '
             fr'$\mu={mu:.0f}$, $\sigma={sigma:.0f}$')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()
# %%

print(type(bins))
print(bins)

# %%

y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)

print(type(y))
print(y)

# %%

fig, ax = plt.subplots()
plt.plot(y)

# %%


rng = np.random.default_rng(12345)
rints = rng.integers(low=0, high=10, size=3)
print(rints)

# %%

rand_int = random.random()
print(rand_int*10)

# %%

text = np.array(text)

# %%

rng = np.random.default_rng()

random_words = rng.choice(text, size=3, replace=False)

print(random_words)

# %%
