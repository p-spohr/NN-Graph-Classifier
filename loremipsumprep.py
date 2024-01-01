

x3 = 3

# %%

import re



with open('lorem-ipsum-raw.txt') as f:
    rawlines = f.readlines()
    
type(rawlines)

lines = re.split(r'[\W]', str(rawlines).lower())
setlines = set(lines)
plines = list(setlines)
plines.pop(0)
print(plines)
print(len(plines))

lorumipsum = plines
print(lorumipsum)

# %%

import numpy as np

lorum = np.random.choice(plines, size = 3, replace=False)

print(' '.join(lorum))

# %%

print(lorumipsum)




# %%
