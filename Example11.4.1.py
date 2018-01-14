import numpy as  np
import math

uniform_sample = np.random.uniform(0,1,10000)

Y = -0.5*np.log(1-uniform_sample)

new_uniform_sample = np.random.uniform(0,1,10000)

XgivenY = -np.log(1-new_uniform_sample) + Y

x_bar = np.mean(XgivenY)
s2 = np.var(XgivenY,ddof=1)
s = s2**(0.5)

print(x_bar)
print(s)	
