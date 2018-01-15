import numpy as  np
import math
import scipy.stats as st
from scipy.stats import norm

uniform_sample = np.random.uniform(0,1,10000)

Y = -0.5*np.log(1-uniform_sample)

#print (Y)	

new_uniform_sample = np.random.uniform(0,1,10000)

XgivenY = -np.log(1-new_uniform_sample) + Y

x_bar = np.mean(XgivenY)
s2 = np.var(XgivenY,ddof=1)

s = s2**(0.5)

print "Sample Mean is",x_bar
print "Sample standard deviation is",s

#standard error
stan_error = st.mstats.sem(XgivenY)

#95% confidence interval for X
z = norm.ppf(0.95)
low = x_bar - stan_error*z
high = x_bar + stan_error*z
 
print "The confidence interval for X is (",low,",",high,")"
#print(low)
#print(high)	
