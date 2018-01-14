
import numpy as  np
import math

uniform_sample = np.random.uniform(0,1,10000)

#u_bar = np.mean(uniform_sample)

#u_s2 = np.var(uniform_sample,ddof=1)

#u_s = u_s2**(0.5)

#print (u_bar)

#print(u_s)

Y = -0.5*np.log(1-uniform_sample)

print (Y)	

new_uniform_sample = np.random.uniform(0,1,10000)

XgivenY = -np.log(1-new_uniform_sample) + Y

x_bar = np.mean(XgivenY)
s2 = np.var(XgivenY,ddof=1)

s = s2**(0.5)

print(x_bar)
print(s)

#for xgy in np.nditer(XgivenY , op_flags = ['readwrite']):


#for u in np.nditer(uniform_sample):
#	print u

#for y in np.nditer(Y):
#	print y			





