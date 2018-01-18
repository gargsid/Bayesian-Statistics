#Example 11.4.2
#Introduction to Mathematical Statistics
#Hogg, Mckean, Craig

#alpha = 10
# Y|X = Gamma(alpha+X,1/2)
# X|Y = Poisson(Y)

import numpy as np 
import array
import scipy.stats as st
from scipy.stats import norm

XgivenY = array.array('f',[])
YgivenX = array.array('f',[])

x=y=0

alpha = 10

for i in range(6000):
	y = np.random.gamma(alpha+x,0.5)
	x = np.random.poisson(y)
	XgivenY.append(x)
	YgivenX.append(y)

y_bar = 0
for i in range(3000,6000):
	y_bar+=YgivenX[i]
	
y_bar/=3000.0

x_bar = 0
for i in range(3000,6000):
	x_bar += XgivenY[i]

x_bar/=3000.0

x_S2=0 
y_S2=0

for i in range(3000,6000):
	x_S2 += (x_bar-XgivenY[i])**2
	y_S2 += (y_bar-YgivenX[i])**2

x_S2/=2999.0
y_S2/=2999.0

z = norm.ppf(0.95)

x_se = x_S2**(0.5)/(3000.0**(0.5))
y_se = y_S2**(0.5)/(3000.0**(0.5))

xlow = x_bar - x_se*z
xhigh = x_bar + x_se*z

ylow = y_bar - y_se*z
yhigh = y_bar + y_se*z

print "Estimate : X"
print "Sample Estimate : ",x_bar
print "Sample Variance : ",x_S2
print "Confidence Interval for X : (",xlow,",",xhigh,")"

print "Estimate : Y"
print "Sample Estimate : ",y_bar
print "Sample Variance : ",y_S2
print "Confidence Interval for Y : (",ylow,",",yhigh,")"

