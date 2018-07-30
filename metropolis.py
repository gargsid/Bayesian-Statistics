import numpy as np
from GSN import GSN
import math
from scipy.integrate import quad, dblquad
from scipy.stats import nbinom,norm,geom,gamma
from scipy.stats import beta as betadis
from tqdm import tqdm
from scipy.special import beta as betafunc
from scipy.special import comb
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import invgamma,expon
import pickle
import sys
from random import randint

mu=0.0
sig=1.0
p=0.5
mu1=0
sig1=1
a=4
b=4
alpha=2
beta=1

# generate from geometric distribution	

def generate_GE(p):
	N = np.random.geometric(p=p) 
	return N

# generate from Geometric Skewed Distribution

def generate_GSN(n,mu,sig,p):
	X = []
	for i in range(n):		
		N = generate_GE(p)
		x = np.random.normal(loc=N*mu,scale=N*sig)	
		X.append(x)
	return np.array(X)

def gsn_likelihood(X,mu,sig,p):
	ret=1.0
	for x in X:
		val=0.0
		for t in range(1,100):
			val += norm.pdf(x,loc=t*mu,scale=math.sqrt(t)*sig)*geom.pmf(t,p)
			# print("norm=",norm.pdf(x,loc=t*mu,scale=math.sqrt(t)*sig),"geo=",p*(1-p)**(t-1))
		# print("val=",val)	
		if val>1e-5:
			ret*=val
	# print("ret=",ret)		
	if ret==1.0:
		return 0
	else:
		return ret	

def GSNpdf(x,mu,sig,p):
	val=0.0
	for t in range(1,100):
		val += norm.pdf(x,loc=mu*t,scale=math.sqrt(t)*sig)*geom.pmf(t,p)
	return val			

# proposal distribution for metropolis distribution from which we generate samples

def proposal(y1,y2,y3,x1,x2,x3,mu1,sig1,alpha,beta,a,b):
	prop = norm.pdf(y1,loc=x1,scale=0.01)*gamma.pdf(y2,x2*100,scale=0.01)*betadis.pdf(y3,100*x3,100*(1-x3))
	# print("prop=",prop)
	return prop

# target distribution from which we want to generate samples from

def target(X,y1,y2,y3,mu1,sig1,alpha,beta,a,b):
	tar = gsn_likelihood(X,y1,y2,y3)*norm.pdf(y1,loc=mu1,scale=sig1)*invgamma.pdf(y2,alpha,scale=beta)*betadis.pdf(y3,a,b)
	# print("tar=",gsn_likelihood(X,y1,y2,y3),norm.pdf(y1,loc=mu1,scale=sig1),invgamma.pdf(y2,alpha,scale=beta),betadis.pdf(y3,a,b))
	return tar

# acceptance probability of accepting a sample 

def acceptance(X,y1,y2,y3,x1,x2,x3,mu1,sig1,alpha,beta,a,b):
	r1 = (target(X,y1,y2,y3,mu1,sig1,alpha,beta,a,b)*1.0/target(X,x1,x2,x3,mu1,sig1,alpha,beta,a,b))
	r2 = (proposal(x1,x2,x3,y1,y2,y3,mu1,sig1,alpha,beta,a,b)*1.0/proposal(y1,y2,y3,x1,x2,x3,mu1,sig1,alpha,beta,a,b))
	# print("r1=",r1,"r2=",r2)
	ret = r1*r2
	# print("acceptance=",min(1,ret))
	return min(1,ret)

# We generate a sample from proposal distribution and accept it with the 'acceptance probability' so that the sample is 
# from the target distribution.

def metropolis_hastings(X,mu,sig,p,mu1,sig1,alpha,beta,a,b):
	samples = []
	x1 = mu
	x2 = sig
	x3 = p
	accepted=0
	rejected=0
	for i in range(100):
		y1 = np.random.normal(loc=x1,scale=0.01)
		# y2 = 1.0/np.random.gamma(shape=0.01/x2+1,scale=0.01)	
		y2 = np.random.gamma(shape= x2*100, scale = 0.01)
		y3 = np.random.beta(100*x3,100*(1-x3))
		# y2 = sig
		# y3 = p
		if i%10 == 0:
			print(i,y1,y2,y3)

		accept = acceptance(X,y1,y2,y3,x1,x2,x3,mu1,sig1,alpha,beta,a,b)
		
		u = np.random.uniform(low=0.0,high=1.0)
		# print("acceptance=",accept,"u=",u)
		if(u<accept):
			# print("accepted")
			accepted+=1
			if i > 20:
				print("Accepted")
				samples.append([y1,y2,y3])
			x1=y1
			x2=y2
			x3=y3
		else:
			rejected+=1
			# print("rejected")
	# print(accepted,rejected)				
	return samples



Data Sample Size
n=100
iter=20
estimates = np.full(3,0.0)
sample_variance = np.full(3,0.0)
confidence_intervals = np.full(3,0.0)
for i in range(iter):
	X = generate_GSN(n,mu,sig,p)
	samples = metropolis_hastings(X,mu,sig,p,mu1,sig1,alpha,beta,a,b)	
	estimates += np.sum(samples,axis=0)*1.0/len(samples)
	tmp=(samples-estimates)**2
	sample_variance += np.sum(tmp,axis=0)/(1.0*(len(samples)-1))
	confidence_intervals += (1.96/math.sqrt(len(samples)))*np.sqrt(sample_variance)
	with open('data'+str(i),'wb') as f:
		pickle.dump(X,f)
	with open('samples'+str(i),'wb') as f:
		pickle.dump(samples,f)

estimates/=(1.0*iter)	
sample_variance/=(1.0*iter)
confidence_intervals/=(1.0*iter)

for i in range(20):
	with open('data'+str(i),'rb') as f:
		X = pickle.load(f)
	with open('samples'+str(i),'rb') as f:
		samples = pickle.load(f)
	estimates = np.sum(samples,axis=0)*1.0/len(samples)		
	print(estimates)
	plt.figure()
	m, bins, patches = plt.hist(X, 50, normed=1, facecolor='g')
	y = []
	for bin in bins:
		y.append(GSNpdf( bin, estimates[0],estimates[1],estimates[2]))
	l = plt.plot(bins, y, color='black',linewidth=2.0)
	plt.grid(True,color='grey')
	plt.savefig('fitted_data'+str(i)+'.png',bbox_inches='tight')
	# plt.show()