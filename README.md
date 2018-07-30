# Geometric Skewed Normal Distribution

It is a three parameter distribution introduced in https://link.springer.com/article/10.1007/s13571-014-0082-y. This distribution is obtained by using geometric sum of independent identically distributed normal random variables. In this paper the EM algorithm has been proposed to compute the maximum likelihood estimators of the unknown parameters. 

In this project, we tried to find the Bayes estimates of the distribution using squared error loss function. The Bayes estimates in this case are mean of the posterior distribution of the parameters. In report, 'Bayesian Statistics, Section 4.1.1', we see that those cannot be calculated analytically. So we study different inference techniques in section 2. of the report. For the estimation of the parameters, we employed Metropolis-Hastings Algorithm to find the Bayes Estimates of the GSN distribution

File metropolis.py contains the implementation of the algorithm and the section 4.1.2 of the report has the complete algorithm.
