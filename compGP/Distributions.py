import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

class Normal:

	def __init__(self,x,mean,K,log_det_K=None,inv_K=None,df_K=None):
		self.x=x
		self.K=K
		self.log_det_K=log_det_K
		self.inv_K=inv_K
		self.df_K=df_K
		self.mean=mean
		if self.K.shape[0]!=self.K.shape[1]:
			self.K=np.identity(self.K.shape[0])*self.K

	def log_likelihood(self,y):
		if hasattr(self.inv_K, '__iter__'):
			return -1/2.*self.log_det_K-np.sum(1/2.*np.dot(np.dot((y-self.mean).T,self.inv_K),(y-self.mean)))-(len(y)/2.)*np.log(2*np.pi)
		else:
			var=np.diag(self.K)
			return reduce(np.multiply,[-1/2.*np.log(2*np.pi)-1/2.*np.log(var[i])-(1/2*var[i])*(y[i]-self.mean[i]) for i in xrange(len(y))],1)

	def df_log_likelihood(self,y):
		a=np.dot(self.inv_K,y)
		return [np.sum(1/2.*np.trace(np.dot((np.dot(a,a.T)-self.inv_K),df))) for df in self.df_K]

	def sample(self,nsamples=1):
		return [np.random.multivariate_normal(np.ravel(self.mean),self.K)[None].T for i in xrange(nsamples)]

	def plot_samples(self,nsamples=1,show=True,color='b',train=None):
		samples=self.sample(nsamples)
		likelihoods=[self.log_likelihood(y) for y in samples]
		exponentiated_likelihoods=[np.exp(l-max(likelihoods)) for l in likelihoods]
		normalized=[e/sum(exponentiated_likelihoods) for e in exponentiated_likelihoods]
		alpha=[n if n>.2 else .2 for n in normalized]
		for i in xrange(len(samples)):
			plt.plot(self.x,samples[i],color,alpha=alpha[i])
		if train!=None:
			plt.plot(train[0],train[1],'ro')
		if show==True:
			plt.show()

	def plot_mean(self,show=True,train=None):
		var=np.diag(self.K)
		std=[np.sqrt(v) for v in var]
		low=[np.sum(self.mean[i]-2*std[i]) for i in xrange(len(self.mean))]
		high=[np.sum(self.mean[i]+2*std[i]) for i in xrange(len(self.mean))]
		plt.plot(self.x,self.mean)
		plt.fill_between(np.ravel(self.x),low,high,alpha=.2)
		if train!=None:
			plt.plot(train[0],train[1],'ro')
		if show==True:
			plt.show()