import numpy as np
from scipy import optimize
import linalg
from Distributions import Normal

class Model(object):

	K={}
	inv_K={}
	log_det_K={}
	df_K={}
	m={}
	log_transformation={'constant':False,'slope':False,'variance':True,'lengthscale':True}

	def set_vars(self,x,summed=True):
		key=(tuple(np.ravel(x)),tuple(self.hyper_map.items()))
		if key not in self.K:
			K=self.covf(x,x,summed=summed)
			self.K[key],self.inv_K[key]=linalg.jit_inv(K)
			self.log_det_K[key]=np.linalg.slogdet(self.K[key])[1]
			self.df_K[key]=self.df_covf(x,x,summed=summed)
			self.m[key]=self.meanf(x,summed=summed)
		return key

	def prior(self,x,summed=True):
		key=self.set_vars(x,summed=summed)
		if summed==True:
			return Normal(x,self.m[key],self.K[key],self.log_det_K[key],self.inv_K[key],self.df_K[key])

	def posterior(self,x,y,testx,summed=True):
		key=self.set_vars(x,summed=summed)
		m,log_det_K,inv_K=self.m[key],self.log_det_K[key],self.inv_K[key]
		k=[self.covf(x_,x,summed=summed) for x_ in testx]
		kk=[self.covf(x_,x_,summed=summed) for x_ in testx]
		mx=[self.meanf(x_,summed=summed) for x_ in testx]

		if summed==True:
			mean=np.array([np.sum(k[i]*inv_K*(y-m))+np.sum(mx[i]) for i in xrange(len(k))])[None].T
			var=np.array([np.sum(kk[i]-np.sum(k[i]*inv_K*k[i].T)) for i in xrange(len(k))])[None].T
			return Normal(testx,mean,var)

	def f_df(self,x0,x,y):
		self.update_hyperparams(x0,log=True,fix_mean=True)
		prior=self.prior(x)
		f=-prior.log_likelihood(y)
		df=np.array([-p for p in prior.df_log_likelihood(y)])
		return f,df

	def optimize(self,x,y,verbose=False):
		x0=np.random.normal(size=len(self.cov_hyper_id))
		opt=optimize.minimize(self.f_df,x0,(x,y),jac=True,method='L-BFGS-B')
		if verbose==True:
			print opt
		self.update_hyperparams(opt.x,log=True,fix_mean=True)
