import numpy as np
from scipy import optimize
import linalg
from Distributions import Normal

class BaseModel(object):

	indices=[[0]]
	K={}
	inv_K={}
	log_det_K={}
	df_K={}
	m={}
	log_transformation={'constant':False,'slope':False,'variance':True,'lengthscale':True}
	defaults={'constant':0,'slope':0,'variance':1,'lengthscale':1}

	def __init__(self,hyperparams=None,fix_mean=False,log=False):
		self.update_hyperparams(hyperparams=hyperparams,fix_mean=fix_mean,log=log)

	def update_hyperparams(self,hyperparams=None,fix_mean=False,log=False):
		if fix_mean==True:
			self.input_hyperparams=self.cov_hyper_id
			if hasattr(hyperparams, '__iter__')==False:
				hyperparams=[self.defaults[h.split('_')[1]] for h in self.input_hyperparams]
			assert len(hyperparams)==len(self.input_hyperparams), 'mean fixed; expected {0} hyperparams'.format(len(self.input_hyperparams))
			try:
				self.hyper_map=self.hyper_map
			except:
				self.hyper_map={hyp:0 for hyp in self.mean_hyper_id}
		else:
			self.input_hyperparams=self.mean_hyper_id+self.cov_hyper_id
			if hasattr(hyperparams, '__iter__')==False:
				hyperparams=[self.defaults[h.split('_')[1]] for h in self.input_hyperparams]
			assert len(hyperparams)==len(self.input_hyperparams), 'expected {0} hyperparams'.format(len(self.input_hyperparams))
			self.hyper_map={}
		if type(hyperparams)==dict:
			for hyp in input_hyperparams:
				self.hyper_map[hyp]=hyperparams[hyp]
		else:
			for i in xrange(len(hyperparams)):
				self.hyper_map[self.input_hyperparams[i]]=hyperparams[i]
		if log==True:
			self.hyper_map={hyp:np.exp(self.hyper_map[hyp]) if self.log_transformation[hyp]==True else self.hyper_map[hyp] for hyp in self.hyper_map.keys()}

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


class CompModel(BaseModel):

	def update_hyperparams(self,hyperparams=None,fix_mean=False,log=False):
		super(CompModel,self).update_hyperparams(hyperparams=hyperparams,fix_mean=fix_mean,log=log)
		self.parameterized_components=[]
		parameterized_map={}
		for i in xrange(len(self.components)):
			self.parameterized_components.append([])
			for j in xrange(len(self.components[i])):
				if self.indexed_components[i][j] not in parameterized_map:
					base_hyperparams=[self.hyper_map[hyp] for hyp in self.parent_hyper_map[self.indexed_components[i][j]] if hyp in self.input_hyperparams]
					parameterized_component=self.components[i][j](base_hyperparams,fix_mean=fix_mean,log=log)
					parameterized_map[self.indexed_components[i][j]]=parameterized_component
				else:
					parameterized_component=parameterized_map[self.indexed_components[i][j]]
				self.parameterized_components[-1].append(parameterized_component)

	def meanf(self,x,summed=True):
		products=[reduce(np.multiply,[c.meanf(x) for c in prod],1) for prod in self.parameterized_components]
		if summed==True:
			return sum(products)
		else:
			return products

	def covf(self,x1,x2,summed=True):
		products=[reduce(np.multiply,[c.covf(x1,x2) for c in prod],1) for prod in self.parameterized_components]
		if summed==True:
			return sum(products)
		else:
			return products

	def df_meanf(self,x,summed=True):
		if summed==True:
			return [d for f in [c.df_meanf(x) for c in list(set([e for g in self.parameterized_components for e in g]))] for d in f]
		else:
			return [d for f in [[c.df_meanf(x) for c in prod] for prod in self.parameterized_components] for d in f]

	def df_covf(self,x1,x2,summed=True):
		if summed==True:
			return [d for f in [c.df_covf(x1,x2) for c in list(set([e for g in self.parameterized_components for e in g]))] for d in f]
		else:
			return [d for f in [[c.df_covf(x1,x2) for c in prod] for prod in self.parameterized_components] for d in f]