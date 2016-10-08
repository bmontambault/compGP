import numpy as np
from scipy import optimize
import linalg
from Distributions import Normal,NormalMix

class BaseModel(object):

	indices=[[0]]
	K={}
	inv_K={}
	log_det_K={}
	df_K={}
	m={}
	log_transformation={'constant':False,'slope':False,'variance':True,'lengthscale':True}
	defaults={'constant':0,'slope':0,'varianlce':1,'lengthscale':1}

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
			self.m[key]=self.meanf(x)
			self.K[key]=self.covf(x,x)
			self.df_K[key]=self.df_covf(x,x)
			self.m[key].append(sum(self.m[key]))
			self.df_K[key].append(d for f in self.df_K[key])
			self.K[key],self.inv_K[key]=linalg.jit_inv(self.K[key])
			self.log_det_K[key]=[np.linalg.slogdet(K)[1] for K in self.K[key]]
		return key

	def prior(self,x):
		key=self.set_vars(x)
		return NormalMix([Normal(x,self.m[key][i],self.K[key][i],self.log_det_K[key][i],self.inv_K[key][i],self.df_K[key][i]) for i in xrange(len(self.m[key]))])

	def posterior(self,x,y,testx):
		key=self.set_vars(x)
		m=self.m[key]
		inv_K=self.inv_K[key]
		log_det_K=self.log_det_K[key]
		k=zip(*[self.covf(x_,x) for x_ in testx])
		kk=zip(*[self.covf(x_,x_) for x_ in testx])
		mx=zip(*[self.meanf(x_) for x_ in testx])
		k.append([sum(ki) for ki in zip(*k)])
		kk.append([sum(kki) for kki in zip(*kk)])
		mx.append([sum(mxi) for mxi in zip(*mx)])

		mean=[np.array([np.sum(k[i][j]*inv_K[i]*(y-m[i]))+np.sum(mx[i][j]) for j in xrange(len(k[i]))])[None].T for i in xrange(len(m))]
		var=[np.array([np.sum(kk[i][j]-np.sum(k[i][j]*inv_K[i]*k[i][j].T)) for j in xrange(len(k[i]))])[None].T for i in xrange(len(m))]
		return NormalMix([Normal(testx,mean[i],var[i]) for i in xrange(len(self.parameterized_components)+1)])


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

	def meanf(self,x):
		return [reduce(np.multiply,[c.meanf(x) for c in prod],1) for prod in self.parameterized_components]

	def covf(self,x1,x2):
		return [reduce(np.multiply,[c.covf(x1,x2) for c in prod],1) for prod in self.parameterized_components]

	def df_meanf(self):
		return [d for f in [[c.df_meanf(x) for c in prod] for prod in self.parameterized_components] for d in f]

	def df_covf(self,x1,x2):
		return [d for f in [[c.df_covf(x1,x2) for c in prod] for prod in self.parameterized_components] for d in f]