import numpy as np
from ModelClass import Model

class BaseModel(Model):

	indices=[[0]]

	def __init__(self,hyperparams,fix_mean=False,log=False):
		self.update_hyperparams(hyperparams,fix_mean=fix_mean,log=log)

	def update_hyperparams(self,hyperparams,fix_mean=False,log=False):
		if fix_mean==True:
			assert len(hyperparams), 'mean fixed; expected {0} hyperparams'.format(len(self.cov_hyper_id))
			input_hyperparams=self.cov_hyper_id
			try:
				self.hyper_map=self.hyper_map
			except:
				self.hyper_map={hyp:0 for hyp in self.mean_hyper_id}
		else:
			assert len(hyperparams)==len(self.mean_hyper_id)+len(self.cov_hyper_id), 'expected {0} hyperparams'.format(len(self.mean_hyper_id)+len(self.cov_hyper_id))
			input_hyperparams=self.mean_hyper_id+self.cov_hyper_id
			self.hyper_map={}
		if type(hyperparams)==dict:
			for hyp in input_hyperparams:
				self.hyper_map[hyp]=hyperparams[hyp]
		else:
			for i in xrange(len(hyperparams)):
				self.hyper_map[input_hyperparams[i]]=hyperparams[i]
		if log==True:
			self.hyper_map={hyp:np.exp(self.hyper_map[hyp]) if self.log_transformation[hyp]==True else self.hyper_map[hyp] for hyp in self.hyper_map.keys()}

class W(BaseModel):
	mean_hyper_id=['constant']
	cov_hyper_id=['variance']
	W_count=1
	S_count=0
	N_count=0

	def meanf(self,x,summed=True):
		return np.ones(x.shape)*self.hyper_map['constant']

	def covf(self,x1,x2,summed=True):
		mat=x1*x2.T
		if 1 in mat.shape:
			return np.ones(mat.shape)*self.hyper_map['variance']
		else:
			return np.identity(np.sqrt(mat.size))*self.hyper_map['variance']

	def df_meanf(self,x,summed=True):
		return [np.zeros(x.shape)]

	def df_covf(self,x1,x2,summed=True):
		return [np.zeros((x1*x1.T).shape)]

class S(BaseModel):
	mean_hyper_id=['constant']
	cov_hyper_id=['variance','lengthscale']
	S_count=1
	N_count=0
	W_count=0

	def meanf(self,x,summed=True):
		return self.hyper_map['constant']*np.ones(x.size)[None].T

	def covf(self,x1,x2,summed=True):
		return self.hyper_map['variance']*np.exp(-(x1-x2.T)**2/(2*self.hyper_map['lengthscale']**2))

	def df_meanf(self,x,summed=True):
		return [np.zeros(x.size)[None].T]

	def df_covf(self,x1,x2,summed=True):
		return [np.exp(-(x1-x2.T)**2/(2*self.hyper_map['lengthscale']**2)),
				self.hyper_map['variance']*(x1-x2.T)**2*np.exp(-(x1-x2.T)**2/(2*self.hyper_map['lengthscale']**2))/self.hyper_map['lengthscale']**3]