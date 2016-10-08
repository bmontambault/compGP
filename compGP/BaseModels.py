import numpy as np
from Model import BaseModel

class W(BaseModel):
	mean_hyper_id=['constant']
	cov_hyper_id=['variance']
	W_count=1
	S_count=0
	N_count=0

	def meanf(self,x,summed=True):
		return np.ones(x.shape)*self.hyper_map['constant']

	def covf(self,x1,x2,summed=True):
		return np.array([[self.hyper_map['variance'] if xi==xj else 0 for xi in x1] for xj in x2]).T

	def df_meanf(self,x,summed=True):
		return [np.zeros(x.shape)]

	def df_covf(self,x1,x2,summed=True):
		return [np.zeros((x1.size,x2.size))]

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