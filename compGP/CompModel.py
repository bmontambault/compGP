import numpy as np
from ModelClass import Model
from BaseModel import S,S

class CompModel(Model):

	def update_hyperparams(self,hyperparams=None,fix_mean=False,log=False):
		super(CompModel,self).update_hyperparams(hyperparams=hyperparams,fix_mean=fix_mean,log=log)
		self.parameterized_components=[]
		parameterized_map={}
		for i in xrange(len(self.components)):
			self.parameterized_components.append([])
			for j in xrange(len(self.components[i])):
				if self.indexed_components[i][j] not in parameterized_map:
					base_hyperparams=[self.hyper_map[hyp] for hyp in self.parent_hyper_map[self.indexed_components[i][j]] if hyp in input_hyperparams]
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
