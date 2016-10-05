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

	def operation(self,m2,op):
		S_count=self.S_count+m2.S_count
		N_count=self.N_count+m2.N_count
		index_stripped_names=[''.join([a for a in m.__name__ if not a.isdigit()]) for m in (self,m2)]
		prods1,self.prods2=[[list(prod) for prod in n.split('_')] for n in self.index_stripped_names]
		m1_max_index=max([i for j in self.indices for i in j])
		m2_start_index=m1_max_index+1
		m2_reindexed=[[i+m2_start_index for i in j] for j in m2.indices]
		if op=='+':
			unindexed_name='{0}{1}{2}'.format(self.index_stripped_names[0],'_',self.index_stripped_names[1])
			indices=self.indices+self.m2_reindexed
		elif op=='*':
			unindexed_name=''.join([''.join(prod) for prod in [prodi+prodj+['_'] for prodi in prods1 for prodj in prods2]])[:-1]
			indices=[mi+mj for mi in self.indices for mj in m2_reindexed]

		index_map={}
		Si=0
		Ni=0
		Wi=0
		name=''
		unindexed_name,indices=op()
		str_components=[list(prod) for prod in unindexed_name.split('_')]
		for i in xrange(len(indices)):
			for j in xrange(len(indices[i])):
				if str_components[i][j]=='S':
					if indices[i][j] not in index_map:
						Si+=1
						index_map[indices[i][j]]=Si
				elif str_components[i][j]=='N':
					if indices[i][j] not in index_map:
						Ni+=1
						index_map[indices[i][j]]=Ni
				elif str_components[i][j]=='W':
					if indices[i][j] not in index_map:
						Wi+=1
						index_map[indices[i][j]]=Wi
				name+=str_components[i][j]+str(index_map[indices[i][j]])
			name+='_'
		name=name[:-1]
		components=[[self.model_str_map[c] for c in prod] for prod in str_components]
		indexed_components=[[prod[i:i+2] for i in xrange(0,len(prod),2)] for prod in name.split('_')]

		log_transformation={}
		flat_components=[x for y in components for x in y]
		flat_indexed_components=[x for y in indexed_components for x in y]
		mean_hyper_id=[]
		cov_hyper_id=[]
		parent_hyper_map={}
		name_map={flat_indexed_components[i]:flat_components[i] for i in xrange(len(flat_components))}
		parent_hyper_map={}
		for name in name_map.keys():
			mean_ids=[name+'_'+hyp for hyp in name_map[name].mean_hyper_id]
			mean_hyper_id+=mean_ids
			cov_ids=[name+'_'+hyp for hyp in name_map[name].cov_hyper_id]
			cov_hyper_id+=cov_ids
			parent_hyper_map[name]=mean_ids+cov_ids
			for i in xrange(len(mean_ids+cov_ids)):
				log_transformation[(mean_ids+cov_ids)[i]]=name_map[name].log_transformation[(name_map[name].mean_hyper_id+name_map[name].cov_hyper_id)[i]]
		return type(name,(CompModel,),{'index_map':index_map,'parent_hyper_map':parent_hyper_map,'mean_hyper_id':mean_hyper_id,'cov_hyper_id':cov_hyper_id,'log_transformation':log_transformation,'indices':indices,'components':components,'indexed_components':indexed_components,'S_count':self.S_count,'N_count':self.N_count})

		def __add__(self,m2):
			return self.operation(m2,'+')

		def __mul__(self,m2):
			return self.operation(m2,'*')


class CompModel(BaseModel):

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