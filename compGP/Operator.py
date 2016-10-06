from Model import CompModel
from BaseModels import W,S

class Operator:

	model_str_map={'S':S,'W':W}

	def __init__(self,m1,m2):
		self.m1=m1
		self.m2=m2
		self.S_count=m1.S_count+m2.S_count
		self.N_count=m1.N_count+m2.N_count
		self.index_stripped_names=[''.join([a for a in m.__name__ if not a.isdigit()]) for m in (m1,m2)]
		self.prods1,self.prods2=[[list(prod) for prod in n.split('_')] for n in self.index_stripped_names]
		self.m1_max_index=max([i for j in m1.indices for i in j])
		self.m2_start_index=self.m1_max_index+1
		self.m2_reindexed=[[i+self.m2_start_index for i in j] for j in m2.indices]

	def add_indices(self):
		unindexed_name='{0}{1}{2}'.format(self.index_stripped_names[0],'_',self.index_stripped_names[1])
		indices=self.m1.indices+self.m2_reindexed
		return unindexed_name,indices

	def mul_indices(self):
		unindexed_name=''.join([''.join(prod) for prod in [prodi+prodj+['_'] for prodi in self.prods1 for prodj in self.prods2]])[:-1]
		indices=[mi+mj for mi in self.m1.indices for mj in self.m2_reindexed]
		return unindexed_name,indices

	def build(self,op):
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
		mean_hyper_id,cov_hyper_id,log_transformation,index_map,parent_hyper_map=self.name_hyperparams(components,indexed_components)
		return type(name,(CompModel,),{'index_map':index_map,'parent_hyper_map':parent_hyper_map,'mean_hyper_id':mean_hyper_id,'cov_hyper_id':cov_hyper_id,'log_transformation':log_transformation,'indices':indices,'components':components,'indexed_components':indexed_components,'S_count':self.S_count,'N_count':self.N_count})

	def name_hyperparams(self,components,indexed_components):
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

		return mean_hyper_id,cov_hyper_id,log_transformation,name_map,parent_hyper_map

	def add(self):
		return self.build(self.add_indices)

	def mul(self):
		return self.build(self.mul_indices)