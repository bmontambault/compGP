import numpy as np

def jit_inv(covs,start=.00001,eps=.0001,max_tries=100):
	covs.append(sum(covs))
	jit_K=[]
	jit_inv_K=[]
	for K in covs:
		jitter=start
		tries=0
		for i in xrange(max_tries):
			try:
				K=K+jitter*np.identity(np.sqrt(K.size))
				inv_K=np.linalg.inv(K)
				jit_K.append(K)
				jit_inv_K.append(inv_K)
				break
			except:
				jitter+=eps
				tries+=1
				if i==max_tries:
					raise ValueError('too many tries')
	return jit_K,jit_inv_K