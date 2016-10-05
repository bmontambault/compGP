import numpy as np

def jit_inv(K,jitter=.00001,eps=.0001,max_tries=100):
	tries=0
	while tries<max_tries:
		try:
			K+=jitter*np.identity(np.sqrt(K.size))
			inv_K=np.linalg.inv(K)
			return K,inv_K
		except:
			jitter+=eps
			tries+=1
	raise ValueError('too many tries')