import numpy as np
from compGP import W,S,Operator

X=np.random.uniform(-3.,3.,(20,1))
Y=np.sin(X)+np.random.randn(20,1)*0.5
testX=np.linspace(-3,3)[None].T

WS=Operator(W,S).add()
ws=WS([0,0,1,1,1])
ws.optimize(X,Y,verbose=True)

posterior=ws.posterior(X,Y,testX)
posterior.plot_mean(train=(X,Y))

'''
s=compGP.S([0,1,1])
#s.optimize(X,Y)

prior=s.prior(testX)
prior.plot_samples(nsamples=10)
prior.plot_mean()

posterior=s.posterior(X,Y,testX)
posterior.plot_samples(nsamples=10,train=(X,Y))
posterior.plot_mean(train=(X,Y))
'''