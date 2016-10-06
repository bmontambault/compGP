import numpy as np
import compGP

X=np.random.uniform(-3.,3.,(20,1))
Y=np.sin(X)+np.random.randn(20,1)*0.5
testX=np.linspace(-3,3)[None].T

m=compGP.RBF()
m.optimize(X,Y)

prior=m.prior(testX)
prior.plot_samples(nsamples=10)
prior.plot_mean()

posterior=m.posterior(X,Y,testX)
posterior.plot_samples(nsamples=10,train=(X,Y))
posterior.plot_mean(train=(X,Y))