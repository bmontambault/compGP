import numpy as np
import compGP

X=np.random.uniform(-3.,3.,(20,1))
Y=np.sin(X)+np.random.randn(20,1)*0.05
testX=np.linspace(-3,3)[None].T

m=compGP.RBF([0,0, 0.8151,1.8037,.0025])
#prior=m.prior(testX)
#prior.plot_samples(summed=False,nsamples=10)
posterior=m.posterior(X,Y,testX)
posterior.plot_mean(summed=True,train=(X,Y))