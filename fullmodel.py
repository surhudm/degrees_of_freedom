import numpy as np 
import pylab as pl
from scipy.optimize import leastsq
from scipy.stats import chi2


class linear_model:
    def __init__(self, a=10, b=3, Ndata=20, sigi=0.1, siga_prior=0.0000001, Nreal=10000):
        self.a = a
        self.b = b
        self.Ndata = Ndata
        self.sigi = sigi
        self.siga_prior = siga_prior
        self.Nreal = Nreal

    def func_data(self, p0, xi, yi, sigi):
        a0, b0 = p0
        ymodel = a0 + b0*xi
        datachi = (yi-ymodel)/sigi
    
        return datachi
    
    def func(self, p0, xi, yi, sigi, a_prior, siga_prior):
        a0, b0 = p0
    
        datachi = self.func_data(p0, xi, yi, sigi)
        priorchi = np.array([(a0-a_prior)/siga_prior])
    
        return np.concatenate([datachi, priorchi])
    
    def get_bestfit_chisq(self):
    
        # Generate data according to the linear model with errorbar
        xi = np.arange(self.Ndata)
        yi = self.a + self.b * xi + np.random.normal(size=self.Ndata)*self.sigi
    
        # Generate prior parameter according to the siga_prior
        a_prior = self.a + np.random.normal()*self.siga_prior
        
        p0 = [self.a, self.b]
        popt = leastsq(self.func, p0, args=(xi, yi, self.sigi, a_prior, self.siga_prior))
    
        return np.sum(self.func(popt[0], xi, yi, self.sigi, a_prior, self.siga_prior)**2), np.sum(self.func_data(popt[0], xi, yi, self.sigi)**2)

    def make_plots(self, ax1, ax2):
        chisq = np.zeros(self.Nreal)
        chisq_data = np.zeros(self.Nreal)
        import frogress
        for ii in range(self.Nreal):
            chisq[ii], chisq_data[ii] = self.get_bestfit_chisq()
        
        n, bins, patch = ax1.hist(chisq, histtype="step", normed=1, bins=100, label="Posterior chisq")
        x = bins[1:]/2+bins[:-1]/2
        ax1.plot(x, chi2.pdf(x, self.Ndata-2), label=r"$\chi^2(N_{\rm data}-2)$")
        ax1.plot(x, chi2.pdf(x, self.Ndata-1), label=r"$\chi^2(N_{\rm data}-1)$")
        ax1.plot(x, chi2.pdf(x, self.Ndata), label=r"$\chi^2(N_{\rm data})$")
        ax1.legend()
        
        
        n, bins, patch = ax2.hist(chisq_data, histtype="step", normed=1, bins=bins, label="Likelihood chisq")
        x = bins[1:]/2+bins[:-1]/2
        ax2.plot(x, chi2.pdf(x, self.Ndata-2), label=r"$\chi^2(N_{\rm data}-2)$")
        ax2.plot(x, chi2.pdf(x, self.Ndata-1), label=r"$\chi^2(N_{\rm data}-1)$")
        ax2.plot(x, chi2.pdf(x, self.Ndata), label=r"$\chi^2(N_{\rm data})$")
        ax2.legend()

        return 
        

# This is the case where the posterior of the nuisance parameter is driven by the prior
ax1 = pl.subplot(221)
ax2 = pl.subplot(222)
a = linear_model()
a.make_plots(ax1, ax2)
pl.savefig("Prior_driven.pdf")
pl.clf()

# This is the case where the posterior of the nuisance parameter is driven by the data
ax1 = pl.subplot(223)
ax2 = pl.subplot(224)
a = linear_model(siga_prior=1.e5)
a.make_plots(ax1, ax2)
pl.savefig("Data_driven.pdf")
pl.clf()
