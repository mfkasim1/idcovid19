from abc import abstractmethod, abstractproperty
import pickle
import numpy as np
import torch
import pyro
from pyro.distributions import Uniform, Normal
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.nuts import NUTS, HMC
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from idcovid19.utils.maxeig import maxeig

class BaseModel(object):
    def __init__(self, data, dtype=torch.float64):
        # data is a matrix with column: (day#, new_infect, new_rec, new_dec, cum_infect, cum_rec, cum_dec)
        self._dtype = dtype
        self.obs = self.get_observable(data)
        self.obsnames = ["gradient", "dec_by_rec", "dec_by_infection"]
        self.paramnames = list(self.prior.keys())
        self.nparams = len(self.paramnames)
        self.nobs = len(self.obsnames)

    @property
    def dtype(self):
        return self._dtype

    ################# model specification #################
    ##### parameter-related #####
    @abstractproperty
    def prior(self):
        # return a dictionary of paramname to distribution
        pass

    @abstractmethod
    def construct_jac(self, params):
        # params: dictionary with paramnames as keys and their values as the values
        pass

    @abstractproperty
    def display_fcn(self):
        # returns a dictionary with display name as the keys and a function of (params -> display params) as the values
        pass

    ##### state-related #####
    @abstractproperty
    def vecstate(self):
        # returns a dictionary with states as the key and the order in state vector as the values
        pass

    @abstractproperty
    def simdata_fcn(self):
        # returns a dictionary with key listed below and a function of (vec: (nparams,) tensor) -> a value as the values
        # key: enum("confirmed_case", "confirmed_death", "confirmed_recovery")
        pass

    ################# observation #################
    def get_simobservable(self, params):
        jac = self.construct_jac(params) # (nparams, nparams)
        max_eigval, max_eigvec, _ = maxeig.apply(jac)

        # calculate the observable
        gradient = max_eigval
        dec_by_rec = self.simdata_fcn["confirmed_death"](max_eigvec) / \
                     self.simdata_fcn["confirmed_recovery"](max_eigvec)
        dec_by_infection = self.simdata_fcn["confirmed_death"](max_eigvec) / \
                           self.simdata_fcn["confirmed_case"](max_eigvec)
        return (gradient, dec_by_rec, dec_by_infection)

    def get_observable(self, data):
        ninfectious = data[:,-3]
        nrec = data[:,-2]
        ndec = data[:,-1]
        ndays = data.shape[0]
        x = np.arange(ndays)

        # fit the infectious in the logplot
        logy = np.log(ninfectious)
        gradient, offset = np.polyfit(x, logy, 1)
        logyfit = offset + gradient * x
        std_gradient = np.sqrt(1./(x.shape[0]-2) * np.sum((logy - logyfit)**2) / np.sum((x-np.mean(x))**2))

        # the ratio of the graph
        dec_by_rec_mean = np.mean(ndec / nrec)
        dec_by_rec_std = np.std(ndec / nrec)
        dec_by_infection_mean = np.mean(ndec / ninfectious)
        dec_by_infection_std = np.std(ndec / ninfectious)

        # collect the distribution of the observation
        # obs_t_rec_total      = torch.tensor((18.0, 5.0))
        obs_gradient         = torch.tensor((gradient, std_gradient), dtype=self.dtype)
        obs_dec_by_rec       = torch.tensor((dec_by_rec_mean, dec_by_rec_std), dtype=self.dtype)
        obs_dec_by_infection = torch.tensor((dec_by_infection_mean, dec_by_infection_std), dtype=self.dtype)

        return (obs_gradient, obs_dec_by_rec, obs_dec_by_infection)

    ###################### util functions ######################
    def prior_params(self):
        # draw a sample from prior distribution of parameters
        return {name: pyro.sample(name, prior) for (name, prior) in self.prior.items()}

    def unpack(self, params):
        return [params[paramname] for paramname in self.paramnames]

    def inference(self, params=None): # a pytorch operation
        # get the parameters
        if params is None:
            params = self.prior_params()
        simobs = self.get_simobservable(params)
        obs = self.obs # (nobs, 2)

        logp = 0.0
        for i in range(self.nobs):
            dist = Normal(simobs[i], obs[i][1])
            pyro.sample(self.obsnames[i], dist, obs=obs[i][0])
            logp = logp + dist.log_prob(obs[i][0])
        return logp

    ###################### postprocess ######################
    def sample_observations(self, samples): # return a np.array (nobs, nsamples)
        nsamples = len(samples[self.paramnames[0]])
        simobs = []
        for i in range(nsamples):
            params = {name: samples[name][i] for name in self.paramnames}
            simobs.append(self.get_simobservable(params)) # (nsamples, nobs)
        simobs = list(zip(*simobs)) # (nobs, nsamples)
        return np.asarray(simobs)

    def filter_samples(self, samples, filters_dict, filters_keys):
        idx = samples[self.paramnames[0]] > -float("inf")
        for key in filters_keys:
            filter_fcn = filters_dict[key]
            idx = idx * filter_fcn(samples)
        new_samples = {}
        for name in self.paramnames:
            new_samples[name] = samples[name][idx]
        return new_samples

    def plot_obs_inferece(self, simobs):
        # simobs (nobs, nsamples)

        nobs = self.nobs
        obs = self.obs
        nrows = int(np.sqrt(nobs*1.0))
        ncols = int(np.ceil((nobs*1.0) / nrows))
        for i in range(nobs):
            plt.subplot(nrows, ncols, i+1)
            plt.hist(simobs[i])
            plt.axvline(float(obs[i][0]), color='C1', linestyle='-')
            plt.axvline(float(obs[i][0])-float(obs[i][1]), color='C1', linestyle='--')
            plt.axvline(float(obs[i][0])+float(obs[i][1]), color='C1', linestyle='--')
            plt.title(self.obsnames[i])
        plt.show()

    def plot_samples(self, samples):
        disp_names = list(self.display_fcn.keys())
        ndraw = len(disp_names)
        nrows = int(np.sqrt(ndraw*1.0))
        ncols = int(np.ceil((ndraw*1.0) / nrows))
        for i in range(ndraw):
            dispname = disp_names[i]
            samples_disp = self.display_fcn[dispname](samples)
            plt.subplot(nrows, ncols, i+1)
            plt.hist(samples_disp)
            plt.xlabel(dispname)
            print("%15s: (median) %.3e, (1sigma+) %.3e, (1sigma-) %.3e" % \
                 (dispname, np.median(samples_disp),
                  np.percentile(samples_disp, 86.1)-np.median(samples_disp),
                  np.median(samples_disp)-np.percentile(samples_disp, 15.9)))
        plt.show()
