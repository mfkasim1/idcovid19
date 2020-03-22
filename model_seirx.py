import pickle
import numpy as np
import torch
import pyro
from pyro.distributions import Uniform, Normal
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.nuts import NUTS, HMC
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from utils.eig import eig


class Model:
    def __init__(self, fdata="data/data.csv", day_offset=33):
        # vectors: exposed, infectious-dec, infectious-rec, dec, rec

        self.prior = {
            "t_incub": Uniform(0.1, 30.0),
            "inf_rate": Uniform(0.01, 2.0),
            "surv_rate": Uniform(0.01, 1.0),
            "t_dec": Uniform(0.1, 30.0),
            "t_rec": Uniform(0.1, 30.0),
        }
        self.vecnames = {
            "exposed": 0,
            "infectious_dec": 1,
            "infectious_rec": 2,
            "dec": 3,
            "rec": 4,
        }
        self.obsnames = ["t_rec_total", "gradient", "dec_by_rec", "dec_by_infection"]
        self.paramnames = list(self.prior.keys())

        self.nparams = len(self.paramnames)
        self.nobs = len(self.obsnames)

        # load the data
        self.obs = self.get_observable(fdata, day_offset)

    ###################### model specification ######################
    def construct_jac(self, params):
        t_incub, inf_rate, surv_rate, t_dec, t_rec = self.unpack(params)

        nparams = self.nparams
        K_rate = torch.zeros(nparams, nparams)
        K_rate[0,0] = -1./t_incub
        K_rate[0,1] = inf_rate
        K_rate[0,2] = inf_rate
        K_rate[1,0] = (1-surv_rate)/t_incub
        K_rate[1,1] = -1./t_dec
        K_rate[2,0] = surv_rate/t_incub
        K_rate[2,2] = -1./t_rec
        K_rate[3,1] = 1./t_dec
        K_rate[4,2] = 1./t_rec

        # K_rate = np.asarray([
        #     [-1./t_incub, inf_rate, inf_rate, 0.0, 0.0], # dn(exposed)/dt
        #     [(1-surv_rate)/t_incub, -1./t_dec, 0.0, 0.0, 0.0], # dn(infectious-dec)/dt
        #     [surv_rate/t_incub, 0.0, -1./t_rec, 0.0, 0.0], # dn(infectious-rec)/dt
        #     [0.0, 1./t_dec, 0.0, 0.0, 0.0], # dn(dec)/dt
        #     [0.0, 0.0, 1./t_rec, 0.0, 0.0], # dn(rec)/dt
        # ]) # (nfeat,nfeat)

        return K_rate

    ###################### observation specification ######################
    def get_simobservable(self, params):
        t_incub = params["t_incub"]
        t_rec = params["t_rec"]
        jac = self.construct_jac(params) # (nparams, nparams)
        eigvals, eigvecs = eig.apply(jac)
        max_eigvecs = eigvecs[:,-1] * torch.sign(eigvecs[-1,-1])

        # calculate the observable
        gradient = eigvals[-1] # the largest eigenvalue
        dec_by_rec = max_eigvecs[self.vecnames["dec"]] / max_eigvecs[self.vecnames["rec"]]
        dec_by_infection = max_eigvecs[self.vecnames["infectious_dec"]] / \
            (max_eigvecs[self.vecnames["infectious_rec"]] + max_eigvecs[self.vecnames["infectious_dec"]])
        return (t_incub+t_rec, gradient, dec_by_rec, dec_by_infection)

    def get_observable(self, fdata, day_offset):
        data0 = np.loadtxt(fdata, skiprows=1, delimiter=",", usecols=list(range(1,8))).astype(np.float32)
        data = data0[day_offset:,:]
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
        obs_t_rec_total      = torch.tensor((18.0, 5.0))
        obs_gradient         = torch.tensor((gradient, std_gradient))
        obs_dec_by_rec       = torch.tensor((dec_by_rec_mean, dec_by_rec_std))
        obs_dec_by_infection = torch.tensor((dec_by_infection_mean, dec_by_infection_std))

        return (obs_t_rec_total, obs_gradient, obs_dec_by_rec, obs_dec_by_infection)

    ###################### util functions ######################
    def prior_params(self):
        return {name: pyro.sample(name, prior) for (name, prior) in self.prior.items()}

    def unpack(self, params):
        return [params[paramname] for paramname in self.paramnames]

    def inference(self): # a pytorch operation
        # get the parameters
        params = self.prior_params()
        simobs = self.get_simobservable(params)
        obs = self.obs # (nobs, 2)

        for i in range(self.nobs):
            pyro.sample(self.obsnames[i], Normal(simobs[i], obs[i][1]), obs=obs[i][0])

    def plot_obs_inferece(self, samples):
        # samples is a dictionary with paramnames as keys and list of values
        # as the values
        nsamples = len(samples[self.paramnames[0]])
        simobs = []
        for i in range(nsamples):
            params = {name: samples[name][i] for name in self.paramnames}
            simobs.append(self.get_simobservable(params)) # (nsamples, nobs)
        simobs = list(zip(*simobs)) # (nobs, nsamples)

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
        nkeys = self.nparams
        nrows = int(np.sqrt(nkeys*1.0))
        ncols = int(np.ceil((nkeys*1.0) / nrows))
        for i in range(nkeys):
            plt.subplot(nrows, ncols, i+1)
            plt.hist(samples[self.paramnames[i]])
            plt.title(self.paramnames[i])
        plt.show()

if __name__ == "__main__":
    mode = "infer"
    mode = "display"
    samples_fname = "pyro_samples.pkl"
    day_offset = 33
    model = Model(day_offset=day_offset)

    if mode == "infer":
        hmc_kernel = NUTS(model.inference, step_size=0.1)
        posterior = MCMC(hmc_kernel,
                         num_samples=1000,
                         warmup_steps=50)
        posterior.run()
        samples = posterior.get_samples()
        with open(samples_fname, "wb") as fb:
            pickle.dump(samples, fb)

    with open(samples_fname, "rb") as fb:
        samples = pickle.load(fb)

    keys = list(samples.keys())
    nkeys = len(keys)

    # plot the observation
    model.plot_obs_inferece(samples)
    model.plot_samples(samples)
