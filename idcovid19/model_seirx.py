import pickle
import numpy as np
import torch
import pyro
from pyro.distributions import Uniform, Normal
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.nuts import NUTS, HMC
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from utils.maxeig import maxeig

dtype = torch.float64

class Model:
    def __init__(self, fdata="data/data.csv", day_offset=33):
        # vectors: exposed, infectious-dec, infectious-rec, dec, rec

        self.prior = {
            "r_incub":   Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "inf_rate":  Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "surv_rate": Uniform(torch.tensor(0.5 , dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "r_dec":     Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "r_rec":     Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
        }
        self.param_display = {
            "r_incub": (lambda t: 1./t, "t_incub"),
            "inf_rate": (lambda t: t, "inf_rate"),
            "surv_rate": (lambda t: t, "surv_rate"),
            "r_dec": (lambda t: 1./t, "t_dec"),
            "r_rec": (lambda t: 1./t, "t_rec"),
        }
        self.vecnames = {
            "exposed": 0,
            "infectious_dec": 1,
            "infectious_rec": 2,
            "dec": 3,
            "rec": 4,
        }
        self.obsnames = ["gradient", "dec_by_rec", "dec_by_infection"]
        self.paramnames = list(self.prior.keys())

        self.nparams = len(self.paramnames)
        self.nobs = len(self.obsnames)

        # load the data
        self.obs = self.get_observable(fdata, day_offset)

    ###################### model specification ######################
    def construct_jac(self, params):
        r_incub, inf_rate, surv_rate, r_dec, r_rec = self.unpack(params)

        nparams = self.nparams
        K_rate = torch.zeros(nparams, nparams).to(dtype)
        K_rate[0,0] = -r_incub
        K_rate[0,1] = inf_rate
        K_rate[0,2] = inf_rate
        K_rate[1,0] = (1-surv_rate)*r_incub
        K_rate[1,1] = -r_dec
        K_rate[2,0] = surv_rate*r_incub
        K_rate[2,2] = -r_rec
        K_rate[3,1] = r_dec
        K_rate[4,2] = r_rec

        # K_rate = np.asarray([
        #     [-1./t_incub, inf_rate, inf_rate, 0.0, 0.0], # dn(exposed)/dt
        #     [(1-surv_rate)/t_incub, -1./t_dec, 0.0, 0.0, 0.0], # dn(infectious-dec)/dt
        #     [surv_rate/t_incub, 0.0, -1./t_rec, 0.0, 0.0], # dn(infectious-rec)/dt
        #     [0.0, 1./t_dec, 0.0, 0.0, 0.0], # dn(dec)/dt
        #     [0.0, 0.0, 1./t_rec, 0.0, 0.0], # dn(rec)/dt
        # ]) # (nfeat,nfeat)

        return K_rate

    def r0(self, p):
        # p is a dictionary
        return (p["surv_rate"] / p["r_rec"] + (1-p["surv_rate"]) / p["r_dec"]) * p["inf_rate"]

    ###################### observation specification ######################
    def get_simobservable(self, params):
        jac = self.construct_jac(params) # (nparams, nparams)
        max_eigval, max_eigvec, _ = maxeig.apply(jac)

        # calculate the observable
        gradient = max_eigval
        dec_by_rec = max_eigvec[self.vecnames["dec"]] / max_eigvec[self.vecnames["rec"]]
        dec_by_infection = max_eigvec[self.vecnames["dec"]] / \
            (max_eigvec[self.vecnames["infectious_rec"]] + max_eigvec[self.vecnames["infectious_dec"]])
        return (gradient, dec_by_rec, dec_by_infection)

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
        # obs_t_rec_total      = torch.tensor((18.0, 5.0))
        obs_gradient         = torch.tensor((gradient, std_gradient), dtype=dtype)
        obs_dec_by_rec       = torch.tensor((dec_by_rec_mean, dec_by_rec_std), dtype=dtype)
        obs_dec_by_infection = torch.tensor((dec_by_infection_mean, dec_by_infection_std), dtype=dtype)

        return (obs_gradient, obs_dec_by_rec, obs_dec_by_infection)

    ###################### util functions ######################
    def prior_params(self):
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
    def sample_observations(self, samples):
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
        nkeys = self.nparams
        ndraw = nkeys + 1
        nrows = int(np.sqrt(ndraw*1.0))
        ncols = int(np.ceil((ndraw*1.0) / nrows))
        for i in range(nkeys):
            fcn_transform, dispname = self.param_display[self.paramnames[i]]
            samples_disp = np.asarray(fcn_transform(samples[self.paramnames[i]]))
            plt.subplot(nrows, ncols, i+1)
            plt.hist(samples_disp)
            plt.title(dispname)
            print("%15s: (median) %.3e, (1sigma+) %.3e, (1sigma-) %.3e" % \
                 (dispname, np.median(samples_disp),
                  np.percentile(samples_disp, 86.1)-np.median(samples_disp),
                  np.median(samples_disp)-np.percentile(samples_disp, 15.9)))

        # plot R0
        plt.subplot(nrows, ncols, nkeys+1)
        samples_disp = np.asarray(self.r0(samples))
        plt.hist(samples_disp)
        plt.title("R0")
        print("%15s: (median) %.3e, (1sigma+) %.3e, (1sigma-) %.3e" % \
             ("R0", np.median(samples_disp),
              np.percentile(samples_disp, 86.1)-np.median(samples_disp),
              np.median(samples_disp)-np.percentile(samples_disp, 15.9)))

        plt.show()

class Model2(Model):
    def __init__(self, fdata="data/data.csv", day_offset=33):
        self.prior = {
            "r_incub"        : Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "inf_rate_unconf": Uniform(torch.tensor(0.01, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "inf_rate_conf"  : Uniform(torch.tensor(0.01, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "surv_rate"      : Uniform(torch.tensor(0.5 , dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "r_conf"         : Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "r_dec_conf"     : Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "r_rec_conf"     : Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "r_dec_unconf"   : Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "r_rec_unconf"   : Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
        }
        self.param_display = {
            "r_incub"        : (lambda t: 1./t, "t_incub"),
            "inf_rate_unconf": (lambda t:    t, "inf_rate_unconf"),
            "inf_rate_conf"  : (lambda t:    t, "inf_rate_conf"),
            "surv_rate"      : (lambda t:    t, "surv_rate"),
            "r_conf"         : (lambda t: 1./t, "t_conf"),
            "r_dec_conf"     : (lambda t: 1./t, "t_dec_conf"),
            "r_rec_conf"     : (lambda t: 1./t, "t_rec_conf"),
            "r_dec_unconf"   : (lambda t: 1./t, "t_dec_unconf"),
            "r_rec_unconf"   : (lambda t: 1./t, "t_rec_unconf"),
        }
        self.vecnames = {
            "exposed": 0,
            "infectious_dec_unconf": 1,
            "infectious_rec_unconf": 2,
            "infectious_dec_conf": 3,
            "infectious_rec_conf": 4,
            "dec_conf": 5,
            "rec_conf": 6,
        }
        self.obsnames = ["gradient", "dec_by_rec", "dec_by_infection"]
        self.paramnames = list(self.prior.keys())

        self.nparams = len(self.paramnames)
        self.nobs = len(self.obsnames)

        # load the data
        self.obs = self.get_observable(fdata, day_offset)

    def r0(self, p):
        # p is a dictionary
        return (p["surv_rate"] / p["r_rec_conf"] + (1-p["surv_rate"]) / p["r_dec_conf"]) * p["inf_rate_conf"] # ???

    ###################### model specification ######################
    def construct_jac(self, params):
        r_incub, \
        inf_rate_unconf, \
        inf_rate_conf, \
        surv_rate, \
        r_conf, \
        r_dec_conf, \
        r_rec_conf, \
        r_dec_unconf, \
        r_rec_unconf = self.unpack(params)

        nparams = self.nparams
        K_rate = torch.zeros(nparams, nparams).to(dtype)
        K_rate[0,0] = -r_incub
        K_rate[0,1] = inf_rate_unconf
        K_rate[0,2] = inf_rate_unconf
        K_rate[0,3] = inf_rate_conf
        K_rate[0,4] = inf_rate_conf
        K_rate[1,0] = (1-surv_rate)*r_incub
        K_rate[1,1] = -r_dec_unconf - r_conf
        K_rate[2,0] = surv_rate*r_incub
        K_rate[2,2] = -r_rec_unconf - r_conf
        K_rate[3,1] = r_conf
        K_rate[3,3] = -r_dec_conf
        K_rate[4,2] = r_conf
        K_rate[4,4] = -r_rec_conf
        K_rate[5,3] = r_dec_conf
        K_rate[6,4] = r_rec_conf

        return K_rate

    ###################### observation specification ######################
    def get_simobservable(self, params):
        jac = self.construct_jac(params) # (nparams, nparams)
        max_eigval, max_eigvec, _ = maxeig.apply(jac)

        # calculate the observable
        gradient = max_eigval # the largest eigenvalue
        dec_by_rec = max_eigvec[self.vecnames["dec_conf"]] / max_eigvec[self.vecnames["rec_conf"]]
        dec_by_infection = max_eigvec[self.vecnames["dec_conf"]] / \
            (max_eigvec[self.vecnames["infectious_dec_conf"]] + max_eigvec[self.vecnames["infectious_rec_conf"]])
        return (gradient, dec_by_rec, dec_by_infection)

class Model3(Model2):
    def __init__(self, fdata="data/data.csv", day_offset=33):
        self.prior = {
            "r_incub"        : Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "r_rec_not_inf"  : Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "inf_rate_unconf": Uniform(torch.tensor(0.01, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "inf_rate_conf"  : Uniform(torch.tensor(0.01, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "surv_rate"      : Uniform(torch.tensor(0.5 , dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "r_conf"         : Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "r_dec_conf"     : Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "r_rec_conf"     : Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "r_dec_unconf"   : Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
            "r_rec_unconf"   : Uniform(torch.tensor(0.03, dtype=dtype), torch.tensor(1.0, dtype=dtype)),
        }
        self.param_display = {
            "r_incub"        : (lambda t: 1./t, "t_incub"),
            "r_rec_not_inf"  : (lambda t: 1./t, "t_rec_not_inf"),
            "inf_rate_unconf": (lambda t:    t, "inf_rate_unconf"),
            "inf_rate_conf"  : (lambda t:    t, "inf_rate_conf"),
            "surv_rate"      : (lambda t:    t, "surv_rate"),
            "r_conf"         : (lambda t: 1./t, "t_conf"),
            "r_dec_conf"     : (lambda t: 1./t, "t_dec_conf"),
            "r_rec_conf"     : (lambda t: 1./t, "t_rec_conf"),
            "r_dec_unconf"   : (lambda t: 1./t, "t_dec_unconf"),
            "r_rec_unconf"   : (lambda t: 1./t, "t_rec_unconf"),
        }
        self.vecnames = {
            "exposed": 0,
            "infectious_dec_unconf": 1,
            "infectious_rec_unconf": 2,
            "infectious_dec_conf": 3,
            "infectious_rec_conf": 4,
            "dec_conf": 5,
            "rec_conf": 6,
        }
        self.obsnames = ["gradient", "dec_by_rec", "dec_by_infection"]
        self.paramnames = list(self.prior.keys())

        self.nparams = len(self.paramnames)
        self.nobs = len(self.obsnames)

        # load the data
        self.obs = self.get_observable(fdata, day_offset)

    def construct_jac(self, params):
        r_incub, \
        r_rec_not_inf, \
        inf_rate_unconf, \
        inf_rate_conf, \
        surv_rate, \
        r_conf, \
        r_dec_conf, \
        r_rec_conf, \
        r_dec_unconf, \
        r_rec_unconf = self.unpack(params)

        nparams = self.nparams
        K_rate = torch.zeros(nparams, nparams).to(dtype)
        K_rate[0,0] = -r_incub - r_rec_not_inf
        K_rate[0,1] = inf_rate_unconf
        K_rate[0,2] = inf_rate_unconf
        K_rate[0,3] = inf_rate_conf
        K_rate[0,4] = inf_rate_conf
        K_rate[1,0] = (1-surv_rate)*r_incub
        K_rate[1,1] = -r_dec_unconf - r_conf
        K_rate[2,0] = surv_rate*r_incub
        K_rate[2,2] = -r_rec_unconf - r_conf
        K_rate[3,1] = r_conf
        K_rate[3,3] = -r_dec_conf
        K_rate[4,2] = r_conf
        K_rate[4,4] = -r_rec_conf
        K_rate[5,3] = r_dec_conf
        K_rate[6,4] = r_rec_conf

        return K_rate

class Model_A(Model):
    """
    Model 1 without the dec/infectious data
    """
    def __init__(self, *args, **kwargs):
        super(Model_A, self).__init__(*args, **kwargs)
        self.obsnames = self.obsnames[:2]
        self.nobs = len(self.obsnames)

    def get_simobservable(self, *args, **kwargs):
        return super(Model_A, self).get_simobservable(*args, **kwargs)[:2]

    def get_observable(self, *args, **kwargs):
        return super(Model_A, self).get_observable(*args, **kwargs)[:2]

class Model_B(Model):
    """
    Model 1 without the dec/rec data
    """
    def __init__(self, *args, **kwargs):
        super(Model_B, self).__init__(*args, **kwargs)
        self.obsnames = self.obsnames[::2]
        self.nobs = len(self.obsnames)

    def get_simobservable(self, *args, **kwargs):
        return super(Model_B, self).get_simobservable(*args, **kwargs)[::2]

    def get_observable(self, *args, **kwargs):
        return super(Model_B, self).get_observable(*args, **kwargs)[::2]

class Model_C(Model):
    """
    Model 1 without fitting the exponential gradient
    """
    def __init__(self, *args, **kwargs):
        super(Model_C, self).__init__(*args, **kwargs)
        self.obsnames = self.obsnames[1:]
        self.nobs = len(self.obsnames)

    def get_simobservable(self, *args, **kwargs):
        return super(Model_C, self).get_simobservable(*args, **kwargs)[1:]

    def get_observable(self, *args, **kwargs):
        return super(Model_C, self).get_observable(*args, **kwargs)[1:]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model1")
    parser.add_argument("--infer", action="store_const", default=False, const=True)
    parser.add_argument("--large", action="store_const", default=False, const=True)
    parser.add_argument("--nchains", type=int, default=1)
    parser.add_argument("--filters", type=str, nargs="*")
    args = parser.parse_args()

    # get the mode of operation
    if args.infer:
        mode = "infer"
    else:
        mode = "display"

    # get the sample size
    suffix = "" if args.nchains==1 else ("_%d"%args.nchains)
    if args.large:
        suffix = suffix + "_large"
        nsamples = 10000
        nwarmup = 500
    else:
        nsamples = 1000
        nwarmup = 50

    # choose model
    day_offset = 33
    if args.model == "model1":
        model = Model(day_offset=day_offset)
        samples_fname = "pyro_samples%s.pkl"%suffix
        filters_dict = {
            "low_infection_rate": lambda s: s["inf_rate"] < 0.5,
            "incubation_period_lt_14": lambda s: (s["r_incub"] > 1./14),
            "med_survive_rate": lambda s: s["surv_rate"] > 0.7,
            "r0_24": lambda s: model.r0(s) - 2 < 2,
        }
    elif args.model == "model2":
        model = Model2(day_offset=day_offset)
        samples_fname = "pyro_samples_model2%s.pkl"%suffix
        filters_dict = {
            "low_infection_rate": lambda s: s["inf_rate"] < 0.5,
            "med_survive_rate": lambda s: s["surv_rate"] > 0.7,
            "r0_24": lambda s: model.r0(s) - 2 < 2,
        }
    elif args.model == "model3":
        model = Model2(day_offset=day_offset)
        samples_fname = "pyro_samples_model3%s.pkl"%suffix
        filters_dict = {
            "med_survive_rate": lambda s: s["surv_rate"] > 0.7,
            "r0_24": lambda s: model.r0(s) - 2 < 2,
        }
    elif args.model == "modela":
        model = Model_A(day_offset=day_offset)
        samples_fname = "pyro_samples_modelA%s.pkl"%suffix
        filters_dict = {
            "low_infection_rate": lambda s: s["inf_rate"] < 0.5,
            "med_survive_rate": lambda s: s["surv_rate"] > 0.7,
            "r0_24": lambda s: model.r0(s) - 2 < 2,
        }
    elif args.model == "modelb":
        model = Model_B(day_offset=day_offset)
        samples_fname = "pyro_samples_modelB%s.pkl"%suffix
        filters_dict = {
            "low_infection_rate": lambda s: s["inf_rate"] < 0.5,
            "med_survive_rate": lambda s: s["surv_rate"] > 0.7,
            "r0_24": lambda s: model.r0(s) - 2 < 2,
            "slow_recovery": lambda s: s["r_rec"] < 1./7.,
        }
    elif args.model == "modelc":
        model = Model_C(day_offset=day_offset)
        samples_fname = "pyro_samples_modelC%s.pkl"%suffix
        filters_dict = {
            "low_infection_rate": lambda s: s["inf_rate"] < 0.5,
            "med_survive_rate": lambda s: s["surv_rate"] > 0.7,
            "r0_24": lambda s: model.r0(s) - 2 < 2,
            "slow_recovery": lambda s: s["r_rec"] < 1./7.,
        }


    if mode == "infer":
        hmc_kernel = NUTS(model.inference, step_size=0.1)
        posterior = MCMC(hmc_kernel,
                         num_samples=nsamples,
                         warmup_steps=nwarmup,
                         num_chains=args.nchains)
        posterior.run()
        samples = posterior.get_samples()
        with open(samples_fname, "wb") as fb:
            pickle.dump(samples, fb)

    with open(samples_fname, "rb") as fb:
        samples = pickle.load(fb)
    print("Collected %d samples" % len(samples[list(samples.keys())[0]]))

    filter_keys = args.filters
    if filter_keys is not None:
        # filter the samples
        samples = model.filter_samples(samples, filters_dict, filter_keys)
        print("Filtered into %d samples" % len(samples[list(samples.keys())[0]]))

    # simobs: (nobs, nsamples)
    simobs = model.sample_observations(samples)

    # plot the observation
    model.plot_obs_inferece(simobs)
    model.plot_samples(samples)
