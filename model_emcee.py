from multiprocessing import Pool
import pickle
import numpy as np
from scipy.stats import poisson, binom, multivariate_normal
from scipy.special import logsumexp
import matplotlib.pyplot as plt

def model(ndays0,
          offset_days, # day offset to the left
          n0, # the initial number of patients at day-0
          r0, # infectious rate
          confirmed_prob, # the proportion of infected people that is confirmed
          recovery_rate, # recovery rate of the patients

          # day-related variables
          infectious_delay_mean, # incubation period, where the patient is not infectious
          infectious_delay_std,
          confirmed_delay_mean, #  how many days since infected is confirmed (if it will be)
          confirmed_delay_std,
          days_to_recover_mean, # how many days since infected to be recovered
          days_to_recover_std,
          days_to_deceased_mean, # how many days since infected to be deceased
          days_to_deceased_std
          ):
    # output: logprob of the data given the parameters
    ndays = ndays0 + int(offset_days)

    n0 = int(n0)
    nprev = n0
    ninfecteds = [n0] # the number of infected people at day-i
    ninfectious = [0 for _ in range(ndays)] # the number of people that becomes infectious at day-i
    nconfirmeds = [0 for _ in range(ndays)] # the number of patients newly confirmed at day-i
    nrecovereds = [0 for _ in range(ndays)] # the number of confirmed patients recovered at day-i
    ndeceaseds = [0 for _ in range(ndays)] # the number of confirmed patients deceased at day-i

    # the delay distribution
    inf_delay = multivariate_normal(infectious_delay_mean, infectious_delay_std**2)
    conf_delay = multivariate_normal(confirmed_delay_mean, confirmed_delay_std**2)
    conf_to_rec = multivariate_normal(days_to_recover_mean, days_to_recover_std**2)
    conf_to_dec = multivariate_normal(days_to_deceased_mean, days_to_deceased_std**2)

    # day 0, map the newly infected people to the infectious
    ninfectious = map_forward(0, ninfectious, n0, inf_delay)

    # simulate the infection and recovery from day 1 and so on
    for i in range(1,ndays):
        # get the number of infectious patients
        ninfcious = np.sum(ninfectious[:i+1]) - np.sum(nrecovereds[:i+1]) - np.sum(ndeceaseds[:i+1])
        if ninfcious < 0: ninfcious = 0
        # get the mean number of infected patients
        rate = ninfcious * (r0-1.0)
        nnew = poisson.rvs(rate)
        ninfecteds.append(nnew)

        # map the newly infected patient to the infectious phase
        ninfectious = map_forward(i, ninfectious, nnew, inf_delay)

        # from the newly infected patient, map forward to determine when they
        # are confirmed positive
        nadd_will_be_confirmed = binom.rvs(nnew, confirmed_prob)
        nconfirmeds = map_forward(i, nconfirmeds, nadd_will_be_confirmed, conf_delay)

        # map forward the newly confirmed patient predicting their recovery
        nconf = nadd_will_be_confirmed
        # nconf = nconfirmeds[i]
        nconf_will_recovered = binom.rvs(nconf, recovery_rate)
        nrecovereds = map_forward(i, nrecovereds, nconf_will_recovered, conf_to_rec)
        # and the deceased ...
        nconf_will_deceased = nconf - nconf_will_recovered
        ndeceaseds = map_forward(i, ndeceaseds, nconf_will_deceased, conf_to_dec)

    # get the accummulated numbers
    cum_confirmeds = np.cumsum(nconfirmeds)
    cum_recovereds = np.cumsum(nrecovereds)
    cum_deceaseds = np.cumsum(ndeceaseds)

    return np.asarray(nconfirmeds[-ndays0:]), np.asarray(nrecovereds[-ndays0:]), np.asarray(ndeceaseds[-ndays0:])

def map_forward(i, arr, ntot, day_dist):
    if ntot == 0: return arr
    day_idxs = i + np.maximum(day_dist.rvs(size=ntot), 1)
    if ntot == 1:
        day_idxs = [day_idxs]
    for day_idx in day_idxs:
        if day_idx < len(arr):
            arr[int(day_idx)] += 1
    return arr

# wrapper of the model for emcee
def logprob_model(x, lb, ub, obs_data, ndays, nrepeat=1000, plot=False):
    # x is in range (0,1)
    # obs_data: (nobs, ndays)
    if np.any(x < 0) or np.any(x > 1):
        return -9e99

    obs_data = np.asarray(obs_data)
    nobs = obs_data.shape[0]
    simdata = []
    xparam = x * (ub - lb) + lb
    for i in range(nrepeat):
        sim = model(ndays, *xparam)
        simdata.append(sim)

    simdata = np.asarray(simdata) # (nsim, nobs, ndays)
    var_obs = (obs_data * .1 + 2.0)**2 # (nobs, ndays)
    mean_simdata = np.mean(simdata, axis=0) # (nobs, ndays)
    simdata_dist = []
    covs = []
    for i in range(nobs):
        cov = np.cov(simdata[:,i,:], rowvar=False) # (ndays, ndays)
        # cov *= np.eye(cov.shape[0]) * 1.0
        cov += np.eye(cov.shape[0]) * var_obs[i,:]
        covs.append(cov)
        simdata_dist.append(multivariate_normal(mean_simdata[i,:], cov))

    logprob = 0.0
    for i in range(nobs):
        logprob += simdata_dist[i].logpdf(obs_data[i,:])
    # logprob = logsumexp(lps) - np.log(nrepeat)
    print(x, logprob)

    if plot:
        x = np.arange(simdata.shape[-1])
        for i in range(nobs):
            plt.subplot(1,nobs,i+1)
            plt.plot(x, mean_simdata[i,:], 'C%d-'%i)
            plt.fill_between(x, mean_simdata[i,:]-np.diag(covs[i])**.5, mean_simdata[i,:]+np.diag(covs[i])**.5, alpha=0.3)
            plt.plot(x, simdata[:,i,:].T, 'C%d-'%i, alpha=0.01)
            plt.plot(x, obs_data[i,:], 'C%do'%i)
        plt.show()
    return logprob

# load the data
data = np.loadtxt("data/data.csv", skiprows=20, delimiter=",", usecols=(1,2,3,4,5,6,7))
new_conf = data[:,1]
new_rec = data[:,2]
new_dec = data[:,3]
cum_conf = data[:,4]
cum_rec = data[:,5]
cum_dec = data[:,6]
ndays = len(data)

# parameters for the model
# (initval, lbound, ubound)
params = np.array([
    [5.0000000 , 0.0, 10.0], # offset_days: offset days to the back
    [10.0000000 , 2.0, 100.0], # n0: the initial number of patient at day 0
    [2.43010937 , 1.0, 2.5], # r0: the infection rate
    [0.8955862 , 0.0, 1.0], # confirmed_prob1: the proportion of infected people that is confirmed
    [0.32255113 , 0.0, 1.0], # recovery_rate

    # day-related variables
    [1.52957385 , 1.0, 14.0], # infectious_delay_mean: incubation period, where the patient is not infectious
    [1.97515711 , 1.0, 10.0], # infectious_delay_std
    [1.468378 , 1.0, 10.0], # confirmed_delay_mean: how many days since infected is confirmed (if it will be)
    [6.08453398 , 1.0, 10.0], # confirmed_delay_std
    [13.54482505, 1.0, 20.0], # days_to_recover_mean: how many days since infected to be recovered (if will be confirmed)
    [2.14816375 , 1.0, 10.0], # days_to_recover_std
    [6.07907008, 1.0, 10.0], # days_to_deceased_mean: how many days since infected to be deceased (if will be confirmed)
    [3.64004245 , 1.0, 10.0], # days_to_deceased_std
])
# better fit
pp0 = np.array([float(p) for p in """0.0 0.39487316 0.48078534 0.84274779 0.89724724 0.57197012 0.80188228
 0.21142343 0.6033941  0.99233592 0.48549393 0.54471405 0.25049241""".split()]) # -125.9
params[:,0] = pp0 * (params[:,2]-params[:,1]) + params[:,1]
# pp0 = np.array([2., 1.58677579,  0.95209077,  0.56049289,  2.54274016,  6.80718115**.5,
#         1.39511485,  5.53546927**.5, 10.07925279,  6.08758741**.5,  6.2961364 ,
#         2.40890077**.5])
# params[:,0] = pp0
print(params[:,0])

# combine the arguments
p0 = params[:,0]
lb = params[:,1]
ub = params[:,2]
x0 = (p0 - lb) / (ub - lb)
obs_data = (new_conf, new_rec, new_dec)

ndim = p0.shape[0]
nwalkers = ndim * 3
nrepeat = 2000
mode = "see"

if mode == "see":
    logprob = logprob_model(x0, lb, ub, obs_data, ndays, nrepeat=nrepeat//3, plot=True)
    print(logprob)
elif mode == "opt":
    import cma

    cma_options = cma.CMAOptions()
    cma_options.set("popsize", 16)
    es = cma.CMAEvolutionStrategy(x0*0+0.5, 0.3, cma_options)
    es.optimize(lambda x: -logprob_model(x, lb, ub, obs_data, ndays, nrepeat=nrepeat))
    es.result_pretty()

elif mode == "mcmc":
    import emcee

    filename = "emcee_samples.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    xseed0 = x0 + np.random.randn(nwalkers, ndim) * 0.1
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob_model,
                                        args=[lb, ub, obs_data, ndays, nrepeat],
                                        backend=backend, pool=pool)
        sampler.run_mcmc(xseed0, 10000, progress=True)
