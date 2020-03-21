import pickle
import pyro
import torch
import numpy as np
from pyro.distributions import Uniform, Poisson, Normal, Binomial
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.nuts import HMC, NUTS
import matplotlib.pyplot as plt

def model(ndays, n0, infectious_delay, confirmed_delay, days_to_recover, days_to_deceased):
    # prior distribution
    r0 = pyro.sample("r0", Uniform(1.0, 3.0))
    confirmed_prob = pyro.sample("confirmed_prob", Uniform(0.0, 1.0))
    survival_rate = pyro.sample("survival_rate", Uniform(0.0, 1.0))

    # array for the simulation
    n0 = torch.tensor(n0*1.0)
    nprev = n0
    ninfecteds = [n0]*(infectious_delay+1) # the cummulative infected patients
    nnew_infecteds = [n0]+([0]*infectious_delay) # newly infected patient at day-i
    nconfirmeds = [torch.tensor(0.) for _ in range(ndays)] # the number of patients newly confirmed at day-i
    nsurvives = [torch.tensor(0.) for _ in range(ndays)] # the number of confirmed patients recovered at day-i
    ndeceaseds = [torch.tensor(0.) for _ in range(ndays)] # the number of confirmed patients deceased at day-i

    # simulate the infection and survival
    for i in range(1+infectious_delay,ndays):
        # get the number of infectious patients
        ninfectious = ninfecteds[i-1-infectious_delay]
        # get the mean number of infected patients
        rate = ninfectious * (r0-1)
        # the new patient infected this day
        # nnew = pyro.sample("infected_%d"%i, Poisson(rate))
        nnew = pyro.sample("infected_%d"%i, Normal(rate, torch.sqrt(rate))) # estimate poisson for large rate
        nnext = nprev + nnew
        ninfecteds.append(nnext)
        nnew_infecteds.append(nnew)

        # from the newly infected patient, map forward to determine when they
        # are confirmed positive
        # nadd_will_be_confirmed = pyro.sample("will_be_confirmed_from_%d"%i,
        #         Binomial(total_count=nnew, probs=confirmed_prob))
        # approximate binomial
        # nadd_will_be_confirmed = pyro.sample("will_be_confirmed_from_%d"%i,
        #         Normal(nnew*confirmed_prob, torch.sqrt(nnew*confirmed_prob*(1-confirmed_prob))))
        nadd_will_be_confirmed = nnew * confirmed_prob
        iconfirmed_day = i + confirmed_delay
        if iconfirmed_day < ndays:
            nconfirmeds[iconfirmed_day] = nconfirmeds[iconfirmed_day] + nadd_will_be_confirmed

        # map forward the newly confirmed patient predicting their survival
        nconf = nconfirmeds[i]
        if nconf > 0:
            # for the survivor
            # nconf_will_survive = pyro.sample("will_survive_from_%d"%i,
            #         Binomial(total_count=nconf, probs=survival_rate))
            # approximate binomial
            # nconf_will_survive = pyro.sample("will_survive_from_%d"%i,
            #         Normal(nconf*survival_rate, torch.sqrt(nconf*survival_rate*(1-survival_rate))))
            nconf_will_survive = nconf * survival_rate
            isurvive_day = i + days_to_recover
            if isurvive_day < ndays:
                nsurvives[isurvive_day] = nsurvives[isurvive_day] + nconf_will_survive

            # for the deceased
            nconf_will_deceased = nconf - nconf_will_survive
            ideceased_day = i + days_to_deceased
            if ideceased_day < ndays:
                ndeceaseds[ideceased_day] = ndeceaseds[ideceased_day] + nconf_will_deceased

        # prepare the variables for the next days
        nprev = nnext

    # accummulate the number of patients
    # use the trick from example 2 to make the nconfirmeds, nsurvives, ndeceaseds
    # differentiable and observable
    all_cum_confirmeds = []
    all_cum_survives = []
    all_cum_deceaseds = []
    cum_nconfirmeds = 0
    cum_nsurvives = 0
    cum_ndeceaseds = 0
    for i in range(ndays):
        cum_nconfirmeds = cum_nconfirmeds + nconfirmeds[i]
        cum_nsurvives = cum_nsurvives + nsurvives[i]
        cum_ndeceaseds = cum_ndeceaseds + ndeceaseds[i]
        rtol = 0.05
        atol = 1.0 # avoid std = 0 and capturing the uncertainty at the beginning
        pyro.sample("ncum_confirmeds_%d"%i, Normal(cum_nconfirmeds, cum_nconfirmeds*rtol+atol))
        pyro.sample("ncum_survives_%d"%i, Normal(cum_nsurvives, cum_nsurvives*rtol+atol))
        pyro.sample("ncum_deceaseds_%d"%i, Normal(cum_ndeceaseds, cum_ndeceaseds*rtol+atol))
        all_cum_confirmeds.append(cum_nconfirmeds)
        all_cum_survives.append(cum_nsurvives)
        all_cum_deceaseds.append(cum_ndeceaseds)

    # print(ninfecteds, all_cum_confirmeds, all_cum_survives, all_cum_deceaseds)
    return ninfecteds, all_cum_confirmeds, all_cum_survives, all_cum_deceaseds

# load the data
data = torch.tensor(np.loadtxt("data/data.csv", skiprows=20, delimiter=",", usecols=(1,2,3,4,5,6,7)), dtype=torch.float)
new_conf = data[:,1]
new_rec = data[:,2]
new_dec = data[:,3]
cum_conf = data[:,4]
cum_rec = data[:,5]
cum_dec = data[:,6]

n0 = 1
ndays = data.shape[0]
obs_data = {}
for i in range(ndays):
    if new_conf[i] > 0:
        obs_data["ncum_confirmeds_%d"%i] = cum_conf[i]
    obs_data["ncum_survives_%d"%i] = cum_rec[i]
    obs_data["ncum_deceaseds_%d"%i] = cum_dec[i]

# variables
confirmed_delay = 3
infectious_delay = 0
days_to_recover = 4 # from confirmed to recover
days_to_deceased = 3 # from confirmed to be deceased

conditioned_example = pyro.condition(model, data=obs_data)
hmc_kernel = NUTS(conditioned_example, step_size=0.1)
posterior = MCMC(hmc_kernel,
                 num_samples=1000,
                 warmup_steps=50)
posterior.run(ndays, n0, infectious_delay, confirmed_delay, days_to_recover, days_to_deceased)
samples = posterior.get_samples()
with open("posterior_samples.pkl", "wb") as fb:
    pickle.dump(samples, fb)

plt.hist(samples["confirmed_prob"].numpy())
plt.title("P(r0 | n1 = 100)")
plt.xlabel("r0")
plt.ylabel("#")
plt.show()
