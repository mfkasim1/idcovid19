import numpy as np
import matplotlib.pyplot as plt
import emcee

paramnames = ["Offset days", "Init patients", "Infection rate", "Confirmed prob",
    "Recovery rate", "Infect delay mean", "Infect delay std",
    "Confirmed delay mean", "Confirmed delay std",
    "Days to recover mean", "Days to recover std",
    "Days to deceased mean", "Days to deceased std"]
idx_to_show = [1, 2, 3, 4, 5, 7, 9, 11]
filter_dict = {
    "high_recovery_rate": lambda chain: chain[:,4] > 0.94,
    "low_recovery_rate": lambda chain: chain[:,4] < 0.90,
    "low_confirmation": lambda chain: chain[:,3] < 0.65,
    "high_confirmation": lambda chain: chain[:,3] > 0.8,
}
# choosing which filter to use
filters = [
    # "high_recovery_rate",
    # "low_recovery_rate",
    # "low_confirmation",
    "high_confirmation",
]
bounds = np.array([
    [0.0, 10.0], # offset_days: offset days to the back
    [2.0, 100.0], # n0: the initial number of patient at day 0
    [1.0, 2.5], # r0: the infection rate
    [0.0, 1.0], # confirmed_prob1: the proportion of infected people that is confirmed
    [0.0, 1.0], # recovery_rate

    # day-related variables
    [1.0, 14.0], # infectious_delay_mean: incubation period, where the patient is not infectious
    [1.0, 10.0], # infectious_delay_std
    [1.0, 10.0], # confirmed_delay_mean: how many days since infected is confirmed (if it will be)
    [1.0, 10.0], # confirmed_delay_std
    [1.0, 20.0], # days_to_recover_mean: how many days since infected to be recovered (if will be confirmed)
    [1.0, 10.0], # days_to_recover_std
    [1.0, 10.0], # days_to_deceased_mean: how many days since infected to be deceased (if will be confirmed)
    [1.0, 10.0], # days_to_deceased_std
]) # (nfeat, 2)
reader = emcee.backends.HDFBackend("emcee_samples.h5", read_only=True)
flatchain = reader.get_chain(flat=True)
flatchain = flatchain[:flatchain.shape[0]//2,:] # (nsamples, nfeat)
flatchain = flatchain * (bounds[:,1] - bounds[:,0]) + bounds[:,0]

# filter the data
idx = flatchain[:,0] < np.inf
for filter_k in filters:
    newidx = filter_dict[filter_k](flatchain)
    idx = np.logical_and(idx, newidx)

fchain = flatchain[idx,:]

# print the summary
print("Data collected: %d" % flatchain.shape[0])
print("Filtered data: %d" % fchain.shape[0])
for i in range(len(paramnames)):
    print("%25s: (median) %.3e (std) %.3e" % (paramnames[i], np.median(fchain[:,i]), np.std(fchain[:,i])))

nrows = int(np.sqrt(len(idx_to_show)))
ncols = int(np.ceil(len(idx_to_show)*1.0 / nrows))
for i in range(len(idx_to_show)):
    idx = idx_to_show[i]
    plt.subplot(nrows, ncols, i+1)
    plt.hist(fchain[:,idx])
    plt.xlabel(paramnames[idx])
plt.show()
