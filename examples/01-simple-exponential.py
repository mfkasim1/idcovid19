import pyro
import torch
from pyro.distributions import Uniform, Poisson, Normal, Binomial
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.nuts import NUTS
from pyro.infer import EmpiricalMarginal
import matplotlib.pyplot as plt

# data
ndays = 4
xdays = torch.arange(ndays)
infecteds = torch.tensor([80.0, 100, 125, 156])

def example(ndays, n0):
    r0 = pyro.sample("r0", Uniform(1.0, 3.0))
    nprev = n0
    ninfecteds = [n0] # the cummulative infected patients
    for i in range(1,ndays):
        # get the number of infected patients
        rate = nprev * r0
        nnew = pyro.sample("infected_%d"%i, Poisson(rate))
        ninfecteds.append(nnew)
        nprev = nnew

    return ninfecteds

n0 = infecteds[0]
data = {}
for i in range(1,ndays):
    data["infected_%d"%i] = infecteds[i]

conditioned_example = pyro.condition(example, data=data)
hmc_kernel = NUTS(conditioned_example, step_size=0.1)
posterior = MCMC(hmc_kernel,
                 num_samples=1000,
                 warmup_steps=50)
posterior.run(ndays, n0)
plt.hist(posterior.get_samples(1000)["r0"].numpy())
plt.title("P(r0 | n1 = 100)")
plt.xlabel("r0")
plt.ylabel("#")
plt.show()
