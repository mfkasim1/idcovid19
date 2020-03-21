import pyro
import torch
from pyro.distributions import Uniform, Poisson, Normal, Binomial
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.nuts import NUTS
from pyro.infer.discrete import infer_discrete
import matplotlib.pyplot as plt

@infer_discrete(first_available_dim=-1, temperature=0)
def example2():
    # a = pyro.sample("a", Uniform(1.0, 3.0))
    # b = pyro.sample("b", Uniform(1.2, 3.2))
    a = pyro.sample("a", Poisson(3.0), infer={"enumerate":"sequential"})
    b = pyro.sample("b", Poisson(3.0), infer={"enumerate":"sequential"})
    c = pyro.sample("c", Normal(a + b, 0.1))
    return c

conditioned_example = pyro.condition(example2, data={"c": torch.tensor(6.0)})
hmc_kernel = NUTS(conditioned_example, step_size=1.0)
posterior = MCMC(hmc_kernel,
                 num_samples=1000,
                 warmup_steps=500)
posterior.run()
plt.hist(posterior.get_samples(1000)["a"].numpy())
plt.title("P(a)")
plt.xlabel("a")
plt.ylabel("#")
plt.show()
