import torch
from pyro.distributions import Uniform, Normal
from idcovid19.models.base_model import BaseModel
from idcovid19.utils.misc import memoize

class Model2(BaseModel):
    """
    Model 2 is a modification of SEIR model with path splitting into two and
    isolated patients.

    exposed -> infectious --> isolated_rec -> recovered
                          |-> isolated_dec -> deceased
    """
    @property
    @memoize
    def prior(self):
        # prior of parameters
        return {
            "r_incub":   Uniform(torch.tensor(0.04, dtype=self.dtype), torch.tensor(1.0, dtype=self.dtype)),
            "inf_rate":  Uniform(torch.tensor(0.04, dtype=self.dtype), torch.tensor(1.0, dtype=self.dtype)),
            "surv_rate": Uniform(torch.tensor(0.5 , dtype=self.dtype), torch.tensor(1.0, dtype=self.dtype)),
            "r_iso":     Uniform(torch.tensor(0.04, dtype=self.dtype), torch.tensor(1.0, dtype=self.dtype)),
            "r_dec":     Uniform(torch.tensor(0.04, dtype=self.dtype), torch.tensor(1.0, dtype=self.dtype)),
            "r_rec":     Uniform(torch.tensor(0.04, dtype=self.dtype), torch.tensor(1.0, dtype=self.dtype)),
        }

    @property
    @memoize
    def display_fcn(self):
        return {
            "Incubation period": lambda p: 1./p["r_incub"],
            "Infection rate": lambda p: p["inf_rate"],
            "Survival rate": lambda p: p["surv_rate"],
            "Isolation period": lambda p: 1./p["r_iso"],
            "Deceased period": lambda p: 1./p["r_dec"],
            "Recovery period": lambda p: 1./p["r_rec"],
            "R0": lambda p: p["inf_rate"] / p["r_iso"],
        }

    def construct_jac(self, params):
        # jacobian from parameters
        r_incub, inf_rate, surv_rate, r_iso, r_dec, r_rec = self.unpack(params)
        nparams = len(self.prior)
        K_rate = torch.zeros(nparams, nparams).to(self.dtype)

        K_rate[0,0] = -r_incub
        K_rate[0,1] = inf_rate
        K_rate[1,0] = r_incub
        K_rate[1,1] = -r_iso
        K_rate[2,1] = (1-surv_rate) * r_iso
        K_rate[2,2] = -r_dec
        K_rate[3,1] = surv_rate * r_iso
        K_rate[3,3] = -r_rec
        K_rate[4,2] = r_dec
        K_rate[5,3] = r_rec
        return K_rate

    @property
    @memoize
    def vecstate(self):
        # state
        return {
            "exposed": 0,
            "infectious": 1,
            "isolated_dec": 2,
            "isolated_rec": 3,
            "dec": 4,
            "rec": 5,
        }

    @property
    @memoize
    def simdata_fcn(self):
        # get the processed states
        return {
            "confirmed_case": lambda vec: vec[self.vecstate["isolated_dec"]] + vec[self.vecstate["isolated_rec"]],
            "confirmed_death": lambda vec: vec[self.vecstate["dec"]],
            "confirmed_recovery": lambda vec: vec[self.vecstate["rec"]],
        }
