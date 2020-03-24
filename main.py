import argparse
import pickle
import numpy as np
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.nuts import NUTS, HMC
from idcovid19.models.factory import get_model

def main(args, fdata="data/data.csv", day_offset=33, ndays=1000, prefix=""):
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

    # load the data
    data0 = np.loadtxt(fdata, skiprows=1, delimiter=",", usecols=list(range(1,8))).astype(np.float64)
    data = data0[day_offset:day_offset+ndays,:]

    # choose model
    modelname = args.model.lower()
    model = get_model(modelname, data)
    samples_fname = "%spyro_samples_%s%s.pkl" % (prefix, modelname, suffix)

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

    # load and display
    with open(samples_fname, "rb") as fb:
        samples = pickle.load(fb)
    print("Collected %d samples" % len(samples[list(samples.keys())[0]]))

    filter_keys = args.filters
    if filter_keys is not None:
        # filter the samples
        samples = model.filter_samples(samples, filter_keys)
        print("Filtered into %d samples" % len(samples[list(samples.keys())[0]]))

    # simobs: (nobs, nsamples)
    simobs = model.sample_observations(samples)

    # plot the observation
    model.plot_obs_inferece(simobs)
    model.plot_samples(samples)

if __name__ == "__main__":
    # parse args from cmd line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="id")
    parser.add_argument("--model", type=str, default="model1")
    parser.add_argument("--infer", action="store_const", default=False, const=True)
    parser.add_argument("--large", action="store_const", default=False, const=True)
    parser.add_argument("--nchains", type=int, default=1)
    parser.add_argument("--filters", type=str, nargs="*")
    args = parser.parse_args()

    if args.data == "id":
        main(args, fdata="data/data.csv", day_offset=33, ndays=1000)
    elif args.data == "cn":
        main(args, fdata="data/cndata.csv", day_offset=1, ndays=6, prefix="cn_")
