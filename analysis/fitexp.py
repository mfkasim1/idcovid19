import argparse
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_const", default=False, const=True)
args = parser.parse_args()

data = np.loadtxt("../data/data.csv", skiprows=1, usecols=list(range(1,8)), delimiter=",")[33:,:]
xdays = data[:,0] - np.mean(data[:,0])
deaths = data[:,-1]
print(xdays, deaths)
logdeaths = np.log(deaths)

slope, offset, rval, pval, stderr = linregress(xdays, logdeaths)
stderr = np.sqrt(np.sum((logdeaths-(slope*logdeaths+offset))**2) / (len(logdeaths)-2.)) / np.sqrt(np.sum((xdays - np.mean(xdays))**2))

if args.plot:
    plt.plot(xdays, np.exp(offset + slope*xdays), 'C0-')
    plt.plot(xdays, np.exp(offset + (slope+stderr)*xdays), 'C0--')
    plt.plot(xdays, np.exp(offset + (slope-stderr)*xdays), 'C0--')
    plt.plot(xdays, deaths, 'C0o')
    plt.gca().set_yscale("log")
    plt.show()

print("Slope: %.3e" % slope)
print("Doubling every: %.2f" % (np.log(2)/slope))
print("R-squared: %.3f" % (rval*rval))
print("Stderr: %.3e" % stderr)
