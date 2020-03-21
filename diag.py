import pickle
import matplotlib.pyplot as plt

with open("posterior_samples.pkl", "rb") as fb:
    samples = pickle.load(fb)

nrows = 1
ncols = 3
keys = ["confirmed_prob", "survival_rate", "r0"]
for i in range(len(keys)):
    key = keys[i]
    plt.subplot(nrows, ncols, i+1)
    plt.hist(samples[key].numpy())
    plt.title("P(%s)"%key)
    plt.xlabel(key)
plt.show()
