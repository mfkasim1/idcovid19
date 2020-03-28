import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

country = "id"
fdir = "../data/"
if country == "it":
    data = np.loadtxt(fdir+"itdata.csv", skiprows=1, delimiter=",", usecols=(1,2,3))[16:,:]
    day_since_ldown = data[:,0] - 16
elif country == "id":
    data = np.loadtxt(fdir+"data.csv", skiprows=1, delimiter=",", usecols=(1,2,4))[32:,:]
    day_since_ldown = data[:,0] - 31

new_case = data[:,1]
new_death = data[:,2]

def fcase(xdays, logyoffset, logygrad1, xchange, logygrad2):
    idxbefore = xdays < xchange
    idxafter = xdays >= xchange
    xbefore = xdays[idxbefore]
    xafter = xdays[idxafter]
    logybefore = logyoffset + logygrad1 * xbefore
    logyoffset2 = logyoffset + (logygrad1 - logygrad2) * xchange
    logyafter = logyoffset2 + logygrad2 * xafter
    y = xdays * 0
    y[idxbefore] = logybefore
    y[idxafter] = logyafter
    return y

p0case = (7.6, 0.19, 8.0, 0.01)
p0death = (4.7, 0.19, 8.0, 0.01)
pcase, pcasevar = curve_fit(fcase, day_since_ldown, np.log(new_case), p0=p0case)
pdeath, pdeathvar = curve_fit(fcase, day_since_ldown, np.log(new_death), p0=p0death)

pcasestd = np.sqrt(np.diag(pcasevar))
pdeathstd = np.sqrt(np.diag(pdeathvar))

dgcase = pcase[1] - pcase[3]
dgdeath = pdeath[1] - pdeath[3]
stddgcase = (pcasestd[1]**2 + pcasestd[3]**2)**.5
stddgdeath = (pdeathstd[1]**2 + pdeathstd[3]**2)**.5

print(pcase, pcasestd)
print(pdeath, pdeathstd)
print(dgcase / stddgcase)
print(dgdeath / stddgdeath)

country_name = {
    "it": "Italia",
    "id": "Indonesia"
}[country]
plt.title("Studi kasus: %s" % country_name)
plt.plot(day_since_ldown, new_case, 'C0o-', alpha=0.3, label="Kasus baru")
plt.plot(day_since_ldown, np.exp(fcase(day_since_ldown, *pcase)), 'C0-')
plt.plot(day_since_ldown, new_death, 'C1o-', alpha=0.3, label="Kematian / hari")
plt.plot(day_since_ldown, np.exp(fcase(day_since_ldown, *pdeath)), 'C1-')
plt.xlabel("Hari setelah social distancing")
plt.ylabel("Jumlah kasus")
plt.gca().set_yscale("log")
plt.legend()
plt.show()
