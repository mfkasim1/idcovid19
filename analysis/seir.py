import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

deaths = []
delays = np.arange(2,30)*1.0 # [2, 4, 7, 10, 14, 21, 30]
fracs = np.linspace(0, 1.0, 200)
use_delay = True
# for frac in fracs:
for delay in delays:
    if not use_delay:
        delay = 7
    else:
        frac = 0.0
    daynow = 66 # 29/03/2020

    pop = 263614931.0
    y0 = (pop, 1.0, 0.0, 0.0)
    R0 = 3.80
    R1 = 0.33 * (1-frac) + (frac) * R0
    tinterv = daynow + delay
    print(tinterv)
    Tinc = 5.2
    Tinf = 6.3
    cfr = 0.82/100

    b = Tinc + Tinf
    a = Tinc * Tinf
    c = 1-R0
    d = (-b + np.sqrt(b*b-4*a*c))/(2*a)
    print("Exponential factor: %.3e" % d)
    print("Doubling day: %.3e" % (np.log(2)/d))

    def func(y, t):
        # y: (S, E, I, R)
        R = R0 if t <= tinterv else R1
        dsdt = -R/Tinf * y[2]*y[0]/pop
        dedt = R/Tinf * y[2]*y[0]/pop - y[1] / Tinc
        didt = y[1]/Tinc - y[2]/Tinf
        drdt = y[2]/Tinf
        dydt = np.array([dsdt, dedt, didt, drdt])
        return dydt

    t = np.arange(20000)*0.1
    yt = odeint(func, y0, t)

    deaths.append(yt[-1,3]*cfr)

if use_delay:
    plt.plot(delays, deaths, '-')
    plt.xlabel("Total hari sejak 29 Maret 2020")
    plt.ylabel("Total kematian")
    plt.title("Total kematian vs waktu penundaan karantina wilayah")
    # plt.gca().set_yscale("log")
    plt.show()
else:
    plt.plot(fracs, deaths, '-')
    plt.xlabel("Porsi ketidakpatuhan")
    plt.ylabel("Total kematian")
    plt.gca().set_yscale("log")
    plt.show()

# delaying lockdown 4 days double the number of deaths
