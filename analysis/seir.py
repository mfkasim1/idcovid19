import pickle
import argparse
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

death_now = 114 # 30 March 2020

def main():
    daynow = 66 # 29/03/2020
    delay = 7
    K = 1.484e-01
    frac = 0.0
    pop = 267.0e6
    R0 = 3.80
    R1 = 0.67 * (frac) + (1-frac) * R0
    Tinc = 5.2
    Tinf = 6.3
    cfr = 0.82/100
    tmax = 120

    # fit the data to the observation
    data = np.loadtxt("../data/data.csv", skiprows=1, usecols=list(range(1,8)), delimiter=",")[33:,:]
    xdays = data[:,0] - data[-1,0]
    deaths = data[:,-1]

    def getfcn(t, ndeaths):
        def loss(xdays, xdelay):
            xdays0 = xdays + xdelay
            death_interp = np.interp(xdays0, t, ndeaths)
            return death_interp
        return loss

    # simulate the system
    t, yt = simulate(daynow, delay, pop, Tinc, K, R0, R1, tmax)
    t2, yt2 = simulate(daynow, delay, 1e4, Tinc, K, R0, R1, tmax)

    # get the new deaths and accumulated
    newdeaths, cumdeaths = getdeaths(t, yt, cfr)
    newdeaths2, cumdeaths2 = getdeaths(t2, yt2, cfr)

    xdelay , _ = curve_fit(getfcn(t, cumdeaths), xdays, deaths)
    xdelay2, _ = curve_fit(getfcn(t2, cumdeaths2), xdays, deaths)

    plt.plot(t-xdelay, cumdeaths)
    plt.plot(t2-xdelay2, cumdeaths2)
    plt.plot(xdays, deaths, 'o')
    plt.ylim([-10, 1000])
    plt.ylabel("Total kematian")
    plt.xlabel("Hari sejak 30 Maret 2020")
    # plt.gca().set_yscale("log")
    plt.show()

    # durations.append(np.max(t[1:][ddeath>20]))
    deaths.append(ndeaths)

def main2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_const", default=False, const=True)
    args = parser.parse_args()

    # fixed parameters
    daynow = 66 # 29/03/2020
    delay = 7
    pop = 267.0e6
    tmax = 18*30
    nsamples = 100

    # uncertain parameters
    Kmean = 1.484e-01
    Kstd = 6.15e-2 # fitting
    R0min = 2.0
    R0max = 4.0
    Tinc_mean = 5.2
    Tinc_std = 0.5 # https://www.mdpi.com/2077-0383/9/2/538
    cfr_mean = 0.82e-2
    cfr_std = 0.09e-2 # https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf with Indonesian's demographic

    # cases
    R1_whn_max = .37
    R1_whn_min = .28 # https://www.medrxiv.org/content/10.1101/2020.03.03.20030593v1.full.pdf
    R1_it_max = 1.1
    R1_it_min = 0.9
    t = np.arange(tmax*10)*0.1

    def normal(mean, std):
        return np.random.random() * 2*std - std + mean

    def sample(case):
        K = normal(Kmean, Kstd)
        R0 = np.random.random() * (R0max - R0min) + R0min
        Tinc = normal(Tinc_mean, Tinc_std)
        cfr = normal(cfr_mean, cfr_std)
        if case == "wuhan":
            R1 = np.random.random() * (R1_whn_max - R1_whn_min) + R1_whn_min
        elif case == "italy":
            R1 = np.random.random() * (R1_it_max - R1_it_min) + R1_it_min
        elif case == "wuhan_half":
            R1min = np.random.random() * (R1_whn_max - R1_whn_min) + R1_whn_min
            def R1(delay):
                if delay < 0: return R0
                if (delay // 7) % 2 == 0: return R1min
                else: return R0
        elif case == "wuhan_half_month":
            R1min = np.random.random() * (R1_whn_max - R1_whn_min) + R1_whn_min
            def R1(delay):
                if delay < 0: return R0
                if (delay // 30) % 2 == 0: return R1min
                else: return R0
        elif case == "wuhan_75":
            R1min = np.random.random() * (R1_whn_max - R1_whn_min) + R1_whn_min
            def R1(delay):
                if delay < 0: return R0
                if (delay // 7) % 4 < 3: return R1min
                else: return R0
        elif case == "rtest":
            Tinf = np.random.randn() * 1 + 4 # need reference
            Tinf0 = (R0 - 1 - K*Tinc) / K / (1 + K*Tinc)
            R1 = R0 * Tinf / Tinf0
            if R1 > R0:
                R1 = R0
        elif case == "no":
            R1 = R0
        return K, R0, Tinc, cfr, R1

    # collect the samples
    cases = ["wuhan", "wuhan_half", "wuhan_75", "no"]
    # cases = ["wuhan", "wuhan_half", "wuhan_75", "wuhan_half_month", "rtest", "no"]
    if not args.show:
        all_cumdeaths = {}
        all_durations = {}
        all_yts = {}
        for case in cases:
            print(case)
            all_cumdeaths[case] = []
            all_durations[case] = []
            all_yts[case] = []
            for i in range(nsamples):
                K, R0, Tinc, cfr, R1 = sample(case)
                dly = delay if cases != "rtest" else 0
                t, yt = simulate(daynow, dly, pop, Tinc, K, R0, R1, tmax, cfr=cfr)
                if np.any(np.isnan(yt)) or np.any(np.isinf(yt)): continue
                newdeaths, cumdeaths = getdeaths(t, yt, cfr)
                dur = getduration(t, yt, newdeaths, 1)
                if dur is None: continue
                all_cumdeaths[case].append(cumdeaths[-1])
                all_durations[case].append(dur)
                all_yts[case].append(yt)

        with open("samples.pkl", "wb") as fb:
            obj = (all_cumdeaths, all_durations, all_yts)
            pickle.dump(obj, fb)

    else:
        with open("samples.pkl", "rb") as fb:
            obj = pickle.load(fb)
            all_cumdeaths, all_durations, all_yts = obj

    # plot the samples
    labels = {
        "wuhan": "Karantina wilayah (100% seperti di Wuhan)",
        "wuhan_half": "Karantina wilayah (50%: 1 minggu aktif, 1 minggu off)",
        "wuhan_half_month": "Karantina wilayah (50%: 1 bulan aktif, 1 bulan off)",
        "wuhan_75": "Karantina wilayah (75%: 1 minggu aktif, 3 minggu off)",
        "rtest": "Rapid test tanpa karantina wilayah",
        "no": "Tanpa intervensi"
    }
    for i,case in enumerate(cases):
        cumdeaths = np.asarray(all_cumdeaths[case])
        durations = np.asarray(all_durations[case])
        if case == "wuhan_75":
            print(cumdeaths, durations)
        plt.plot(durations[:1]/30, cumdeaths[:1], 'C%d.'%i, label=labels[case])
        plt.plot(durations/30, cumdeaths, 'C%d.'%i, alpha=0.2)
    plt.gca().set_yscale("log")
    plt.xlabel("Bulan dari sekarang")
    plt.ylabel("Jumlah kematian")
    plt.xlim([0, 18.])
    plt.legend()
    plt.show()

    daymonth = 30.
    for i,case in enumerate(cases):
        yts = np.asarray(all_yts[case]) # (nsamples, 4, nt)
        plt.plot(t/daymonth, yts[0,2,:], 'C%d'%i, label=labels[case])
        plt.plot(t/daymonth, yts[:,2,:].T, 'C%d'%i, alpha=0.2)
    plt.gca().set_yscale("log")
    plt.xlabel("Bulan dari sekarang")
    plt.ylabel("Jumlah kasus aktif")
    plt.xlim([0, 18.])
    plt.legend()
    plt.show()

def simulate(daynow, delay, pop, Tinc, K, R0, R1, tmax, cfr=None):
    y0 = (pop, 1.0, 0.0, 0.0)
    tinterv = daynow + delay
    Tinf = (R0 - 1 - K*Tinc) / K / (1 + K*Tinc)
    b = Tinc + Tinf
    a = Tinc * Tinf
    c = 1-R0
    d = (-b + np.sqrt(b*b-4*a*c))/(2*a)
    timenow = [daynow if cfr is None else 9e99, False]

    def func(y, t):
        # y: (S, E, I, R)
        if cfr is not None:
            if y[-1] * cfr >= death_now and timenow[1] == False:
                timenow[0] = t
                timenow[1] = True
        if hasattr(R1, "__call__"):
            R = R1(t-delay)
        else:
            R = R0 if t <= timenow[0] + delay else R1
        dsdt = -R/Tinf * y[2]*y[0]/pop
        dedt = R/Tinf * y[2]*y[0]/pop - y[1] / Tinc
        didt = y[1]/Tinc - y[2]/Tinf
        drdt = y[2]/Tinf
        dydt = np.array([dsdt, dedt, didt, drdt])
        return dydt

    t = np.arange(tmax*10)*0.1
    yt = odeint(func, y0, t)
    return t, yt # (nt, 4)

def getdeaths(t, yt, cfr):
    newdeaths = np.zeros(t.shape[0])
    newdeaths[1:] = (yt[1:,-1] - yt[:-1,-1]) * cfr / (t[1] - t[0])
    cumdeaths = np.cumsum(newdeaths) * (t[1] - t[0])
    return newdeaths, cumdeaths

def getduration(t, yt, nd, minval=100):
    # ei = yt[:,1] + yt[:,2]
    # idx = ei > minval
    idx = nd > minval
    tidx = t[idx]
    if len(tidx) == 0:
        return None
    period = np.max(tidx) - np.min(tidx)
    return period

if __name__ == "__main__":
    main2()
