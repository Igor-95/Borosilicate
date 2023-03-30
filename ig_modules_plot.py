import matplotlib.pyplot as plt
import numpy as np
import statistics
import math
import pandas as pd
import ig_modules_data as igd


def double_barplot(x, height1, height2, label1, label2, xlabel, ylabel1, ylabel2, ylim1_low, ylim1_up,
                   ylim2_low, ylim2_up, title, show=True):
    """
    Plottet zwei Balkendiagramme, die Balken sind nach der x input Variable sortiert. Bietet sich an wenn
    unterschiedliche Daten der selben Kategorie, geplottet werden sollen z.B. Anzahl / Ausschuss einer Maschine

    :param x: Kategorien wonach die Balken geplotet werden sollen.
    :type x: list
    :param height1: Werte für Blaue Balken
    :type height1: list
    :param height2: Werte für orange Balken
    :type height2: list
    :param label1: legende für blaue Balken
    :type label1: str
    :param label2: legende für orange balken
    :type label2: str
    :param xlabel: Name der X-Achse
    :type xlabel: str
    :param ylabel1: Name der linken Y-Achse
    :type ylabel1: str
    :param ylabel2: Name der rechten Y-Achse
    :type ylabel2: str
    :param ylim1_low: unterer Y-Grenzwert für die blauen Balken
    :type ylim1_low: float
    :param ylim1_up: oberer Y-Grenzwert für die blauen Balken
    :type ylim1_up: float
    :param ylim2_low: unterer Y-Grenzwert für die orangen Balken
    :type ylim2_low: float
    :param ylim2_up: oberer Y-Grenzwert für die orangen Balken
    :type ylim2_up: float
    :param title: Grafik Titel
    :type title: str
    :param show: Trigger für plt.show()
    :type show: bool
    """
    x, height1, height2 = np.array(x), np.array(height1), np.array(height2)

    plt.rcParams.update({'font.size': 18})
    fig, axs = plt.subplots()
    width = 0.35
    plt.rcParams.update({'font.size': 18})

    axs.grid(b=True, which='major', color='#666666', linestyle='-')
    axs.bar(x=x - (width / 2), height=height1, width=width, color="royalblue", edgecolor="k", label=label1)
    axs.set_ylabel(ylabel1)
    axs.set_xlabel(xlabel)
    axs.ticklabel_format(useOffset=False)
    axs.set_ylim(ylim1_low, ylim1_up)
    plt.xticks(ticks=x, labels=x)

    axs2 = axs.twinx()
    axs2.bar(x=x + width / 2, height=height2, width=width, color="darkorange", edgecolor="k", label=label2)
    axs2.set_ylabel(ylabel2)
    axs.legend(loc="upper left")
    axs2.legend(loc="upper right")
    axs.tick_params(axis='y', colors='royalblue')
    axs2.tick_params(axis='y', colors='darkorange')
    axs.yaxis.label.set_color('royalblue')
    axs2.yaxis.label.set_color('darkorange')

    axs2.set_ylim(ylim2_low, ylim2_up)
    plt.title(title)
    if show is True:
        plt.show()


def normalize_function(x, mean, std):
    """
    y = Wurzel((1/(2*pi*std)²)) * exp(-(x-mean)² /(2*std²))

    :param x: x-coordinate
    :type x:
    :param mean: Mittelwert
    :type mean:
    :param std: Standardabweichung
    :type std:
    :return: y-coordinate
    :rtype:
    """
    y = (1/(2*math.pi * std**2)**(1/2)) * math.exp(-(x-mean)**2 / (2*std**2))
    return y


def dist_plot(x, bins=50, tol_low=None, tol_up=None, color="royalblue", xlabel=None, title=None, xlim_low=None,
              xlim_high=None, logscale=False, hist=True, dist=False, process=False, share_set=None, show=True):
    """
    Plottet die Verteilung der eingegbenen Daten. Normalverteilung / Prozessanalyse kann ebenfalls dargestellt werden.
    Wenn Histogramm ausgeschaltet -> keine Prozessanalyse möglich.

    :param x: Daten die geplotet werden soll
    :type x: list
    :param bins: Zählbereiche bei einem Histogramm, default: 50
    :type bins: int
    :param tol_low: untere Toleranz, default: None
    :type tol_low: float
    :param tol_up: obere Toleranz, default: None
    :type tol_up: float
    :param color: Farbe des Histogramms, default: "royalblue"
    :type color: str
    :param xlabel: Label für die X-Achse, default: None
    :type xlabel: str
    :param title: Titel für die Grafik
    :type title: str
    :param xlim_low: Kleinster anzuzeigender Wert, default: None
    :type xlim_low: float
    :param xlim_high: Größter anzuzeigender Wert, default: None
    :type xlim_high: float
    :param hist: soll Histogramm dargestellt werden? default: True
    :type hist: bool
    :param dist: soll Normalverteilung dargestellt werdeb? default: False
    :type dist: bool
    :param process: Soll Prozessfähigkeitsanalyse dargestellt werden? default: False
    :type process: bool
    :param share_set: Teilgruppe eingeben
    :type share_set: list
    :param show: Trigger für plt.show(), default: True
    :type show: bool
    """
    if len(x) < 2:
        pass

    elif hist is True:
        plt.rcParams.update({'font.size': 18})
        fig, axs = plt.subplots()
        axs.grid(b=True, which='major', color='#666666', linestyle='-')
        axs.hist(x, bins=bins, color=color, edgecolor="k")
        axs.set_xlabel(f"{xlabel}")
        if logscale is True:
            axs.set_yscale("log")
            axs.set_ylabel("log Anzahl")
        else:
            axs.set_ylabel("Anzahl")
        if tol_low is not None:
            axs.axvline(x=tol_low, ymin=0, ymax=len(x)*0.1, color="r", linestyle="dashed", label=tol_low)
        if tol_up is not None:
            axs.axvline(x=tol_up, ymin=0, ymax=len(x)*0.1, color="r", linestyle="dashed", label=tol_up)
        if xlim_low is not None and xlim_high is not None:
            axs.set_xlim(xlim_low, xlim_high)
        plt.title(f"{title}")

        if dist is True:
            df = pd.DataFrame()

            df["r"] = list(x)
            df["mean"] = x

            if df["mean"].dtype == "float" or df["mean"].dtype == "int":
                mean, std = np.mean(x), np.std(x)

                if len(x) < 10_000:
                    scale = 10_000
                else:
                    scale = len(x)

                xdata = np.linspace(start=min(x) * 0.85, stop=max(x) * 1.15, num=scale)
                ydata = []
                for a in xdata:
                    ydata.append(normalize_function(a, mean, std))

                plt.rcParams.update({'font.size': 18})
                axs1 = axs.twinx()
                axs1.plot(xdata, ydata, color="r")
                axs1.set_ylim(0, max(ydata) * 1.15)

                if process is False:
                    textstr = '\n'.join((
                        r'$\mu=%.2f$' % (mean,),
                        r'$\sigma=%.2f$' % (std,)))
                    plt.legend(loc="upper left")
                    axs.text(0.05, 0.95, textstr, transform=axs.transAxes, fontsize=14,
                             verticalalignment='top')
                else:
                    ppk, cpk = cpk_analysis(data=x, tol_low=tol_low, tol_up=tol_up, share_set=share_set)
                    if ppk[2] < 1:
                        rounding = 3
                    else:
                        rounding = 2
                    if cpk[0] is None:
                        textstr = '\n'.join((
                            r'$\mu=%.2f$' % (ppk[0],),
                            r'$\sigma=%.3f$' % (ppk[1],),
                            f"cpk:  {round(ppk[2], rounding)}",
                            f"cpl:  {round(ppk[3], rounding)}",
                            f"cpu:  {round(ppk[4], rounding)}",
                            f"cp:   {round(ppk[5], rounding)}",
                            f"Anzahl:   {ppk[6]}"))
                        axs.text(0.05, 0.95, textstr, transform=axs.transAxes, fontsize=14,
                                 verticalalignment='top')
                    else:
                        textstr = '\n'.join((
                            "Gesamtprozessfähigkeit",
                            r'$\mu=%.2f$' % (ppk[0],),
                            r'$\sigma=%.3f$' % (ppk[1],),
                            f"cpk:  {round(ppk[2], rounding)}",
                            f"cpl:  {round(ppk[3], rounding)}",
                            f"cpu:  {round(ppk[4], rounding)}",
                            f"cp:   {round(ppk[5], rounding)}",
                            f"Anzahl:   {ppk[6]}"))

                        textstr1 = '\n'.join((
                            "Teilprozessfähigkeit",
                            r'$\mu=%.2f$' % (cpk[0],),
                            r'$\sigma=%.3f$' % (cpk[1],),
                            f"cpk:  {round(cpk[2], rounding)}",
                            f"cpl:  {round(cpk[3], rounding)}",
                            f"cpu:  {round(cpk[4], rounding)}",
                            f"cp:   {round(cpk[5], rounding)}"))

                        ydata = []
                        for a in xdata:
                            ydata.append(normalize_function(a, cpk[0], cpk[1]))
                        axs1.set_ylim(0, max(ydata) * 1.15)
                        axs1.plot(xdata, ydata, color="dimgray", linestyle="dashed")
                        axs.text(0.5, 0.95, textstr, transform=axs.transAxes, fontsize=18,
                                 verticalalignment='top')
                        axs.text(0.75, 0.95, textstr1, transform=axs.transAxes, fontsize=18,
                                 verticalalignment='top')
                        axs.legend(loc="upper left")

            else:
                print("here3")

        if show is True:
            plt.show()

    elif hist is False:
        mean, std = np.mean(x), np.std(x),
        if len(x) < 10_000:
            scale = 10_000
        else:
            scale = len(x) * 10

        xdata = np.linspace(start=min(x) * 0.85, stop=max(x) * 1.15, num=scale)
        ydata = []
        for a in xdata:
            ydata.append(normalize_function(a, mean, std))

        plt.rcParams.update({'font.size': 18})
        fig, axs = plt.subplots()
        axs.grid(b=True, which='major', color='#666666', linestyle='-')
        axs.plot(xdata, ydata, color=color)
        if xlabel is not None:
            axs.set_xlabel(f"{xlabel}")
        if xlim_low is not None and xlim_high is not None:
            axs.set_xlim(xlim_low, xlim_high)
        if title is not None:
            plt.title(f"{title}")

        if show is True:
            plt.show()

    else:
        raise AttributeError("either hist or dist are missing")


def cpk_analysis(data, tol_low, tol_up, share_set=None, sig=3):
    """
    Berechnet den Cpk der Gesamtheit mit einer Liste / np.array oder pd.df[""] und gibt den [Mittelwert /
    Standardabw. /Cpk / Cpl / Cpu und Cp] zurück. Ist eine Teilgruppe angegeben, so werden auch die Prozessfähigkeiten
    (cpk etc.) auch innerhalb der Teilgruppen berechnet und als zweite Liste zurückgegeben

    :param data: Daten für die die Prozessfähigkeit berechnet werden soll
    :type data: list
    :param tol_low: untere Toleranz
    :type tol_low: float
    :param tol_up: obere Toleranz
    :type tol_up: float
    :param share_set: Teilgruppen, falls vorhanden
    :type share_set: list
    :param sig: (...)/(sig * Standardabweichung)
    :type sig: int
    :return: 2 Listen mit Gesamtprozessfähigkeit und Prozessfähigkeit innerhalb Teilgruppen wenn keine Teilgruppen
    vorhanden -> [None, None. etc...]
    :rtype: list
    """

    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    count = len(data)

    if tol_low is not None:
        cpl = (mean - tol_low)/(sig*std)
    else:
        cpl = None

    if tol_up is not None:
        cpu = (tol_up - mean)/(sig*std)
    else:
        cpu = None

    if cpl is not None and cpu is not None:
        cp = (tol_up - tol_low)/(6*std)
        if cpl <= cpu:
            cpk = cpl
        else:
            cpk = cpu

    elif cpl is None and cpu is not None:
        cpk = cpu
        cp = None
    elif cpu is None and cpl is not None:
        cpk = cpl
        cp = None
    else:
        raise ValueError("no valid tolerance given")

    Ppk = [mean, std, cpk, cpl, cpu, cp, count]
    Cpk = [None, None, None, None, None, None]

    if share_set is None:
        return Ppk, Cpk

    else:
        del Cpk
        means, stds, cpks, cpls, cpus, cps = [], [], [], [], [], []
        df = pd.DataFrame()
        if isinstance(share_set, int):
            share_set = igd.ShareSetGenerator(share_set, len(data))
        else:
            pass
        df["share_set"], df["data"] = list(share_set), data
        share_set_uniques = df["share_set"].drop_duplicates()

        for a in share_set_uniques:
            df1 = df[df["share_set"] == a]
            if len(df1) < 2:
                continue
            else:
                means.append(mean)
                stds.append(df1["data"].std()*len(df1))

        mean , std = statistics.mean(means), sum(stds)/len(df)

        if tol_low is not None:
            cpl = (mean - tol_low) / (sig * std)
        else:
            cpl = None
        if tol_up is not None:
            cpu = (tol_up - mean) / (sig * std)
        else:
            cpu = None

        if cpl is not None and cpu is not None:
            cp = (tol_up - tol_low) / (6 * std)
            if cpl <= cpu:
                cpk = cpl
            else:
                cpk = cpu

        elif cpl is None and cpu is not None:
            cpk = cpu
            cp = None
        elif cpu is None and cpl is not None:
            cpk = cpl
            cp = None

        Cpk = [mean , std, cpk, cpl, cpu , cp]

        return Ppk, Cpk


def line_plot(x, y, tol_low=None, tol_up=None, label=None ,color="royalblue", xlabel=None, ylabel=None, ylim=None,
              dot=False, line=True,title=None, regression=False,show=True):
    """
    Plottet y = f(x). X kann Liste oder None sein. Wenn x=None -> Indexliste wird durch numpy erzeugt und die Daten
    werden in Reihenfolge dargestellt.

    :param x: X-Wert beim plot. liste oder None
    :type x: list
    :param y: Y-Wert beim plot.
    :type y: list
    :param tol_low: Untere Toleranz
    :type tol_low: float
    :param tol_up: obere Toleranz
    :type tol_up: float
    :param label: legende für plot
    :type label: str
    :param color: Farbe
    :type color: str
    :param xlabel: Bezeichnung der X-Achse
    :type xlabel: str
    :param ylabel: Bezeichnung der X-Achse
    :type ylabel: str
    :param dot: Linie mit oder Ohne Markierung
    :type dot: bool
    :param title: Titel der Grafik
    :type title: str
    :param show: Flagge für plt.show()
    :type show: bool
    """
    if dot is False:
        dot = "-"
    elif dot is True and line is True:
        dot = "o-"
    else:
        dot = "."

    if x is None:
        x = np.linspace(1, len(y), len(y))
    else:
        x = x

    plt.rcParams.update({'font.size': 18})
    fig, axs = plt.subplots()
    axs.grid(b=True, which='major', color='#666666', linestyle='-')
    axs.plot(x, y, dot, color=color, label=label)
    if regression is True:
        m, b, r2 = linear_regression(x, y, fit_quality=True)
        xaxs, yaxs, = np.linspace(min(x), max(x), len(x)), []
        for a in xaxs:
            yaxs.append(m*a+b)
        axs.plot(xaxs, yaxs, color="k", label=f"Güte: R² = {r2}")
    if tol_up is not None:
        axs.axhline(y=tol_up, xmin=0, xmax=len(x), color="r", linestyle="dashed")
    if tol_low is not None:
        axs.axhline(y=tol_low, xmin=0, xmax=len(x), color="r", linestyle="dashed")
    if xlabel is not None:
        axs.set_xlabel(f"{xlabel}")
    if ylabel is not None:
        axs.set_ylabel(f"{ylabel}")
    if title is not None:
        plt.title(f"{title}")
    if ylim is not None:
        axs.set_ylim(ylim[0], ylim[-1])
    if show is True:
        plt.legend()
        plt.show()


def binomial_dist(n, k, p, show=True):
    """
    :param n: Anzahl an Teilen / Datensätze
    :type n: int
    :param k: Anzahl an Schlechtteilen
    :type k: int
    :param p: Fehlerrate
    :type p: float
    :param show: Flagge zum plotten der Funktion
    :type show: bool
    :return: Wahrscheinlichkeit bei Fehlerrate p, k Schlechteile in einer n Anzahl zu haben.
    :rtype: float
    """

    n_k = math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
    pb0 = n_k*(p**k)*((1-p)**(n-k))
    x, y = [], []

    for a in range(n+1):
        n_k = math.factorial(n) / (math.factorial(a) * math.factorial(n - a))
        pb = n_k*(p**a)*((1-p)**(n-a))
        x.append(a)
        y.append(pb*100)

    plt.rcParams.update({'font.size': 18})
    fig, axs = plt.subplots()
    axs.grid(b=True, which='major', color='#666666', linestyle='-')
    axs.bar(x, y, color="royalblue", edgecolor="k", label=f"k-Teile({k}) = {round(pb0*100, 2)} %")
    axs.set_xlabel("k")
    axs.set_ylabel("Wahrscheinlichkeit [%]")
    axs.set_title(f"Binomialverteilung n-Teile: {n}, Wahrscheinlichkeit: {p*100} %")
    plt.legend()
    if show is True:
        plt.show()

    return pb0


def linear_regression(x, y, fit_quality=False):
    meanx, meany = statistics.mean(x), statistics.mean(y)
    m_xy, m_xx,  i = [], [], 0
    while i < len(x):
        m_xy.append((x[i]-meanx)*(y[i]-meany))
        m_xx.append((x[i]-meanx)**2)
        i += 1
    m = sum(m_xy)/sum(m_xx)
    b = meany - m*meanx

    if fit_quality is False:
        return m, b
    else:
        y_reg, sqe, sqt, sqr, i = [], [], [], [], 0
        for a in x:
            y_reg.append(m*a +b)
        while i < len(x):
            sqe.append((y_reg[i] - meany)**2)
            sqt.append((y[i] - meany)**2)
            sqr.append((y[i] - y_reg[i])**2)
            i += 1

        r_bestimmt = sum(sqe)/sum(sqt)
        r_unbestimmt = 1- (sum(sqr)/sum(sqt))

        return m, b, round(r_bestimmt, 3)


def scatter_plot(x, y, tol_low=None, tol_up=None, label=None, color="royalblue", xlabel=None, ylabel=None, ylim=None,
                 regression=False, title=None, show=True):

    plt.rcParams.update({'font.size': 18})
    fig, axs = plt.subplots()
    axs.grid(b=True, which='major', color='#666666', linestyle='-')
    axs.scatter(x, y, color=color, label=label)
    if regression is True:
        m, b, r2 = linear_regression(x, y, fit_quality=True)
        yaxs = []
        for a in x:
            yaxs.append(m*a+b)
        axs.plot(x, yaxs, color="k", label=f"Güte: R² = {r2}")
    plt.legend()
    if tol_up is not None:
        axs.axhline(y=tol_up, xmin=0, xmax=len(x), color="r", linestyle="dashed")
    if tol_low is not None:
        axs.axhline(y=tol_low, xmin=0, xmax=len(x), color="r", linestyle="dashed")
    if xlabel is not None:
        axs.set_xlabel(f"{xlabel}")
    if ylabel is not None:
        axs.set_ylabel(f"{ylabel}")
    if title is not None:
        plt.title(f"{title}")
    if ylim is not None:
        axs.set_ylim(ylim[0], ylim[-1])
    if show is True:
        plt.show()


def descriptive(data, quantil=False):
    mean, median, mode, std, min, max = np.mean(data), np.median(data), statistics.mode(data),np.std(data), np.min(data), np.max(data)
    if quantil is False:
        return [mean, median, mode,std, min, max]
    else:
        if quantil is True:
            q1, q3 = np.quantile(data, 0.25), np.quantile(data, 0.75)
            return [mean, median, mode,std, min, max, q1, q3]
        else:
            q1, q3 = np.quantile(data, quantil), np.quantile(data, 1-quantil)
            return [mean, median, mode, std, min, max, q1, q3]


def bins(data):
    xmin = round(min(data), 5)
    xmax = round(max(data), 5)
    range = round(xmax - xmin, 5)
    print(range, xmax, xmin)
    a = round(statistics.median(data), 5)
    counts = 0
    i = 1
    n = 0
    while n < 100:
        a1 = int(a * i)
        if a1 == a * i:
            break
        elif counts == 0:
            i += 9
            counts += 1
            n += 1
        else:
            i = i * 10
            counts += 1
            n += 1

    bins = int((range)*10**counts)
    print(bins)
    return bins


if __name__ == "__main__":
    pass
