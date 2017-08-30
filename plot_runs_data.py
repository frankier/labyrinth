import pickle
import sys
import decimal

import matplotlib.pyplot as plt


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def format_decimal(x, prec=2):
    if x == 0:
        return '0'
    tup = x.as_tuple()
    digits = list(tup.digits[:prec + 1])
    sign = '-' if tup.sign else ''
    print(digits)
    dec = ('.' + ''.join(str(i) for i in digits[1:])) if len(digits) > 1 and digits[1] != 0 else ''
    exp = x.adjusted()
    return '{sign}{int}{dec}e{exp}'.format(sign=sign, int=digits[0], dec=dec, exp=exp)


def mfloat(f):
    #return "${}$".format(latex_float(f))
    return format_decimal(decimal.Decimal(f), 1)


def plot(ax, fn):
    ax.grid(True, which='both', axis='y', linestyle='dashed')
    (x, ys, positions) = pickle.load(open(fn, 'rb'))

    # Draw a plot
    labels = [mfloat(i) for i in x]
    print(labels, ys, positions)
    print(positions)
    ax.set_xticklabels(labels, rotation=-70)
    ax.boxplot(x=ys, labels=labels, positions=positions,
               notch=True, bootstrap=1000, sym="x")
    ax.set_yscale('log')
    ax.set_ylim(1)


def plot_4():
    fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)

    for i in range(4):
        reward = i // 2
        eps = i % 2
        plot(axarr[reward, eps], sys.argv[i + 1])

    fig.text(0.05, 0.70, 'No treasure reward', ha='center', va='center', rotation='vertical')
    fig.text(0.05, 0.25, 'Treasure reward', ha='center', va='center', rotation='vertical')
    fig.text(0.30, 0.92, '$\epsilon$ = 0.05', ha='center', va='center')
    fig.text(0.75, 0.92, '$\epsilon$ = 0.10', ha='center', va='center')

    fig.text(0.02, 0.5, 'Steps to completion', ha='center', va='center', rotation='vertical')
    fig.text(0.50, 0.02, 'Training iterations', ha='center', va='center')

    plt.show()
    if len(sys.argv) == 6:
        fig.savefig(sys.argv[5])


def plot_1():
    ax = plt.gca()
    plot(ax, sys.argv[1])

    ax.set_ylabel("Steps until completion")
    ax.set_xlabel("Training iterations")

    plt.show()
    if len(sys.argv) == 3:
        plt.savefig(sys.argv[2])


def main():
    if len(sys.argv) in (2, 3):
        plot_1()
    elif len(sys.argv) in (5, 6):
        plot_4()
    else:
        print("Either 1 or 4 arguments please!")


if __name__ == '__main__':
    main()
