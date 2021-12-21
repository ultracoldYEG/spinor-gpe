# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:23:45 2020

@author: benjamin
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
#from scipy.stats import median_abs_deviation as mad
from scipy.optimize import curve_fit

SAVE = True
filepath = 'G:\\My Drive\\Research\\GPU paper\\Paper Figures\\'

comp1 = 'TitanV'
comp2 = '980Ti'
comp3 = 'MX150'
N = 2**(np.linspace(12, 24, 13))


def error(dy, y):
    """Make error bars to show up properly on a log plot scale."""
    return 0.434*dy/y


def load(comp, device):
    """Load data from file."""
    filename = comp + '_' + device + '.npz'
    with np.load(filename) as data:
        size = np.trim_zeros(data['size'])
        med = np.trim_zeros(data['med'])
        mad = np.trim_zeros(data['mad'])
#    print(filename, ': ', len(size), len(med), len(mad))
    return size, med, mad


def power(x, a, b):
    """Power model - used in fitting."""
    return a*x**b


def line(x, a, b):
    """Linear model - used in fitting."""
    return a*x + b

# System 1: TitanV + i9
N1G, med1G, mad1G = load('TitanV', 'cuda_step')
N1C, med1C, mad1C = load('TitanV', 'cpu_step')

# System 2: 980 Ti + AMD
N2G, med2G, mad2G = load('980 Ti', 'cuda_step')
N2C, med2C, mad2C = load('980 Ti', 'cpu_step')

# System 3: MX150 + i5
N3G, med3G, mad3G = load('Acer Aspire', 'cuda_step')
N3C, med3C, mad3C = load('Acer Aspire', 'cpu_step')


# %%

# This section generates the benchmark plot - Figure 2a

Nx = np.logspace(12, 24, 10000, base=2)

p_1C, c_1C = curve_fit(power, 2**N1C, med1C, sigma=mad1C)
p_2C, c_2C = curve_fit(power, 2**N2C, med2C, sigma=mad2C)
p_3C, c_3C = curve_fit(power, 2**N3C, med3C, sigma=mad3C)

p_1G, c_1G = curve_fit(line, 2**(N1G), med1G, sigma=mad1G)
p_2G, c_2G = curve_fit(line, 2**(N2G), med2G, sigma=mad2G)
p_3G, c_3G = curve_fit(line, 2**(N3G), med3G, sigma=mad3G)

# Average the fitted parameters and print the results
print('GPU Linear slope: ' , np.mean([p_1G[0], p_2G[0], p_3G[0]]), ' +/- ',
      np.sqrt(np.diag(c_1G)[0] + np.diag(c_2G)[0] + np.diag(c_3G)[0]))
print('CPU Power: ' , np.mean([p_1C[1], p_2C[1], p_3C[1]]), ' +/- ',
      np.sqrt(np.diag(c_1C)[1] + np.diag(c_2C)[1] + np.diag(c_3C)[1]))
print('CPU amplitude: ' , np.mean([p_1C[0], p_2C[0], p_3C[0]]), ' +/- ',
      np.sqrt(np.diag(c_1C)[0] + np.diag(c_2C)[0] + np.diag(c_3C)[0]))



def power_alt(nu, a, k):
    return a * 2**(nu*k)

p_1C_a, c_1C_a = curve_fit(power_alt, N1C, med1C, sigma=mad1C)

# %%

fig = plt.figure(figsize=(3.277, 3.5), facecolor=None)
ax = plt.axes(xscale='linear', yscale='log')

c1 = '#721121'
c2 = '#0A8754'
c3 = '#058ED9'

lw = 2
ms = 7
cs = 3

# GPU Marks and Lines
mark1G, cap1G, bar1G, = ax.errorbar(N1G, med1G, yerr=error(mad1G, med1G),
                                    fmt='o', ms=ms, color=c1, zorder=10,
                                    capsize=cs, elinewidth=1)
mark2G, cap2G, bar2G, = ax.errorbar(N2G, med2G, yerr=error(mad2G, med2G),
                                    fmt='s', ms=ms, color=c2, zorder=10,
                                    capsize=cs, elinewidth=1)
mark3G, cap3G, bar3G, = ax.errorbar(N3G, med3G, yerr=error(mad3G, med3G),
                                    fmt='^', ms=ms, color=c3, zorder=10,
                                    capsize=cs, elinewidth=1)

idx = 9500
line1G, = ax.plot(np.log2(Nx[:idx]), line(Nx[:idx], *p_1G), '-', color=c1,
                  lw=lw)

idx = 9000
line2G, = ax.plot(np.log2(Nx[:idx]), line(Nx[:idx], *p_2G), '-', color=c2,
                  lw=lw)

idx = 7200
line3G, = ax.plot(np.log2(Nx[:idx]), line(Nx[:idx], *p_3G), '-', color=c3,
                  lw=lw)

# CPU Marks and Lines
mark1C, cap1C, bar1C, = ax.errorbar(N1C, med1C, yerr=error(mad1C, med1C),
                                    fmt='o', ms=ms, color=c1,
                                    fillstyle="full", markerfacecolor='w',
                                    mew=2, zorder=7, capsize=cs, elinewidth=1)
mark2C, cap2C, bar21C, = ax.errorbar(N2C[:-1], med2C[:-1],
                                     yerr=error(mad2C[:-1], med2C[:-1]),
                                     fmt='s', ms=ms, color=c2,
                                     fillstyle="full", markerfacecolor='w',
                                     mew=2, zorder=7, capsize=cs, elinewidth=1)
mark3C, cap3C, bar3C, = ax.errorbar(N3C, med3C, yerr=error(mad3C, med3C),
                                    fmt='^', ms=ms, color=c3,
                                    fillstyle="full", markerfacecolor='w',
                                    mew=2, zorder=7, capsize=cs, elinewidth=1)


line1C, = ax.plot(np.log2(Nx), power(Nx, *p_1C), '-', color=c1, lw=lw,
                  zorder=3.5, alpha=0.75)

idx = 9000
line2C, = ax.plot(np.log2(Nx[:idx]), power(Nx[:idx], *p_2C), '-', color=c2,
                  lw=lw, zorder=3.5, alpha=0.75)

line3C, = ax.plot(np.log2(Nx), power(Nx, *p_3C), '-', color=c3,
                  lw=lw, zorder=3.5, alpha=0.75)

# Combinging legend artists
labels = np.array(['TitanV', '980 Ti', 'MX150', 'i9', 'FX', 'i5'])

space = Line2D([], [], linestyle='')
mark_space, cap_space, bar_space, = ax.errorbar([], [], yerr=[], c='w')

leg = ax.legend([(line1G, mark1G), (line2G, mark2G), (line3G, mark3G),
                 (line1C, mark1C), (line2C, mark2C), (line3C, mark3C)],
                labels, fontsize=8, loc=2)
t1, t2, t3, t4, t5, t6 = leg.get_texts()
t1._fontproperties = t2._fontproperties.copy()
t6._fontproperties = t2._fontproperties.copy()
t5._fontproperties = t2._fontproperties.copy()
t1.set_size(9)
t6.set_size(9)
t5.set_size(9)


ax.set_xlabel('\nGrid size', fontsize=12)
ax.set_ylabel('Time per iteration [s]', fontsize=12)
ax.set_xlim(None, 25)
ax.set_ylim(1e-3, 150)

ax.set_xticks([2*n for n in range(6, 13)])
ax.tick_params(axis='both', width=0.5, length=2, direction='in',
               top='true', right='true', labelsize=10)
ax.tick_params(axis='x', which='minor', direction='in',
               bottom=False)
ax.tick_params(axis='y', which='minor', direction='in', right='true')

# Annotations of the grid size
textheight = 2.0e-4
plt.text(12-0.8, textheight, '($64^2$)', fontsize=9)
plt.text(16-1, textheight, '($256^2$)', fontsize=9)
plt.text(20-1.2, textheight, '($1024^2$)', fontsize=9)
plt.text(24-1, textheight, '($4096^2$)', fontsize=9)
if SAVE:
    plt.savefig(filepath + 'Fig2_Benchmarks\\' + 'bench_prop_times.pdf',
                bbox_inches='tight')
plt.show()

#%%
#This section generates the speedup plot - Figure 2b
lw=2

Sp1 = med1C[:-1] / med1G
Sp2 = med2C[:-1] / med2G
Sp3 = med3C[:-4] / med3G

devSp1 = Sp1 * np.sqrt((mad1G/med1G)**2 + (mad1C[:-1]/med1C[:-1])**2)
devSp1_plt = error(devSp1, Sp1)

devSp2 = Sp2 * np.sqrt((mad2G/med2G)**2 + (mad2C[:-1]/med2C[:-1])**2)
devSp2_plt = error(devSp2, Sp2)

devSp3 = Sp3 * np.sqrt((mad3G/med3G)**2 + (mad3C[:-4]/med3C[:-4])**2)
devSp3_plt = error(devSp3, Sp3)

fig = plt.figure(figsize=(3.277, 3.5), facecolor=None)
ax = plt.axes(xscale='log', yscale='log')
ax.errorbar(N[:len(Sp1)], Sp1, yerr=devSp1_plt, fmt='-', ms=8, zorder=9,
              color=c1, lw=0, marker='o', label='i9 / Titan V',
              elinewidth=3, capsize=cs)
ax.errorbar(N[:len(Sp2)], Sp2, yerr=devSp2_plt, fmt='-', ms=8, zorder=8,
              color=c2, lw=0, marker='s', label='FX / 980 Ti',
              elinewidth=3, capsize=cs)
ax.errorbar(N[:len(Sp3)], Sp3, yerr=devSp3_plt, fmt='-', ms=8, zorder=7,
              color=c3, lw=0, marker='^', label='i5 / MX150',
              elinewidth=3, capsize=cs)

idx = 9750
ax.plot(Nx[:idx], power(Nx[:idx], *p_1C) / line(Nx[:idx], *p_1G), '-',
        color=c1, lw=lw, zorder=6, alpha=0.75)

idx = 9000
ax.plot(Nx[:idx], power(Nx[:idx], *p_2C) / line(Nx[:idx], *p_2G), '-',
        color=c2, lw=lw, zorder=6, alpha=0.75)

idx = 7200
ax.plot(Nx[:idx], power(Nx[:idx], *p_3C) / line(Nx[:idx], *p_3G), '-',
        color=c3, lw=lw, zorder=6, alpha=0.75)

leg = ax.legend(fontsize=9, loc=2)
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)

ax.hlines(1, xmin=2**12, xmax=2**24, ls=(0, (2, 1)), color='k', lw=2, zorder=5)

ax.set_xlabel('\nGrid size', fontsize=12)
ax.set_ylabel('Speedup', fontsize=12)
ax.yaxis.set_label_position("right")
ax.set_xticks(N[::2])
ax.set_xticklabels([str(int(np.log2(n))) for n in N[::2]], fontsize=12)
ax.tick_params(axis='y', labelsize=10)
#ax.tick_params(axis='both', width=0.5, length=2, direction='in', top='true',
#               right='true')
#ax.set_xticks([2*n for n in range(6, 13)])
#ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='both', width=0.5, length=2, direction='in',
               top='true', right='true', labelsize=10)
ax.tick_params(axis='x', which='minor', direction='in', bottom=False)
ax.tick_params(axis='y', which='minor', direction='in', right='true')
ax.yaxis.tick_right()

ax.set_ylim(8e-1, 300)

# Annotations of the grid size
textheight = 3.3e-1
plt.text(1.925**12, textheight, '($64^2$)', fontsize=9)
plt.text(1.925**16, textheight, '($256^2$)', fontsize=9)
plt.text(1.93**20, textheight, '($1024^2$)', fontsize=9)
plt.text(1.94**24, textheight, '($4096^2$)', fontsize=9)

if SAVE:
    plt.savefig(filepath + 'Fig2_Benchmarks\\' + 'bench_prop_speedup.pdf',
                bbox_inches='tight')
plt.show()
print(f'Max. speedup - Computer 1: {np.max(Sp1):.1f}')
print(f'Max. speedup - Computer 2: {np.max(Sp2):.1f}')
print(f'Max. speedup - Computer 3: {np.max(Sp3):.1f}')

#%%
#This section generates the image for the FFT/Hadamard benchmarks

# Load in data
NGfft, medGfft, madGfft = load('TitanV', 'cuda_fft')
NGifft, medGifft, madGifft = load('TitanV', 'cuda_ifft')
NGhad, medGhad, madGhad = load('TitanV', 'cuda_had3')

NCfft, medCfft, madCfft = load('TitanV', 'cpu_fft')
NCifft, medCifft, madCifft = load('TitanV', 'cpu_ifft')
NChad, medChad, madChad = load('TitanV', 'cpu_had3')

# Fit CPU function times to power law.
p_Cfft, c_Cfft = curve_fit(power, 2**(NCfft), medCfft, sigma=madCfft)
p_Cifft, c_Cifft = curve_fit(power, 2**(NCifft), medCifft, sigma=madCifft)
p_Chad, c_Chad = curve_fit(power, 2**(NChad), medChad, sigma=madChad)

p_Gfft, c_Gfft = curve_fit(line, 2**(NGfft), medGfft, sigma=madGfft)
p_Gifft, c_Gifft = curve_fit(line, 2**(NGifft), medGifft, sigma=madGifft)
p_Ghad, c_Ghad = curve_fit(line, 2**(NGhad), medGhad, sigma=madGhad)


# Average the fitted parameters and print the results
print('CPU Power: ' , np.mean([p_Cfft[1], p_Cifft[1], p_Chad[1]]), ' +/- ',
      np.sqrt(np.diag(p_Cfft)[0] + np.diag(p_Cifft)[0] + np.diag(p_Chad)[0]))
print('CPU amplitude: ' , np.mean([p_Cfft[0], p_Cifft[0], p_Chad[0]]), ' +/- ',
      np.sqrt(np.diag(p_Cfft)[0] + np.diag(p_Cifft)[0] + np.diag(p_Chad)[0]))

C0 = '#32533D'#79745C'
C1 = '#FE4A49'#709176'
C2 = '#3F88C5'#f9d55b'

width = 0.3
fig = plt.subplots(figsize=(3.277, 3.5))
ax = plt.axes(xscale='log', yscale='log')
lw = 2
ms = 7
cs=3

# TitanV GPU function times
markGfft, capGfft, barGfft, = ax.errorbar(2**NGfft, medGfft, yerr=madGfft,
                                          fmt='s', ms=ms, color=C0,
                                          capsize=cs, label='$\\mathcal{F}$',
                                          alpha=1.0, zorder=5)
markGifft, capGifft, barGifft, = ax.errorbar(2**NGifft, medGifft, yerr=madGifft,
                                             fmt='D', ms=ms, color=C1,
                                             capsize=cs,
                                             label='$\\mathcal{F}^{-1}$',
                                             alpha=1.0, zorder=5)
markGhad, capGhad, barGhad, = ax.errorbar(2**NGhad, medGhad, yerr=madGhad,
                                          fmt='o', ms=ms, color=C2,
                                          capsize=cs, label='$A \\circ B$',
                                          alpha=1.0, zorder=5)

# i9 CPU function times
markCfft, capCfft, barCfft, = ax.errorbar(2**NCfft, medCfft, yerr=madCfft,
                                          fmt='s', ms=ms, color=C0,
                                          fillstyle="full",
                                          markerfacecolor='w', mew=2, 
                                          capsize=cs, label='$\\mathcal{F}$',
                                          alpha=1.0, zorder=2)
markCifft, capCifft, barCifft, = ax.errorbar(2**NCifft, medCifft,
                                             yerr=madCifft, fmt='D', ms=ms,
                                             color=C1, fillstyle="full",
                                             markerfacecolor='w', mew=2, 
                                             capsize=cs,
                                             label='$\\mathcal{F}^{-1}$',
                                             alpha=1.0, zorder=2)
markChad, capChad, barChad, = ax.errorbar(2**NChad, medChad, yerr=madChad,
                                          fmt='o', ms=ms, color=C2,
                                          fillstyle="full",
                                          markerfacecolor='w', mew=2, 
                                          capsize=cs, label='$A \\circ B$',
                                          alpha=1.0, zorder=2)

lineCfft, = ax.plot(Nx, power(Nx, *p_Cfft), '-', lw=lw, color=C0)
lineCifft, = ax.plot(Nx, power(Nx, *p_Cifft), '-', lw=lw, color=C1)
lineChad, = ax.plot(Nx, power(Nx, *p_Chad), '-', lw=lw, color=C2)

lineGfft, = ax.plot(Nx, line(Nx, *p_Gfft), '-', lw=0, color=C0)
lineGifft, = ax.plot(Nx, line(Nx, *p_Gifft), '-', lw=0, color=C1)
lineGhad, = ax.plot(Nx, line(Nx, *p_Ghad), '-', lw=0, color=C2)


# Combinging legend artists
labels = np.array(['$\\mathcal{F}$', '$\\mathcal{F}^{-1}$', '$A \\circ B$',
                   '$\\mathcal{F}$', '$\\mathcal{F}^{-1}$', '$A \\circ B$'])

#space = Line2D([], [], linestyle='')
#mark_space, cap_space, bar_space, = ax.errorbar([], [], yerr=[], c='w')

leg = ax.legend([(lineCfft, markCfft), (lineCifft, markCifft), (lineChad, markChad),
                 (lineGfft, markGfft), (lineGifft, markGifft), (lineGhad, markGhad)],
                labels, fontsize=8, loc=2)
t1, t2, t3, t4, t5, t6 = leg.get_texts()
t1._fontproperties = t2._fontproperties.copy()
t6._fontproperties = t2._fontproperties.copy()
t5._fontproperties = t2._fontproperties.copy()
t1.set_size(9)
t6.set_size(9)
t5.set_size(9)


ax.set_ylabel('Evaluation time [s]', fontsize=12)
ax.set_xlabel('\nGrid size', fontsize=12)

ax.set_xticks(N[::2])
ax.set_xticklabels([str(int(np.log2(n))) for n in N[::2]])
ax.tick_params(axis='both', width=0.75, length=2, direction='in', top='true',
               right='true', labelsize=10)
ax.tick_params(axis='x', which='minor', bottom=False, direction='in',
               right='true')
ax.tick_params(axis='y', which='minor', direction='in', right='true')

# Annotate grid sizes
txtheight =1.5e-6
txtsize = 9

plt.text(1.92**12, txtheight, '($64^2$)', fontsize=txtsize)
plt.text(1.91**16, txtheight, '($256^2$)', fontsize=txtsize)
plt.text(1.92**20, txtheight, '($1024^2$)', fontsize=txtsize)
plt.text(1.93**24, txtheight, '($4096^2$)', fontsize=txtsize)
#plt.grid()
plt.legend(fontsize=9)
if SAVE:
    plt.savefig(filepath + 'Fig3_FFTHadamard\\'  + 'bench_ffthad_times.pdf',
                bbox_inches='tight')
plt.show()


#%%
#This section generates the speedup plots for the different functions calls

# Calculates speedup
Spfft = medCfft[:-1] / medGfft
Spifft = medCifft[:-1] / medGifft
Sphad = medChad / medGhad

# Calculates speedup uncertainties
devSpfft = Spfft * np.sqrt((madGfft/medGfft)**2
                           + (madCfft[:-1]/medCfft[:-1])**2)
devSpifft = Spifft * np.sqrt((madGifft/medGifft)**2
                             + (madCifft[:-1]/medCifft[:-1])**2)
devSphad = Sphad * np.sqrt((madGhad/medGhad)**2 + (madChad/medChad)**2)

lw=2

fig = plt.figure(figsize=(3.277, 3.5), facecolor=None)
ax = plt.axes(xscale='log', yscale='log')

ax.errorbar(N[:len(Spfft)], Spfft, yerr=error(devSpfft, Spfft), fmt='-',
            ms=ms, zorder=9, color=C0, lw=lw, marker='s',
            label='$\\mathcal{F}$', elinewidth=3, capsize=cs)
ax.errorbar(N[:len(Spifft)], Spifft, yerr=error(devSpifft, Spifft), fmt='-',
            ms=ms, zorder=8, color=C1, lw=lw, marker='D',
            label='$\\mathcal{F}^{-1}$', elinewidth=3, capsize=cs)
ax.errorbar(2**NGhad, Sphad, yerr=error(devSphad, Sphad), fmt='-', ms=8,
            zorder=ms, color=C2, marker='o', lw=lw,
            label='$A \\circ B$', elinewidth=3, capsize=cs)

ax.hlines(1, xmin=2**12, xmax=2**23, ls=(0, (2, 1)), color='k', lw=2, zorder=1)


leg = ax.legend(fontsize=9, loc=2)
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)
    

ax.set_xlabel('\nGrid size', fontsize=12)
ax.set_ylabel('Speedup', fontsize=12)
ax.yaxis.set_label_position("right")

ax.set_xticks(N[::2])
ax.set_xticklabels([str(int(np.log2(n))) for n in N[::2]], fontsize=12)
ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='both', width=0.5, length = 2, direction='in', top='true',
               right='true', labelsize=10)
ax.tick_params(axis='x', which='minor', bottom=False, direction='in',
               right='true')
ax.yaxis.tick_right()
ax.tick_params(axis='y', which='minor', direction='in', right='true',
               left='true')


# Annotate grid sizes
txtheight = 2e-1

plt.text(1.925**12, txtheight, '($64^2$)', fontsize=9)
plt.text(1.925**16, txtheight, '($256^2$)', fontsize=9)
plt.text(1.93**20, txtheight, '($1024^2$)', fontsize=9)
plt.text(1.94**24, txtheight, '($4096^2$)', fontsize=9)

if SAVE:
    plt.savefig(filepath + 'Fig3_FFTHadamard\\' + 'bench_ffthad_speedup.pdf',
                bbox_inches='tight')
plt.show()
print(f'Max. speedup - FFT: {np.max(Spfft):.1f}')
print(f'Max. speedup - iFFT: {np.max(Spifft):.1f}')
print(f'Max. speedup - Hadamard: {np.max(Sphad):.1f}')
