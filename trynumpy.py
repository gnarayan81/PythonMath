from numpy import *
from pylab import *

import gfiltpak

# Time axis with 100 times oversampling
t = linspace(0, 1, 100)
# Unit sine
f = sin(2 * pi * t)
figure(1)
# Gridlines
grid(True)
clf()#plot(t, f)

N = 16
b = ones(N)*1/N
a = [1]
y, n = gfiltpak.gimpz(b, N = N)
clf()#figure(2)
grid(True)
stem(n, y)

# Filter response.
fig = figure(2)
mag_p = fig.add_subplot(2,1,1)
ang_p = fig.add_subplot(2,1,2)

gfiltpak.gfreqz(b, a, color = 'k', magfig = mag_p, angfig = ang_p)

hold(True)

from scipy import signal as si

b = si.kaiser(N, 3)
gfiltpak.gfreqz(b, a, color = 'm', magfig = mag_p, angfig = ang_p, label = "kaiser", threedb = True)

hold(True)
b = si.blackmanharris(N)
gfiltpak.gfreqz(b, a, color = 'g', magfig = mag_p, angfig = ang_p, ylimmag = [-500, 0])

hold(True)
b = si.blackman(N)
gfiltpak.gfreqz(b, a, color = 'y', magfig = mag_p, angfig = ang_p)

# Show
mag_p.legend(('r', 'k', '3dB', 'b-h', 'b'), numpoints = 4, fancybox = True)
ang_p.legend(('r', 'k', 'b-h', 'b'), numpoints = 2, fancybox = True)

##############################################################
## IIR filter
c = 1
b = [1]
root_r = 0.10
a = [1, -root_r] #[1, 1, 0.25]

rep = 1

fig = figure('Repeated Poles : root = ' + str(root_r) + ' repeat = ' + str(rep))
#mag_p = fig.add_subplot(2,1,1)
#ang_p = fig.add_subplot(2,1,2)

#gfiltpak.gfreqz(b, a, color = 'm', magfig = mag_p, angfig = ang_p, ylimmag = [-40, 0], threedb = True, logx = True, Fs = 1e6)
Time = 64
y, n = gfiltpak.gimpz(b, a, Time)

subplot(221)
grid(True)
title('Single pole impulse response')
stem(n, y) 
poly_env = (array(n) + 1)
for i in range(2, rep+1, 1):
	poly_env *= (array(n) + i) #** (rep - 1)

import scipy.misc as mi
poly_env /= mi.factorial(rep)
subplot(222)
grid(True)
title('Multiplication envelope')
stem(n, poly_env)
subplot(223)
title('Repeated pole impulse response (simulated envelope)')
stem(n, y * poly_env)
grid(True)

# Get the polynomial coefficients from roots.
polyn = poly(ones(rep + 1) * root_r)
subplot(224)
y, n = gfiltpak.gimpz(b, polyn, Time)
stem(n, y)
title('Repeated pole impulse response (actual)')
grid(True)

## Repeated pole frequency analysis
rptpole = figure("Frequency Response")
fft_N = 32768
mag_p = rptpole.add_subplot(2,1,1)
ang_p = rptpole.add_subplot(2,1,2)

rep_array = arange(6) + 0
color_list = ['r', 'k', 'b', 'm', 'g', 'y']

for i in rep_array:
	ply = poly(ones(i + 1) * root_r)
	gfiltpak.gfreqz(b, ply, fft_N, Fs=1e6, color = color_list[i], magfig = mag_p, angfig = ang_p, logx = True, threedb = True, ylimmag = [-10, 0])
	hold(True)
##########################################

# Residues = Partial fraction expansion
z, p, k = si.tf2zpk(b, a)

#figure("Scatter plot")
#scatter(real(p), imag(p), vmin = -0.5, vmax = 0.5)
#scatter(real(z), imag(z), vmin = -0.5, vmax = 0.5)

#figure("Zplane plot")
#gfiltpak.zplane(b, a)

##########################################
# Residues
b = [0.5, 0.5, 0.25]
a = [1, 0.5, 0.5, 0.75]

ylims = [-50, 0]

multipole = figure("Multipole")
mag_p = multipole.add_subplot(2,1,1)
ang_p = multipole.add_subplot(2,1,2)
w, Ht = gfiltpak.gfreqz(b, a, N = 2048, magfig = mag_p, angfig = ang_p, logx = True, threedb = True, ylimmag = ylims)
r, p, k = si.residuez(b, a)
multipole.hold(True)

# filter 1
b1 = r[0]
a1 = [1, -p[0]]
w, H1 = gfiltpak.gfreqz(b1, a1, N = 2048, color = 'r', magfig = mag_p, angfig = ang_p, logx = True, threedb = True, ylimmag = ylims)

# filter 2
b1 = r[1]
a1 = [1, -p[1]]
w, H2 = gfiltpak.gfreqz(b1, a1, N = 2048, color = 'm', magfig = mag_p, angfig = ang_p, logx = True, threedb = True, ylimmag = ylims)

# filter 2
b1 = r[2]
a1 = [1, -p[2]]
w, H3 = gfiltpak.gfreqz(b1, a1, N = 2048, color = 'g', magfig = mag_p, angfig = ang_p, logx = True, threedb = True, ylimmag = ylims)

H = H1+H2+H3

HdB = 20*log10(abs(H))
HdB -= max(HdB)
Hang = angle(H)
mag_p.semilogx(w, HdB, 'y')
mag_p.set_ylim(ylims)
#ang_p.plot(w, Hang, 'k*')
show()

