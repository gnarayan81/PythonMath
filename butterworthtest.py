import numpy as np
import scipy.signal as si
import pylab as pyl
import gfiltpak as gfp

## 3 dB point
fc = 1000
## sampling 
fs = 8192
# order
order = 5

# Butterworth design
z,p,k = si.butter(order, 2*fc/fs, output='zpk') # [0, pi] -> [0, 1]

sos, g = gfp.gzp2sos(z, p, k)

## Sections numerators and denominators.
Bs = sos.transpose()[0:3].transpose()
As = sos.transpose()[3:6].transpose()

nsec, unu = sos.shape

## Samples
nsamps = 256
## dirac impulse scaled by gain 'g'.
x = g * np.concatenate((np.array([1]), np.zeros(nsamps - 1)))
## filter this from one section to the next.
for i in np.arange(nsec):
	x = si.lfilter(Bs[i],As[i],x)

fig = pyl.figure('Impulse response')
pyl.plot(np.arange(nsamps), x)
pyl.grid(True)

## A simpler way to do magnitude response ?
fig = pyl.figure('Butter Magnitude response')
X = np.fft.fft(x)
f = np.arange(nsamps)*fs/nsamps
pyl.grid(True)
pyl.axis([0, fs/2, -100, 5])
pyl.plot(f[0:nsamps/2], 20*np.log10(X[0:nsamps/2]))
pyl.show()

