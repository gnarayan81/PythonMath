###############################################################################
## Copyright (C) 2013 Gopal Narayanan <gopal.narayan@gmail.com>
##
## This file is a direct port of JOS's vowel 'a' formant found here:
## https://ccrma.stanford.edu/~jos/filters/Formant_Filtering_Example.html
##
## Octave is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or (at
## your option) any later version.
##
## Octave is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.
##
## Octave is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or (at
## your option) any later version.
##
##
## You should have received a copy of the GNU General Public License
## along with Octave; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.
###############################################################################
## TODO: Details


import numpy as np
from gfiltpak import *
import pylab as pyl
import scipy.signal as si
import gaudiopak as gap

## Formant frequencies.
F = np.array([700, 1220, 2600])
## Formant half-power bandwidths.
BW = np.array([130, 70, 160])
## Sampling frequency.
fs = 8192 # 8 kHz

## Second order section count.
nsecs = F.size

## Determine the pole radius per formant. R = exp(-pi*B/fs)
R = np.exp(-np.pi*BW/fs) # EXTREMELY close to the UC, causing resonance.
## Determine the pole angles per formant. theta = 2*pi*F/fs
theta = 2*np.pi*F/fs
## Now the complex poles.
poles = R*exp(theta*1j)
## Get the monic polynomials. This is an all-pole filter.
B = np.ones((1))
# Denominator.
A = np.real(np.poly(np.concatenate((poles, np.conj(poles)))))
## Verify if the frequency response looks good.
pfig = pyl.figure('Frequency response of formants')
mag_fig = pfig.add_subplot(211)
ang_fig = pfig.add_subplot(212)
gfreqz(B,A,N=512,magfig = mag_fig, angfig = ang_fig, ylimmag = [-40, 40], logy
= True, normalize = False)


## Convert to parallel complex ONE POLE sections (complex resonators).
## Get the pole residues. f is the FIR section which will be 0.
[r, p, f] = si.residuez(B, A)

## Conjugate poles seem adjacent with scipy as well. So the creation of a
## genuine SOS is trivial without the need for cplxpair(...).
As = np.zeros((nsecs, 3))	## 3 coefficients per.
Bs = np.zeros((nsecs, 3))	## 	"

## This specific biquad section follows from JOS's derivation of conjugate
## complex (single) pole resonators. Suffice to say, each complex resonator
## biquad can be reduce to, 
## (r + r' - (r*p'+r'*p)z^-1)/(1-(p+p')z^-1+(p*p')z^-2) where r/p' is the
## complex conjugate of r/p.
for i in np.arange(0, 2*nsecs - 1, 2):
	k = i / 2
	#print('+++++++++++', r[i] + r[i + 1])
	#print('+++++++++++', -(r[i] * p[i + 1] + p[i] * r[i + 1]))
	Bs[k][:] = np.array(np.real([r[i] + r[i + 1], -(r[i] * p[i + 1] +
p[i] * r[i + 1]), 0]))
	As[k][:] = np.array(np.real([1, -(p[i] + p[i + 1]), (p[i] * p[i + 1])]))
	
## We now have nsecs parallel real resonators in biquad form that we can plot
## independently.
pfig.hold(True)

gfreqz(Bs[0],As[0],N=512,magfig = mag_fig, angfig = ang_fig, ylimmag =
[-40, 40], logy= True, normalize = False, color = 'm')
gfreqz(Bs[1],As[1],N=512,magfig = mag_fig, angfig = ang_fig, ylimmag =
[-40, 40], logy= True, normalize = False, color = 'g')
gfreqz(Bs[2],As[2],N=512,magfig = mag_fig, angfig = ang_fig, ylimmag =
[-40, 40], logy= True, normalize = False, color = 'r')

pfig.show()

## Now do vowel synthesis.
# First generate a bandlimited impulse train.
nsamps = 512
f0 = 200 # Pitch in Hz
w0T = 2*pi*f0/fs # rads/S

## Compute the number of harmonics in [0, fs/2)
nharm = np.floor((fs/2)/f0)		# That is, the number of copies of f0 in fs/2.

## Bandlimited impulse.
sig = np.zeros(nsamps)
## time index.
n = np.arange(nsamps)
## Synthesize the train.
## This is as per Dodge and Jerse (1985), where a bandlimited impulse train is
## simply a finite number of cosines (since in the frequency domain, they form
## impulses, which resemble a picket fence.
## We consider only a harmonic # of components because it makes sense.
## n * f0.
for i in np.arange(nharm):
	sig = sig + np.cos(i*w0T*n)

sig /= sig.max()
fig = pyl.figure('Bandlimited impulse train')
t_wv = fig.add_subplot(211)
t_wv.plot(np.arange(nsamps), sig)
t_wv.grid(True)
mag_p = fig.add_subplot(212)
fft_N = 512
fft_sig = np.abs(np.fft.fftshift(np.fft.fft(sig, fft_N)))
mag_p.plot(np.arange(fft_N/2), 20*np.log10(fft_sig[fft_N/2:fft_N]))
pyl.grid(True)

## Filter this impulse.
speech = si.filtfilt([1], A, sig) # lfilter spectrum looks awful.
speech /= speech.max()
fig = pyl.figure('Vowel A')
t_wv = fig.add_subplot(211)
t_wv.plot(np.arange(nsamps), speech)
t_wv.grid(True)
mag_p = fig.add_subplot(212)
fft_N = 512
fft_sig = np.abs(np.fft.fftshift(np.fft.fft(speech, fft_N)))
mag_p.plot(np.arange(fft_N/2), 20*np.log10(fft_sig[fft_N/2:fft_N]))
pyl.grid(True)


full_s = np.concatenate((sig, speech))
full_s /= full_s.max()

## Plot it.
fig = pyl.figure('Vowel A with impulse')
t_wv = fig.add_subplot(211)
t_wv.plot(np.arange(nsamps*2), full_s)
t_wv.grid(True)
mag_p = fig.add_subplot(212)
fft_N = 512
fft_sig = np.abs(np.fft.fftshift(np.fft.fft(full_s, fft_N)))
mag_p.plot(np.arange(fft_N/2), 20*np.log10(fft_sig[fft_N/2:fft_N]))
pyl.grid(True)
pyl.show()

## Write the wav file.
gap.put_audio_data('a.wav', np.concatenate((sig, speech)), 1, 2, 8192, nsamps*2,
2**15)