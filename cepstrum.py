## Complex cepstrum 
import pylab as pyl
import numpy as np
from numpy import fft
from scipy import signal as si
import wave as wv

fs = 1024
t = np.linspace(0, 1, fs)
f = 100
# sinewave = np.sin(2 * np.pi * f * t)
# pyl.figure('Sinewave')
# pyl.plot(t, sinewave)
# pyl.grid(True)

frames = 512

read_file = wv.open('male.wav', 'rb')
chans, bytesize, samplerate, framecount, s1, c = read_file.getparams()
data_frames_s = read_file.readframes(frames)
data_frames = np.fromstring(data_frames_s, 'Int16')
data_frames = data_frames.astype('float') / max(data_frames)
## spectrum
N_fft = 1 << int(np.log2(max(samplerate, frames)) + 1)
fft_sine = fft.fft(data_frames, N_fft)
pyl.figure('FFT of speech')
pyl.subplot(211)
pyl.plot(np.linspace(-samplerate/2, samplerate/2, N_fft), np.log10(fft.fftshift(abs(fft_sine))), 'g')
pyl.grid(True)
pyl.xlim([-samplerate/2, samplerate/2])
pyl.xlabel('Hz')

## Real cepstrum
ceps_sound = fft.ifft(np.log10(abs(fft_sine)), N_fft)
pyl.subplot(212)
abs_cs = (ceps_sound)
pyl.plot(np.linspace(-samplerate/2, samplerate/2, N_fft), fft.fftshift((abs_cs)), 'r')
pyl.grid(True)
pyl.xlim([-500, 500])
pyl.xlabel('Hz')

# pyl.subplot(212)
# pyl.plot(np.linspace(-np.pi, np.pi, N_fft), np.unwrap(np.angle(fft_sine)))
# pyl.grid(True)
# pyl.xlim([0, np.pi])

## Window the cepstrum
# Hamming
wndw = si.kaiser(N_fft, 700)
windowed_ceps = wndw * fft.fftshift(abs_cs)
pyl.figure('Windowed ceps')
pyl.subplot(211)
pyl.plot(np.linspace(-samplerate/2, samplerate/2, N_fft), wndw, 'k-')
pyl.plot(np.linspace(-samplerate/2, samplerate/2, N_fft), ((windowed_ceps)), 'r')
pyl.grid(True)
pyl.xlim([-500, 500])
pyl.xlabel('Hz')

## Show all plots.
pyl.show()