## Complex cepstrum 
import pylab as pyl
import numpy as np
from numpy import fft
from scipy import signal as si
import wave as wv
import gaudiopak as gap
from pylab import specgram

fs = 1024
t = np.linspace(0, 1, fs)
f = 100
# sinewave = np.sin(2 * np.pi * f * t)
# pyl.figure('Sinewave')
# pyl.plot(t, sinewave)
# pyl.grid(True)

frames = 4000

read_file = wv.open('male.wav', 'rb')
chans, bytesize, samplerate, framecount, s1, c = read_file.getparams()
data_frames_s = read_file.readframes(frames)
data_frames = np.fromstring(data_frames_s, 'Int16')			
md = max(data_frames)
data_frames = data_frames.astype('float') / md

gap.put_audio_data('precepsd.wav', np.real(data_frames), chans, bytesize, samplerate, frames, md)

## spectrum
N_fft = 1 << int(np.log2(max(samplerate, frames)) + 1)

ceps_sound, fft_sine, angle_sound = gap.ceps(data_frames, N_fft, 'real')

pyl.figure('FFT of speech')
pyl.subplot(211)
pyl.plot(np.linspace(-samplerate/2, samplerate/2, N_fft), np.log10(fft.fftshift(abs(fft_sine))), 'g')
pyl.grid(True)
pyl.xlim([-samplerate/2, samplerate/2])
pyl.xlabel('Hz')

## Real cepstrum
pyl.subplot(212)
abs_cs = (ceps_sound)
pyl.plot(np.linspace(-samplerate/2, samplerate/2, N_fft), fft.fftshift((abs_cs)), 'r')
pyl.grid(True)
pyl.xlim([-samplerate/2, samplerate/2])
pyl.xlabel('Hz')

# pyl.subplot(212)
# pyl.plot(np.linspace(-np.pi, np.pi, N_fft), np.unwrap(np.angle(fft_sine)))
# pyl.grid(True)
# pyl.xlim([0, np.pi])

## Window the cepstrum
# Hamming
wnd_size_div = 32
wndw = gap.wndow(N_fft, N_fft/wnd_size_div, kind = 'kaiser', shaper = 400)
windowed_ceps = wndw*fft.fftshift(abs_cs)

pyl.figure('Windowed ceps')
pyl.subplot(212)
#pyl.plot(np.linspace(-samplerate/2, samplerate/2, N_fft), wndw, 'k-')
pyl.plot(np.linspace(-samplerate/2, samplerate/2, N_fft), ((windowed_ceps)), 'r')
pyl.grid(True)
#pyl.xlim([-500, 500])
pyl.xlabel('Hz')

windowed_ceps = fft.fftshift(windowed_ceps)
fft_win_ceps = fft.fft(windowed_ceps, N_fft)
pyl.subplot(211)
pyl.plot(np.linspace(-samplerate/2, samplerate/2, N_fft), np.log10(fft.fftshift(abs(fft_sine))), 'g')
pyl.plot(np.linspace(-samplerate/2, samplerate/2, N_fft), np.real(fft.fftshift((fft_win_ceps))), 'm')
diff_ceps = np.log10((abs(fft_sine))) - np.real(((fft_win_ceps)))
pyl.plot(np.linspace(-samplerate/2, samplerate/2, N_fft), (fft.fftshift(diff_ceps)), 'k')
pyl.grid(True)
pyl.xlim([-samplerate/2, samplerate/2])


############
# 
unceps = 10**np.real(fft_win_ceps)*np.exp(np.angle(fft_sine)*1j)
unceps = fft.ifft(unceps, N_fft)
unceps_diff = 10**np.real(diff_ceps)*np.exp(np.angle(fft_sine)*1j)
unceps_diff = fft.ifft(unceps_diff, N_fft)
pyl.figure()
pyl.subplot(211)
pyl.plot(np.real(unceps), 'b')
pyl.grid(True)
#pyl.plot(np.real(data_frames), 'r')
pyl.subplot(212)
pyl.plot(np.real(unceps_diff), 'r')
pyl.grid(True)

gap.put_audio_data('cepsd.wav', np.real(unceps), chans, bytesize, samplerate, frames, md)

pyl.figure('Spectrogram')
pyl.subplot(311)
specgram(data_frames, NFFT=128, noverlap=0)
pyl.subplot(312)
specgram(np.real(unceps[0:frames]), NFFT=128, noverlap=0)
pyl.subplot(313)
specgram(np.real(unceps_diff[0:frames]), NFFT=128, noverlap=0)

## Show all plots.
pyl.show()
