import wave as wv
import numpy as np
from numpy import fft
from scipy import signal as si


###############################################################################
# Simple wave reader.
###############################################################################
def get_audio_data(file_name, frames = -1):
	"""
	
	"""
	wr = wv.open(file_name, 'rb')

	# frame count
	fra_cnt = wr.getnframes()
	# channels
	chans = wr.getnchannels()
	# sampling f
	sampl_f = wr.getframerate()
	# read frames.
	aud_data = wr.readframes(frames)
	# sample size
	samp_sz = wr.getsampwidth()
	#
	wr.close()
	#
	if samp_sz == 2:
		aud_data_int = np.fromstring(aud_data, 'Int16')
	else:
		aud_data_int = np.fromstring(aud_data, 'Int8')
	# return
	return aud_data_int, sampl_f, samp_sz, chans, fra_cnt
	

###############################################################################
# Simple wave writer.
###############################################################################
def put_audio_data(file_name, data, chans, samp_sz, sampl_f, frames, scaler):
	"""
	
	"""
	wr = wv.open(file_name, 'wb')
	wr.setnchannels(chans)
	wr.setsampwidth(samp_sz)
	wr.setframerate(sampl_f)
	wr.setnframes(frames)
	if samp_sz == 2:
		wr_type = 'Int16'
	else:
		wr_type = 'Int8'
	str_list = (data*scaler).astype(wr_type).tostring()
	wr.writeframes(str_list)
	wr.close()

###############################################################################
# Cepstrum
###############################################################################
def ceps(frame, fft_N = -1, kind = 'cmplx'):
	if fft_N == -1:
		fft_N = 1 << int(np.log2(len(frame)) + 1)

	fft_frame = fft.fft(frame, fft_N)
	angle_frame = np.angle(fft_frame) # We will return the angle as well.

	if kind == 'cmplx':
		ceps_frame = fft.ifft(np.log10(fft_frame), fft_N)
	else:
		ceps_frame = fft.ifft(np.log10(abs(fft_frame)), fft_N)
	  
	return ceps_frame, fft_frame, angle_frame
	
###############################################################################
# Window of kind K, with parameter P and size
###############################################################################	
def wndow(win_size_N, out_vector_N = -1, **kwargs):
	if win_size_N > out_vector_N:
		out_vector_N = win_size_N
	
	wnd_size_div = out_vector_N / win_size_N
	
	wndw_kind = kwargs.get('kind', 'hamming')
	
	wndw = np.arange(1)
	
	if wndw_kind == 'kaiser':
		shaper = kwargs.get('shaper', 1)
		wndw_s = si.kaiser(win_size_N, shaper)
		wndw = np.concatenate([np.zeros((wnd_size_div -
		1)*out_vector_N/(2*wnd_size_div)), wndw_s])
		wndw = np.concatenate([wndw, np.zeros((wnd_size_div -
		1)*out_vector_N/(2*wnd_size_div))])

	return wndw
	


