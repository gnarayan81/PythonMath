import wave as wv
import numpy as np


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
	str_list = bytes((data*scaler).astype(wr_type))
	wr.writeframes(str_list)
	wr.close()

