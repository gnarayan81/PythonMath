from numpy import *
from pylab import *
from scipy import signal as si

def freqzp(b, a = [1], N = 512, whole = False, **plotargs):
	"""
	Returns the magnitude and phase response of a digital filter defined
	as a transfer function as follows.

                    jw    		 -jw            -jmw
        	    jw  B(e)    b[0] + b[1]e + .... + b[m]e
	H(e) = ---- = ------------------------------------
                	  jw               -jw            -jnw
                 A(e)    a[0] + a[1]e + .... + a[n]e

	Additionally, this function plots the responses using user-supplied magnitude
	and phase subplot handles.

	Parameters
	----------
	b : ndarray
		numerator of a linear filter
	a : ndarray
		denominator of a linear filter
	worN : {None, int}, optional
		If None, then compute at 512 frequencies around the unit circle.
		If a single integer, then compute at that many frequencies.
		Otherwise, compute the response at frequencies given in worN
	whole : bool, optional
		Normally, frequencies are computed from 0 to pi (upper-half of
		unit-circle).  If `whole` is True, compute frequencies from 0 to 2*pi.

	**plotargs: dict, optional
		This dictionary specifies plotting parameters. 
	
	====  	===========
	key   	parameter
	====  	===========
	magfig	subplot handle of magnitude
	angfig	subplot handle of phase
	color	plot color
	label	plot label	
	ylimmag ylimit list for magnitude plot
	ylimang ylimit list for phase plot
		

	Returns
	-------
	w : ndarray
		The frequencies at which h was computed.
	h : ndarray
    	The frequency response.

	"""
	w, h = si.freqz(b, a, worN = N, whole = False)
	c = plotargs.get('color', 'b')
	magplot = plotargs.get('magfig', None)
	angplot = plotargs.get('angfig', None)
	l = plotargs.get('label', '')	
	mylim = plotargs.get('ylimmag', [-100, 0])

	if magplot != None:
		dBH = 20*log(abs(h))
		dBH -= max(dBH)
		magplot.plot(w, dBH, c, label = l) 
		magplot.set_ylim(mylim)
		magplot.grid(True)

	if angplot != None:
		angplot.plot(w, unwrap(angle(h)), c, label = l)
		angplot.grid(True)

	return w, h
