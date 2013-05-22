from numpy import *
from pylab import *
from scipy import signal as si

#
# Copyright (c) 2011 Christopher Felton
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# The following is derived from the slides presented by
# Alexander Kain for CS506/606 "Special Topics: Speech Signal Processing"
# CSLU / OHSU, Spring Term 2011.

import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
    
def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0,
              markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    

    return z, p, k

###############################################################################
# Magnitude and phase response
###############################################################################
def gfreqz(b, a = [1], N = 512, whole = False, Fs = 2, **plotargs):
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
	Fs: float, optional
		Sampling frequency. Useful for plotting.
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
	threedb	3dB point marker on the magnitude plot
	logx	Use a log plot on the X axis
		

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
	mylim = plotargs.get('ylimmag', [-300, 0])
	threedbpt = plotargs.get('threedb', False)
	logxaxis = plotargs.get('logx', False)

	if magplot != None:
		dBH = 20*log10(abs(h))
		dBH -= max(dBH)

		if logxaxis == True:
			magplot.semilogx(w*Fs/2, dBH, c, label = l) 
		else:
			magplot.plot(w*Fs/2, dBH, c, label = l) 
		magplot.set_ylim(mylim)

		if threedbpt == True:
			if mylim[len(mylim) - 1] == 0:
				magplot.hold(True)
				dbpt = -3*ones(len(w))
				if logxaxis == True:
					magplot.semilogx(w*Fs/2, dbpt, 'r--')
				else:
					magplot.plot(w*Fs/2, dbpt, 'r--')

	magplot.grid(True)

	if angplot != None:
		angplot.plot(w, unwrap(angle(h)), c, label = l)
		angplot.grid(True)

	return w, h

###############################################################################
# Impulse response
###############################################################################
def gimpz(b, a = [1], N = 512):
	"""
	Returns the impulse response of a digital filter defined
	as a difference equation.
	"""
	dirac_d = zeros(N)
	dirac_d[0] = 1
	y = si.lfilter(b, a, dirac_d)
	return y, range(len(y))

###############################################################################
# group delay
###############################################################################
def ggroupdelay(b, a = [1], N=2**12):
	"""
	Returns the group delay of a filter defined by its coefficients. This 
	simple implementation uses the factor that the phase response can be shown
	to effectively be,
	Tg = real[DFT(n.h[n]) / DFT(h[n])]
	"""
	
	fft_N = N

	# Get the impulse response.	
	h, w = gimpz(b, a, fft_N)
	# denominator	
	W, denom = si.freqz(h, 1, fft_N) 
	# ramped impulse.
	ramped_impulse = arange(len(h)) * h
	# numerator
	W, numer = si.freqz(ramped_impulse, 1, fft_N) 
	# group delay
	grp_dly = fftshift(real(numer/denom))
	return fft_N, grp_dly
	
	

