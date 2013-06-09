from numpy import *
from pylab import *
from scipy import signal as si
from scipy import sparse as sr

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
	mxlim = plotargs.get('xlimmag', [0, np.pi])
	threedbpt = plotargs.get('threedb', False)
	logxaxis = plotargs.get('logx', False)
	logyaxis = plotargs.get('logy', True)
	normalize = plotargs.get('normalize', False)

	if magplot != None:
		if logyaxis != False:
			dBH = 20*log10(abs(h))
			if normalize == True:
				dBH -= max(dBH)
		else:
			dBH = abs(h)
			if normalize == True:
				dBH /= max(dBH)

		if logxaxis == True:
			magplot.semilogx(w*Fs/2, dBH, c, label = l) 
		else:
			magplot.plot(w*Fs/2, dBH, c, label = l) 
		magplot.set_ylim(mylim)
		magplot.set_xlim(mxlim)

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
	
################################################################################
# Reflection coefficients using Durbin's algorithm
################################################################################
def durbinrefl(A, ordN = -1):
	if ordN < 1: 
		ordN = A.size
	
	refl = np.zeros(ordN - 1) # reflection coefficients
	
	Alower = A # prime
	
	for i in range(ordN - 1):
		Aflip = Alower[::-1] # flip
		refl[refl.size - i - 1] = Alower[Alower.size - 1] # kN
		Alower = (Alower - refl[refl.size - i - 1] * Aflip) / (1 -
		refl[refl.size - i - 1]**2)
		Alower = Alower[:-1] # Get the last N - 1
		#if np.abs(refl[refl.size - i - 1]) >= 1.0:
			#print('Pole outside unit circle at', refl[refl.size - i -
			#1], '.')
	return refl

################################################################################
# Powers of number between s and e
################################################################################
def powersofnum(num, s, e):
	pws = np.ones(e - s)
		
	for i in range(e - s):
		pws[i] = num ** (i + s)
			
	return pws

################################################################################
# The Schur-Cohn criterion for stability states that when a polynomial D(z) is
# evaluated for zeros (i.e., as a denominator in the H(z) = N(z)/D(z) form), the
# system is stable as long as, 
# 1. D(1) > 0. i.e., on the unit circle +
# 2. D(-1) < 0 for N odd and D(-1) > 0 for N even, where N is the order of D.
# all quotient terms of the following sequence, 
# Dk@(z)/Dk(z) = qk + Dk-1@(z)/Dk(z)
# conform to |qk| < 1. These are the reflection coefficients. D@(z) is the
# flipped version of D(z) such that D@(z) = (z^N)D(1/z), aka the reciprocal
# polynomial.
################################################################################
def schurcohnstability(A, ordN = -1):
	"""
				
	"""
	if ordN == -1:
		ordN = A.size
					
	# First criterion - add all terms.
	coef_sum = A.sum()
					
	if coef_sum < 0.0:
		return False
	else:
		# Second criterion
		pw_m1 = powersofnum(-1, 0, ordN)
		coef_sum = (A * pw_m1).sum()
		
		if ordN % 2 == 1: #odd
			if coef_sum >= 0.0:
				return False
		else: # even
			if coef_sum <= 0.0:
				return False
									
		# Third criterion
		refls = durbinrefl(A)
		if sum(np.abs(refls) >= 1.0) > 0.0: 
			return False
		else:
			return True
			
###############################################################################
## Copyright (C) 2013 Gopal Narayanan <gopal.narayan@gmail.com>
##
## This file is part of gfiltpak. It was ported from its GNU Octave counterpart
## named cplxpair.m (signal package) on 08 June 2013. I have included its 
## license header below.
##
##
## Copyright (C) 2000-2012 Paul Kienzle
##
## This file is part of Octave.
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
##
## -*- texinfo -*-
## @deftypefn  {Function File} {} cplxpair (@var{z})
## @deftypefnx {Function File} {} cplxpair (@var{z}, @var{tol})
## @deftypefnx {Function File} {} cplxpair (@var{z}, @var{tol}, @var{dim})
## Sort the numbers @var{z} into complex conjugate pairs ordered by
## increasing real part.  Place the negative imaginary complex number
## first within each pair.  Place all the real numbers (those with
## @code{abs (imag (@var{z}) / @var{z}) < @var{tol})}) after the
## complex pairs.
##
## If @var{tol} is unspecified the default value is 100*@code{eps}.
##
## By default the complex pairs are sorted along the first non-singleton
## dimension of @var{z}.  If @var{dim} is specified, then the complex
## pairs are sorted along this dimension.
##
## Signal an error if some complex numbers could not be paired.  Signal an
## error if all complex numbers are not exact conjugates (to within
## @var{tol}).  Note that there is no defined order for pairs with identical
## real parts but differing imaginary parts.
## @c Set example in small font to prevent overfull line
##
## @smallexample
## cplxpair (exp(2i*pi*[0:4]'/5)) == exp(2i*pi*[3; 2; 4; 1; 0]/5)
## @end smallexample
## @end deftypefn
###############################################################################
def gcplxpair(cp, tol = 100 * np.spacing(1), dim = -1):
	
	if type(cp) != np.ndarray:
		z = np.array(cp)
	else:
		z = cp
	
	# Get the dimension count
	nd = z.ndim
	# original dimensions
	orig_dims = np.array(z.shape) # In Octave this would be size.

	if dim == -1:
		## Find the first non-singleton dimension
		dim = 0
		while (dim < nd - 1) and (orig_dims[dim] == 1):
			dim = dim + 1
		#
		dim = dim + 1
		
		if dim >= nd:
			dim = 0
		else:
			dim = np.floor(dim)
			
			assert (dim >= 0) and (dim < nd), 'gcplxpair: invalid dimension \
			along which to sort'

	## Move dimension to treat first and convert to a 2D matrix.
	perm = np.concatenate((np.arange(dim, nd), np.arange(0, dim - 1)))
	## Permute. i.e., move dimensions around.
	z = np.transpose(z, perm)
	## shape
	sz = np.array(z.shape)
	## Get the first dimension. This may be a singleton
	n = sz[0]
	## Get the product of sz along the first non-singleton dimension.
	m = np.prod(sz) / n # This mostly does what  Octave's prod(...) does.
	## Reshape into a 2D matrix.
	z = np.reshape(z, (n, m))
	## Sort the REAL part of z row-wise.
	q = np.sort(z, 0)
	## Now do the 'arg' sort (index) which returns what it says.
	idx = np.argsort(z, 0)
	## For some reason, the Octave script reuses idx to re-sort z. I'll just
	## assign.
	z = q

	#print('z = ', z.transpose(), z.shape)
	## Put the purely real values at the end of the returned list.
	## We will find all non-complex values.
	arg_reals = np.argwhere(np.abs(np.imag(z)) / (np.abs(z)) < tol)
	#print('reals (x, y) = ', arg_reals.transpose()[0],
	#arg_reals.transpose()[1])

	## Create a sparse matrix with the positions of reals.
	rs = np.array(arg_reals.transpose()[0])
	cs = np.array(arg_reals.transpose()[1])
	ds = np.ones(rs.size)
	q = sr.csr_matrix((ds,(rs,cs)), shape = (n, m))
	# This sum counts the number of reals per column.
	nr = np.sum(q.toarray(), 0)
	# Do an arg sort just to get an array.
	idx = np.argsort(q.toarray(), 0)
	# Relegate the reals to the back.
	z = z[idx]

	# Stow it away. But plain assignment will just copy references, so use an
	# explicit copy constructor.
	y = np.array(z)

	#print('########################y = ', y, y.shape)
	#print('########################z =', z)

	## This loop will scan through the front of z and check for any mismatches.
	for j in np.arange(m):
		p = n - nr[j] # Loop over every column until the reals are reached.
		
		for i in np.arange(0, p, 2): # [0, p) steps of 2.
			assert i+1 < p, 'gcplxpair: could not pair all complex numbers'
			
			# Find the closest conjugate along this column.
			v = np.min(np.abs(z[i+1:p] - np.conj(z[i])))
			idx = np.argmin(np.abs(z[i+1:p] - np.conj(z[i])))
			
			assert v <= tol, 'gcplxpair: could not pair all complex numbers'
			
			## Always stick the negative in first.
			if np.imag(z[i]) < 0:
				#print('neg xchg')
				y[i]	= z[i]
				y[i+1] 	= z[idx + i + 1] 
			else:
				#print('pos xchg')
				y[i]	= z[idx + i + 1]
				y[i+1] 	= z[i] 
		
			
			# This will swap it. That is, it will move the potentially
			# unused
			# i + 1 to idx + i + 1.
			z[idx + i + 1] = z[i + 1]
			

	# Now, we just need to reshape it back to original dimensions.
	y = np.transpose(np.reshape(y, sz), perm[::-1]) # reverse.
	# Finally, return the value.
	return y
	
###############################################################################	
## Copyright (C) 2013 Gopal Narayanan <gopal.narayan@gmail.com>
##
## This file is part of gfiltpak. It was ported from its GNU Octave counterpart
## named cplxreal.m (signal package) on 08 June 2013. I have included its 
## license header below.
##
##
#%% Copyright (C) 2005 Julius O. Smith III <jos@ccrma.stanford.edu>
#%%
#%% This program is free software; you can redistribute it and/or modify it under
#%% the terms of the GNU General Public License as published by the Free Software
#%% Foundation; either version 3 of the License, or (at your option) any later
#%% version.
#%%
#%% This program is distributed in the hope that it will be useful, but WITHOUT
#%% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#%% FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
#%% details.
#%%
#%% You should have received a copy of the GNU General Public License along with
#%% this program; if not, see <http://www.gnu.org/licenses/>.

#%% -*- texinfo -*-
#%% @deftypefn {Function File} {[@var{zc}, @var{zr}] =} cplxreal (@var{z},
#@var{thresh})
#%% Split the vector z into its complex (@var{zc}) and real (@var{zr}) elements,
#%% eliminating one of each complex-conjugate pair.
#%%
#%% INPUTS:@*
#%%   @itemize
#%%   @item
#%%   @var{z}      = row- or column-vector of complex numbers@*
#%%   @item
#%%   @var{thresh} = tolerance threshold for numerical comparisons (default =
#100*eps)
#%%   @end itemize
#%%
#%% RETURNED:@*
#%%   @itemize
#%%   @item
#%% @var{zc} = elements of @var{z} having positive imaginary parts@*
#%%   @item
#%% @var{zr} = elements of @var{z} having zero imaginary part@*
#%%   @end itemize
#%%
#%% Each complex element of @var{z} is assumed to have a complex-conjugate
#%% counterpart elsewhere in @var{z} as well.  Elements are declared real
#%% if their imaginary parts have magnitude less than @var{thresh}.
#%%
#%% @seealso{cplxpair}
#%% @end deftypefn
###############################################################################
def gcplxreal(z, thresh = 100*np.spacing(1)):

	if z.size == 0:
		zc = np.array([])
		zr = np.array([])
	else:
		zcp = gcplxpair(z) 	## sort complex pairs, real roots at the end.
		nz = z.size			## get the size of the input.
		nzrsec = 0
		i = nz - 1			## start index into zcp
		
		## Loop until a complex number is found or the end is reached.
		while i > -1 and np.abs(np.imag(zcp[i])) < thresh:
			zcp[i] = np.real(zcp[i])
			nzrsec = nzrsec + 1
			i = i - 1
		
		nzsect2 = nz - nzrsec	## Get the number of complex pairs.
		
		## Being that they are pairs, they need to be conjugates.
		assert nzsect2 % 2 == 0, 'gcplxreal: Odd number of complex values'
		
		## Get the number of complex quantities.
		nzsec = nzsect2 / 2
		## Get the complex values (only the positive conjugates)
		zc = zcp[1:nzsect2:2]
		## Get the reals
		zr = zcp[nzsect2:nz]
		
	return zc, zr

################################################################################
## Copyright (C) 2013 Gopal Narayanan <gopal.narayan@gmail.com>
##
## This file is part of gfiltpak. It was ported from its GNU Octave counterpart
## named zp2sos.m (signal package) on 08 June 2013. I have included its 
## license header below.
##
##
#%% Copyright (C) 2005 Julius O. Smith III <jos@ccrma.stanford.edu>
#%%
#%% This program is free software; you can redistribute it and/or modify it under
#%% the terms of the GNU General Public License as published by the Free Software
#%% Foundation; either version 3 of the License, or (at your option) any later
#%% version.
#%%
#%% This program is distributed in the hope that it will be useful, but WITHOUT
#%% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#%% FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
#%% details.
#%%
#%% You should have received a copy of the GNU General Public License along with
#%% this program; if not, see <http://www.gnu.org/licenses/>.

#%% -*- texinfo -*-
#%% @deftypefn {Function File} {[@var{sos}, @var{g}] =} zp2sos (@var{z}, @var{p})
#%% @deftypefnx {Function File} {[@var{sos}, @var{g}] =} zp2sos (@var{z},
#@var{p}, @var{g})
#%% Convert filter poles and zeros to second-order sections.
#%%
#%% INPUTS:@*
#%% @itemize
#%% @item
#%%   @var{z} = column-vector containing the filter zeros@*
#%% @item
#%%   @var{p} = column-vector containing the filter poles@*
#%% @item
#%%   @var{g} = overall filter gain factor
#%%   If not given the gain is assumed to be 1.
#%% @end itemize
#%%
#%% RETURNED:
#%% @itemize
#%% @item
#%% @var{sos} = matrix of series second-order sections, one per row:@*
#%% @var{sos} = [@var{B1}.' @var{A1}.'; ...; @var{BN}.' @var{AN}.'], where@*
#%% @code{@var{B1}.'==[b0 b1 b2] and @var{A1}.'==[1 a1 a2]} for 
#%% section 1, etc.@*
#%% b0 must be nonzero for each section.@*
#%% See @code{filter()} for documentation of the
#%% second-order direct-form filter coefficients @var{B}i and
#%% %@var{A}i, i=1:N.
#%%
#%% @item
#%% @var{Bscale} is an overall gain factor that effectively scales
#%% any one of the @var{B}i vectors.
#%% @end itemize
#%% 
#%% EXAMPLE:
#%% @example
#%%   [z,p,g] = tf2zp([1 0 0 0 0 1],[1 0 0 0 0 .9]);
#%%   [sos,g] = zp2sos(z,p,g)
#%% 
#%% sos =
#%%    1.0000    0.6180    1.0000    1.0000    0.6051    0.9587
#%%    1.0000   -1.6180    1.0000    1.0000   -1.5843    0.9587
#%%    1.0000    1.0000         0    1.0000    0.9791         0
#%%
#%% g =
#%%     1
#%% @end example
#%%
#%% @seealso{sos2pz sos2tf tf2sos zp2tf tf2zp}
#%% @end deftypefn
################################################################################
def gzp2sos(z, p = np.array([]), g = 1):

	zc, zr = gcplxreal(z) 		## Get the complex and real vals of zeros.
	pc, pr = gcplxreal(p) 		## Get the complex and real vals of poles.

	## Get lengths.
	nzc = zc.size
	npc = pc.size

	nzr = zr.size
	npr = pr.size

	## Pair up real zeros.
	if nzr > 0:
		if nzr % 2 == 1: 			## Make even by stuffing a zero at the end.
			zr = np.concatenate((zr, np.zeros((1))))
			nzr = nzr + 1
		
		nzrsec = nzr / 2
		zrms = np.real(-zr[0:nzr-1:2] - zr[1:nzr:2]) 	## Negate-sum pairs
		zrp  = np.real(-zr[0:nzr-1:2] * zr[1:nzr:2]) 	## Product of pairs.
	else:
		nzrsec = 0
		
	## Pair up real poles.
	if npr > 0:
		if npr % 2 == 1: 			## Make even by stuffing a zero at the end.
			pr = np.concatenate((pr, np.zeros((1))))
			npr = npr + 1
		
		nprsec = npr / 2
		prms = np.real(-pr[0:npr-1:2] - pr[1:npr:2]) 	## Negate-sum pairs
		prp  = np.real(-pr[0:npr-1:2] * pr[1:npr:2]) 	## Product of pairs.
	else:
		nprsec = 0

	## Compute the number of second order sections. This is the maximum of
	## complex
	## zero count + half the number of real zeros and corresponding pole
	## operations.
	## That is, n_zeros / 2 and n_poles / 2. Think!
	## http://www.mathworks.com/help/signal/ref/zp2sos.html
	nsecs = np.max((nzc + nzrsec, npc + nprsec))

	## Convert the complex poles and zeros to real second order form.
	## Both polynomials are monic.
	zcm2r = -2*np.real(zc)				## 1 - 2 * re{z} + |z|^2
	zca2  = np.abs(zc)**2
	pcm2r = -2*np.real(pc)				## 1 - 2 * re{p} + |p|^2
	pca2  = np.abs(pc)**2

	sos = np.zeros((nsecs, 6))		## L * 6, always

	## Monic it.
	sos.transpose()[0] = np.ones(nsecs)
	sos.transpose()[3] = np.ones(nsecs)

	## Get the index of the last zero and pole... TODO Doc
	nzrl = nzc + nzrsec
	nprl = npc + nprsec

	for i in np.arange(nsecs):
		#print('i = ', i)
		if i < nzc:
			sos[i][1] = zcm2r[i]
			sos[i][2] = zca2[i]
		elif i < nzrl:
			sos[i][1] = zrms[i - nzc - 1]
			sos[i][2] = zrp[i - nzc - 1]


		if i < npc:
			sos[i][4] = pcm2r[i]
			sos[i][5] = pca2[i]
		elif i < nprl:
			sos[i][4] = prms[i - npc - 1]
			sos[i][5] = prp[i - npc - 1]

	## Return.
	return sos, g
	
################################################################################
## Copyright (C) 2013 Gopal Narayanan <gopal.narayan@gmail.com>
##
## This file is part of gfiltpak. 
################################################################################
def gzp2tf(z, p, k = 1):
	b = k * np.poly(z)
	a = np.poly(p)
	
	return b, a

################################################################################
## Copyright (C) 2013 Gopal Narayanan <gopal.narayan@gmail.com>
##
## This file is part of gfiltpak. 
################################################################################
def gtf2zp(B, A = np.array([1])):
	k = 1.0
	if B[0] != 1:
		k *= B[0]
		B /= k
	
	if A[0] != 1:
		k /= A[0]
		A /= k
	
	z = np.roots(B)
	p = np.roots(A)
	
	return z, p, k

