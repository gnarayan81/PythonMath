################################################################################
# Analysis of filter stability using reflection coefficients.

import numpy as np
import pylab as pl
import gfiltpak as gfp

B = np.array([1])
A = np.array([1, 0.65, 0.5, 0.195]) # roots = 3, 2

refl = gfp.durbinrefl(A)
		
# All ref coeffts.
print('Reflection coeffs -', refl)
	#print('Aflip - ', Aflip)
	#print('reflections -', refl)
	#print('lower - ', Alower)
	#print('')
pl.grid(True)
gfp.zplane(B, A)
