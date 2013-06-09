import numpy as np
import gfiltpak as gfp

## First get the 10th order filter.
B, A = gfp.gzp2tf(np.ones(10), 0.9*np.ones(10), 1)
## Get the zeros, poles and gain
z, p, k = gfp.gtf2zp(B, A)

## Errors.
err_z = np.linalg.norm((z) - np.ones(10)) / np.linalg.norm(np.ones(10))
err_p = np.linalg.norm((p) - 0.9*np.ones(10)) / \
	np.linalg.norm(0.9*np.ones(10))