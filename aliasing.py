import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

###############################################################################
# Generate an oversampled sinewave.
oversampl_factor = 100 # 100 samples per cycle.
# frequency
f = 100
display_cycle = 5
# Compute the time axis.
t = np.arange(0, np.double(display_cycle)/f, 1.0/(f*oversampl_factor))
# 
sine_os = np.sin(2 * np.pi * f * t)

# plot.
plt.figure(1)
plt.title('Oversampled sinewave')
plt.plot(sine_os)
plt.grid(1)


###############################################################################
# Sample this sinewave
fs = 900 # Hz
oversampl_factor = fs/f
# Compute the time axis.
t = np.arange(0, np.double(display_cycle)/f, 1.0/(f*oversampl_factor))
# 
sine_os = np.sin(2 * np.pi * f * t)

# plot.
plt.figure(2)
plt.title('Sampled sinewave')
plt.stem(sine_os, 'r')
plt.grid(1)


plt.show()

