import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

num_values = 5000
plot_idx = 1


# Generate randoms (uniform) in [0, 1).
u1 = random.rand(1, num_values)
y_axis = np.arange(num_values)
# Transpose this.
plt.figure(plot_idx)
plot_idx = plot_idx + 1
plt.title("uniform randoms")
plt.grid(1)
plt.plot(y_axis, np.transpose(u1))

###############################################################################
# do the histogram, with 1000 bins.
bins = 1000
hist_u1 = np.histogram(u1, bins)
# plot the histogram.
plt.figure(plot_idx)
plot_idx = plot_idx + 1
plt.title("uniform histogram")
plt.stem(hist_u1[0])
plt.grid(1)

###############################################################################
# Generate another set of uniforms, for a nice little triangle distribution
u2 = random.rand(1, num_values)

tri1 = u1 + u2

hist_tri1 = np.histogram(tri1, bins)
plt.figure(plot_idx)
plot_idx = plot_idx + 1
plt.title("triangle histogram")
plt.stem(hist_tri1[0])
plt.grid(1)

###############################################################################
# Box Muller.
gauss1 = np.sqrt(-2 * np.log(u1)) * np.cos(2*np.pi*u2)

hist_gauss1 = np.histogram(gauss1, bins)
plt.figure(plot_idx)
plot_idx = plot_idx + 1
plt.title("gaussian histogram")
plt.plot(hist_gauss1[0])
plt.grid(1)

###############################################################################
# Show.
plt.show()
