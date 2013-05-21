from numpy import *
from pylab import *
from scipy import signal as si

angles = linspace(0, pi, 180)
negangles = linspace(-pi, 0, 180)

figure('Tangent in [0, pi]')
subplot(211)
grid(True)
plot(angles, tan(angles))
subplot(212)
grid(True)
plot(negangles, tan(negangles))

## Tan vs arctan
figure('Tan vs arctan')
angles = linspace(0, 2*pi, 360)
tans = tan(angles)
rec_angles = arctan(tans)
plot(arange(len(angles)) * pi / 180, angles, 'b')
hold(True)
plot(arange(len(angles)) * pi / 180, rec_angles, 'r')
grid(True)

## Tan vs arctan2
figure('Tan vs arctan2')
angles = linspace(0, 2*pi, 360)
sins = sin(angles)
coss = cos(angles)
rec_angles = arctan2(sins, coss)
plot(arange(len(angles)) * pi / 180, angles, 'b')
hold(True)
plot(arange(len(angles)) * pi / 180, rec_angles, 'r')
grid(True)

## A better example - Aritificial phase unwrap
figure('Sine-Cos-Tan')
N=512
n=arange(N)
fo=1/N
A = pi + 0.01*pi
x = A*sin(2*pi*fo*n)
subplot(211)
grid(True)
plot(x)
xlabel('Sample index')
ylabel('Original phase in radians')
axis([0, 512, -6.5, 6.5])
title('The signal x whose amplitude exceeds the range [-$\pi,\pi$]')
subplot(212)
grid(True)
y = arctan2(sin(x), cos(x))

z = zeros(size(y))
for i in arange(size(y)):
	if (y[i - 1] - y[i]) > pi:
		y[i] = y[i] + 2*pi
	elif (y[i] - y[i - 1]) > pi:
		y[i] = y[i] - 2*pi
	else:
		y[i] = y[i]


plot(y)
xlabel('Sample index')
ylabel('Wrapped phase in radians')
axis([0, 512, -6.5, 6.5])
title('The wrapped phase')
show()

