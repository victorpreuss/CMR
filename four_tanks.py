import math
import numpy as np
import matplotlib.pyplot as plt
import control

# plt.close('all')

# cross section of the tanks (cm^2)
A1 = 28.0
A2 = 32.0
A3 = 28.0
A4 = 32.0

# cross section of the outlet hole (cm^2)
a1 = 0.071
a2 = 0.057
a3 = 0.071
a4 = 0.057

# measuring sensor gain (V/cm)
kc = 0.50

# acceleration of gravity (cm/s^2)
g = 981.0

# proportionality between voltage applied to the pump and flow (cm^3/Vs)
k1 = 3.33
k2 = 3.35

# valves opening percentage
gama1 = 0.70
gama2 = 0.60

# voltage applied to the pumps (V)
v1 = 3.0
v2 = 3.0

# simulation parameters
dt = 0.1
sim_time = 1000.0
num_points = int(sim_time / dt)

# steady-state values for tanks height
h1_bar = 1.0 / (2.0 * g * a1 ** 2) * ((1 - gama2) * k2 * v2 + gama1 * k1 * v1) ** 2
h2_bar = 1.0 / (2.0 * g * a2 ** 2) * ((1 - gama1) * k1 * v1 + gama2 * k2 * v2) ** 2
h3_bar = 1.0 / (2.0 * g * a3 ** 2) * ((1 - gama2) * k2 * v2) ** 2
h4_bar = 1.0 / (2.0 * g * a4 ** 2) * ((1 - gama1) * k1 * v1) ** 2

print("h1 steady-state: %.4lf" % h1_bar)
print("h2 steady-state: %.4lf" % h2_bar)
print("h3 steady-state: %.4lf" % h3_bar)
print("h4 steady-state: %.4lf" % h4_bar)

# time vector
t = np.zeros((num_points+1, 1))

# height data vectors
h1 = np.zeros((num_points+1, 1))
h2 = np.zeros((num_points+1, 1))
h3 = np.zeros((num_points+1, 1))
h4 = np.zeros((num_points+1, 1))

for i in range(num_points):

    dh1 = 1.0 / A1 * (-a1 * math.sqrt(2.0 * g * h1[i]) + a3 * math.sqrt(2.0 * g * h3[i]) + gama1 * k1 * v1) * dt
    dh2 = 1.0 / A2 * (-a2 * math.sqrt(2.0 * g * h2[i]) + a4 * math.sqrt(2.0 * g * h4[i]) + gama2 * k2 * v2) * dt
    dh3 = 1.0 / A3 * (-a3 * math.sqrt(2.0 * g * h3[i]) + (1 - gama2) * k2 * v2) * dt
    dh4 = 1.0 / A4 * (-a4 * math.sqrt(2.0 * g * h4[i]) + (1 - gama1) * k1 * v1) * dt

    t[i+1] = (i + 1) * dt

    h1[i+1] = h1[i] + dh1
    h2[i+1] = h2[i] + dh2
    h3[i+1] = h3[i] + dh3
    h4[i+1] = h4[i] + dh4

A = np.array([[-a1/A1*math.sqrt(g/(2*h1_bar)), 0, a3/A1*math.sqrt(g/(2*h3_bar)), 0],
              [0, -a2/A2*math.sqrt(g/(2*h2_bar)), 0, a4/A2*math.sqrt(g/(2*h4_bar))],
              [0, 0, -a3/A3*math.sqrt(g/(2*h3_bar)), 0],
              [0, 0, 0, -a4/A4*math.sqrt(g/(2*h4_bar))]])

B = np.array([[gama1*k1/A1, 0],
              [0, gama2*k2/A2],
              [0, (1-gama2)*k2/A3],
              [(1-gama1)*k1/A4, 0]])

C = np.array([[kc, 0, 0, 0],
              [0, kc, 0, 0]])

D = np.array([[0, 0],
              [0, 0]])

sys = control.StateSpace(A, B, C, D)

X0 = [-h1_bar, -h2_bar, -h3_bar, -h4_bar]

u = np.array([-2*np.ones(10000), -2*np.ones(10000)])

T = np.arange(0, sim_time, dt)

(t_lin, y, h) = control.forced_response(sys, T=T, u=u, X0=X0)

# print(sys.pole())

# print(control.ss2tf(sys))

h = np.transpose(h)

# plot section
fig = plt.figure(figsize=(6, 8))

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(t, h1, label='h1')
ax1.plot(t, h2, label='h2')
ax1.plot(t, h3, label='h3')
ax1.plot(t, h4, label='h4')

ax1.set(title='Nonlinear model', xlabel='time (s)', ylabel='height (cm)')
ax1.grid()
ax1.legend()

ax2.plot(t_lin, h[:, 0] + h1_bar, label='h1')
ax2.plot(t_lin, h[:, 1] + h2_bar, label='h2')
ax2.plot(t_lin, h[:, 2] + h3_bar, label='h3')
ax2.plot(t_lin, h[:, 3] + h4_bar, label='h4')

ax2.set(title='Linearized model', xlabel='time (s)', ylabel='height (cm)')
ax2.grid()
ax2.legend()

fig.tight_layout()

plt.show()


