import numpy as np
import matplotlib.pyplot as plt
import geostat as geo

n = 139*2  # Number of nodes

x = np.linspace(0, 200, n)  # Creates n regular coordinates from 0 to 200
X = np.zeros((n,2))         # Assembles these values as a 2 D coordinate matrix to be abble to use pdist
X[:,0] = x


covar = geo.covariance(rang=30, sill=4,typ="spherical")
y = geo.unconditionnal_lu( X, covar , nsimuls=3 )

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set(xlabel='Position', ylabel='Simulation',
       title='Example of LU simulation')
ax.grid()
plt.show()