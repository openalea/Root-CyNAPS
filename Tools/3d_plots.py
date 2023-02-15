from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = [k for k in range (10)]
ys = [k for k in range (10)]
zs = [k for k in range (10)]
ax.scatter(xs, ys, zs, marker='*')
x = [-k for k in range (10)]
y = [-k for k in range (10)]
z = [-k for k in range (10)]
ax.scatter(x, y, z, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
