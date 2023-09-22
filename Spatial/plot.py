import matplotlib.pyplot as plt
import torch

ccm = torch.tensor([[1655, -442, -189], [-248, 1466, -194], [-48, -770, 1842]], dtype=torch.float32)
rgb_data = torch.randint(0, 255, (3, 100))
rgb_data = rgb_data.float()

rgb_target = ccm.mm(rgb_data)/1024.0

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
x2 = rgb_data[0]
y2 = rgb_data[1]
z2 = rgb_data[2]

ax1.scatter(x2, y2, z2, marker='*', c='b', label='origin RGB')

ax1.set_xlim(-80, 360)
ax1.set_ylim(-80, 360)
ax1.set_zlim(-80, 360)
ax1.set_xlabel('R')
ax1.set_ylabel('G')
ax1.set_zlabel('B')

x3 = rgb_target[0]
y3 = rgb_target[1]
z3 = rgb_target[2]

ax1.scatter(x3, y3, z3, marker='o', c='c', label='target rgb')

for i in range(len(x3)):
    ax1.plot([x2[i], x3[i]], [y2[i], y3[i]], [z2[i], z3[i]], 'k-.')
ax1.legend()

plt.show()