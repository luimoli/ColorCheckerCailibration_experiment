from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch

ccm = torch.tensor([[1655, -442, -189], [-248, 1466, -194], [-48, -770, 1842]], dtype=torch.float32)
rgb_data = torch.randint(0, 255, (3, 100))
rgb_data = rgb_data.float()

error_manual = torch.randn((3, 100)) * 16
rgb_target = ccm.mm(rgb_data)/1024.0
rgb_target_error = rgb_target + error_manual
ccm_calc1 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc2 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc3 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc5 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc6 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc7 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)

def squared_loss(rgb_tmp, rgb_ideal):
    return torch.sum((rgb_tmp-rgb_ideal)**2)

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad/batch_size;

def net(ccm_calc1, ccm_calc2, ccm_calc3, ccm_calc5, ccm_calc6, ccm_calc7, rgb_data):
    rgb_tmp = torch.zeros_like(rgb_data)
    rgb_tmp[0, :] = ((1024.0 - ccm_calc1 - ccm_calc2) * rgb_data[0, :] + ccm_calc1 * rgb_data[1, :] + ccm_calc2 * rgb_data[2, :]) / 1024.0
    rgb_tmp[1, :] = (ccm_calc3 * rgb_data[0, :] + (1024.0 - ccm_calc3 - ccm_calc5) * rgb_data[1, :] + ccm_calc5 * rgb_data[2, :]) / 1024.0
    rgb_tmp[2, :] = (ccm_calc6 * rgb_data[0, :] + ccm_calc7 * rgb_data[1, :] + (1024.0 - ccm_calc6 - ccm_calc7) * rgb_data[2, :]) / 1024.0
    return rgb_tmp

lr = 3
num_epochs = 200
for epoch in range(num_epochs):
    l = squared_loss(net(ccm_calc1, ccm_calc2, ccm_calc3, ccm_calc5, ccm_calc6, ccm_calc7, rgb_data), rgb_target_error)
    l.backward()
    sgd([ccm_calc1, ccm_calc2, ccm_calc3, ccm_calc5, ccm_calc6, ccm_calc7], lr, 100)
    ccm_calc1.grad.data.zero_()
    ccm_calc2.grad.data.zero_()
    ccm_calc3.grad.data.zero_()
    ccm_calc5.grad.data.zero_()
    ccm_calc6.grad.data.zero_()
    ccm_calc7.grad.data.zero_()
    print('epoch %d, loss %f'%(epoch, l))

res = torch.tensor([[1024.0 - ccm_calc1 - ccm_calc2, ccm_calc1, ccm_calc2],
                    [ccm_calc3, 1024.0-ccm_calc3-ccm_calc5, ccm_calc5],
                    [ccm_calc6, ccm_calc7, 1024.0-ccm_calc6-ccm_calc7]], dtype=torch.float32)
print(res)

rgb_apply_ccm = res.mm(rgb_data)/1024.0

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')

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

ax2.set_xlim(-80, 360)
ax2.set_ylim(-80, 360)
ax2.set_zlim(-80, 360)
ax2.set_xlabel('R')
ax2.set_ylabel('G')
ax2.set_zlabel('B')
ax2.scatter(x3, y3, z3, marker='o', c='c', label='target rgb')

x4 = rgb_apply_ccm[0]
y4 = rgb_apply_ccm[1]
z4 = rgb_apply_ccm[2]
ax2.scatter(x4, y4, z4, marker='^', c='b', label='apply ccm rgb')
ax2.legend()

plt.show()