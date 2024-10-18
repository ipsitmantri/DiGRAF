import numpy as np
import torch
import torch.optim as optim
from activations import CPABActivationSame
## gen data:
import numpy as np
import matplotlib.pyplot as plt


def peaks(x, y):
    z = 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) \
        - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(-x ** 2 - y ** 2) \
        - 1 / 3 * np.exp(-(x + 1) ** 2 - y ** 2)
    return z


if False:
    # Define the grid
    x = np.linspace(-3, 3, 101)
    y = np.linspace(-3, 3, 101)
    x, y = np.meshgrid(x, y)

    # Calculate peaks
    z = peaks(x, y)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')

    plt.show()


##neural net:


class MLP(torch.nn.Module):
    def __init__(self, act=torch.tanh, cpab=True):
        super(MLP, self).__init__()
        self.hidden1 = torch.nn.Linear(in_features=2, out_features=64)
        self.hidden2 = torch.nn.Linear(in_features=64, out_features=64)
        self.output = torch.nn.Linear(in_features=64, out_features=1)
        #self.hidden1 = torch.nn.Parameter(1e-5*torch.randn(2, 64))
        self.act = act
        #self.hidden2 = torch.nn.Parameter(1e-5*torch.randn(64, 64))
        #self.hidden3 = torch.nn.Parameter(1e-3*torch.randn(1, 64))
        self.cpab = cpab
        if cpab:
            self.act = CPABActivationSame(
                radius=10,
                tess_size=16,
                channel=1,
                transform_theta=True,
                use_tanh=True
            )

    def forward(self, x):
        if not self.cpab:
            x = self.act(self.hidden1(x))
            x = self.act(self.hidden2(x))
        else:
            x,_ = self.act(self.hidden1(x), None, None, None, 1)
            x, _ = self.act(self.hidden2(x), None, None, None, 1)
        x = self.output(x)
        return x

ppp = torch.nn.Parameter(1e-5*torch.randn(2, 64)).cuda()
aaa = torch.randn(10,2).cuda()
ttt = aaa @ ppp
# Initialize the model, loss function and optimizer
model = MLP().to('cuda:0')
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

## create data:

# Generate sample data
x = np.linspace(-3, 3, 101)
y = np.linspace(-3, 3, 101)
x, y = np.meshgrid(x, y)
z = peaks(x, y)

# Flatten the data for training
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()

# Combine x and y into a single input array
inputs = np.vstack((x_flat, y_flat)).T
outputs = z_flat

## train:
train_curve = []
# Convert the data to PyTorch tensors
inputs_tensor = torch.tensor(inputs, dtype=torch.float).to('cuda:0')
outputs_tensor = torch.tensor(outputs, dtype=torch.float).unsqueeze(1).to('cuda:0')

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    predictions = model(inputs_tensor)

    # Compute the loss
    loss = criterion(predictions, outputs_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    train_curve.append(loss.item())
## visualize:
torch.save(train_curve, '/home/cluster/users/erant_group/moshe/approx_tanh.txt')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate predictions
model.eval()
with torch.no_grad():
    predictions = model(inputs_tensor.cuda()).detach().cpu().numpy().flatten()

# Reshape predictions to match the grid
predictions_grid = predictions.reshape(x.shape)

# Plot the original function
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
ax.set_title('Original Peaks Function')

# Plot the MLP approximation
ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(x, y, predictions_grid, cmap='viridis')
ax.set_title('MLP Approximation')

plt.show()

