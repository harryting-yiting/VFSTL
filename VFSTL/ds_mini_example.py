import torch
import torch.nn as nn

# Define the dynamic system (simple linear system)
class DynamicSystem(nn.Module):
    def __init__(self):
        super(DynamicSystem, self).__init__()
        
        self.A = torch.tensor([0.9])
        
    def forward(self, x, u):
        return self.A * x + u

# Define the cost function
def cost_function(x, u):
    return x**2 + u**2

# Define the discrete set of controls
discrete_controls = torch.tensor([0, 0.5, 1])

# Initial state and control
x_0 = torch.tensor([1.0], requires_grad=True)
u_bar = torch.tensor([1.0], requires_grad=True)

# Dynamic system
dynamic_system = DynamicSystem()

# Optimization loop
optimizer = torch.optim.SGD([u_bar], lr=0.1)

for i in range(10):
    # Update based on the continuous control
    x_1 = dynamic_system(x_0, u_bar)
    
    # Calculate cost
    cost = cost_function(x_1, u_bar)
    
    # Backpropagation
    optimizer.zero_grad()
    cost.backward(retain_graph=True)
    optimizer.step()
    
    # u_bar.grad.data.zero_()
    # Convert continuous control to discrete control
    u_bar = discrete_controls[(torch.abs(u_bar - discrete_controls)).argmin()]

    print("updating control:", u_bar.item())

    # Update the state
    x_0 = x_1

print("Optimized discrete control:", u_bar.item())