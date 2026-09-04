import torch
import torch.nn as nn

# Replace the torchvision resnet18 with a minimal model
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Match the input shape (3*64*64 = 12288) and output 1000 classes
        self.fc = nn.Linear(3*64*64, 1000)

    def forward(self, x):
        # Flatten the image: (batch, 3, 64, 64) -> (batch, 3*64*64)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = TinyModel()
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# Continue with the autograd example
prediction = model(data)
loss = (prediction - labels).sum()
loss.backward()

# Before gradient descent
print(model.fc.weight)

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step() #gradient descent

# After gradient descent
print(model.fc.weight)


