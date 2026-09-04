import torch
import torch_mlir

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self, x):
        return torch.relu(self.linear(x))

model = TinyModel()
example_input = torch.randn(1, 4)

# Pass the model and example input(s), not the exported program
mlir_module = torch_mlir.compile(
    model,
    example_input,               # or (example_input,)
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
)

print(mlir_module)