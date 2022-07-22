import torch

x = torch.randn([4, 4])
print(x.shape)
print(torch.argmax(x, dim=1).shape)
print(torch.argmax(x, dim=1).unsqueeze(1).shape)

index = torch.tensor([[2, 1, 0]])
tensor_0 = torch.arange(3, 12).view(3, 3)
index = torch.tensor([[2, 1, 0]])
tensor_1 = tensor_0.gather(0, index)
print(tensor_1)
print(tensor_1.shape)