import torch

def f(x, y):
    return x*x + y*y

x = torch.tensor([1.1], requires_grad=True)
y = torch.tensor([2.1], requires_grad=True)
alpha = 0.05
n = 200

for i in range(1, n+1):
    z = f(x, y)
    z.backward()

    x.data -= alpha*x.grad.data
    y.data -= alpha*y.grad.data

    x.grad.zero_()
    y.grad.zero_()

    print(
            f'第 {i} 次迭代后，'
            f'x = {x.item():.3}, '
            f'y = {y.item():.3}, '
            f'f(x, y) = {z.item():.3}'
        )
    