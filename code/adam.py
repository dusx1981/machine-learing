import torch

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

if __name__ == "__main__":
    x = torch.tensor([1.3], requires_grad=True)
    y = torch.tensor([6.7], requires_grad=True)

    optimizer = torch.optim.Adam([x, y], lr=0.001)

    for i in range(30000):
        optimizer.zero_grad()

        z = rosenbrock(x, y)
        z.backward()
        optimizer.step()

        if i%1000 == 0:
            print(
                    f'第 {i+1} 次迭代后, '
                    f'x = {x.item():.3f}, '
                    f'y = {y.item():.3f}, '
                    f'z = {z.item():.3f}'
                )