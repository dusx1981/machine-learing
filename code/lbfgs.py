import torch

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

if __name__ == "__main__":
    x = torch.tensor([1.3], requires_grad=True)
    y = torch.tensor([6.7], requires_grad=True)

    # 这里不需要自己设置学习率，会影响收敛速度
    # optimizer = torch.optim.LBFGS([x, y], lr=0.001)
    optimizer = torch.optim.LBFGS([x, y])

    epoch=[0]
    for i in range(20):
        def closure():
            optimizer.zero_grad()
            z = rosenbrock(x, y)
            z.backward()
            epoch[0] +=1

            print(
                    f'第 {epoch[0]} 次迭代后, '
                    f'x = {x.item():.3f}, '
                    f'y = {y.item():.3f}, '
                    f'z = {z.item():.3f}'
                )

            return z

        optimizer.step(closure)