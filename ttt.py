import torch
from torch import nn


class Net1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net1, self).__init__()
        self.layer = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())

    def forward(self, x):
        y = self.layer(x)
        return y


class Net2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net2, self).__init__()
        self.layer = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())

    def run(self, x):
        y = self.layer(x)
        return y


class Net3():
    def __init__(self, model_list):
        self.model_list = model_list
        
    def run(self, x):
        layer = self.model_list[0]
        x = layer(x)
        
        return x
        





def test1():
    bs = 10
    input_dim = 5
    output_dim = 10
    input = torch.sin(torch.linspace(1, input_dim, input_dim)).reshape(1, -1).repeat(bs, 1)
    gt = 4 * input.repeat(1, 2) ** 2 - 1
    epochs = 10

    net = Net1(input_dim, output_dim)
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.constant_(m.weight, 1.0)
            torch.nn.init.constant_(m.bias, 1.0)

    output = net(input)
    loss = torch.mean((output - gt) ** 2)
    loss.backward()

    print('正确的方法')
    for name, params in net.named_parameters():
        a = params.grad
        b = params
        print(name)
        print(a)
        print(b)


def test2():
    bs = 10
    input_dim = 5
    output_dim = 10
    input = torch.sin(torch.linspace(1, input_dim, input_dim)).reshape(1, -1).repeat(bs, 1)
    gt = 4 * input.repeat(1, 2) ** 2 - 1
    epochs = 10

    net = Net2(input_dim, output_dim)
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.constant_(m.weight, 1.0)
            torch.nn.init.constant_(m.bias, 1.0)

    output = net.run(input)
    loss = torch.mean((output - gt) ** 2)
    loss.backward()

    print('run的方法')
    for name, params in net.named_parameters():
        a = params.grad
        b = params
        print(name)
        print(a)
        print(b)


def test3():
    bs = 10
    input_dim = 5
    output_dim = 10
    input = torch.sin(torch.linspace(1, input_dim, input_dim)).reshape(1, -1).repeat(bs, 1)
    gt = 4 * input.repeat(1, 2) ** 2 - 1
    epochs = 10

    model_list = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU()), nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())])

    net = Net3(model_list)
    for m in model_list.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.constant_(m.weight, 1.0)
            torch.nn.init.constant_(m.bias, 1.0)

    output = net.run(input)
    loss = torch.mean((output - gt) ** 2)
    loss.backward()

    print('run的方法')
    for name, params in model_list.named_parameters():
        a = params.grad
        b = params
        print(name)
        print(a)
        print(b)






if __name__ == '__main__':
    # test1()
    # test2()
    test3()

