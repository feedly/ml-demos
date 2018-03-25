import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import time

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden = nn.Linear(784, 50)
        self.final = nn.Linear(50, 10)

    def forward(self, features):
        x = self.hidden(features.float().view(len(features), -1))
        x = self.final(x)
        return F.log_softmax(x, dim=1)

def fun_with_gpus():
    t1 = torch.cuda.FloatTensor(20,20)
    t2 = torch.cuda.FloatTensor(20,20)
    t3 = t1.matmul(t2)
    print(f"What is t3? Well it's a {type(t3)}")

def this_wont_work_dummy(features, labels):
    dl = DataLoader(list(zip(features, labels)), batch_size=5)
    model = Model()
    model.hidden.cuda()

    batch = next(iter(dl))
    batch = [torch.autograd.Variable(b) for b in batch]
    return model.forward(*batch[:-1])


def view_number(data:torch.FloatTensor, title:str):
    import matplotlib.pyplot as plt
    plt.imshow(data.numpy())
    plt.title(title)
    plt.show()

def data_shipping_experiment(n:int):
    #let's run all on the CPU
    array1 = np.random.randn(200,200)
    array2 = np.random.randn(200,200)
    t0 = time.time()
    for i in range(n):
        array3 = array1.matmul(array2)
        array1 = array3
    t1 = time.time()

    print(f'CPU only operations took {t1-t0}')


    #let's run all on the GPU
    tensor1 = torch.cuda.FloatTensor(200, 200)
    tensor2 = torch.cuda.FloatTensor(200, 200)

    t0 = time.time()
    for i in range(n):
        tensor3 = tensor1.matmul(tensor2)
        del tensor1
        tensor1 = tensor3
    t1 = time.time()

    print(f'GPU only operations took {t1-t0}')

    #let's ship data like a mofo
    tensor1 = torch.FloatTensor(200, 200)
    tensor2 = torch.FloatTensor(200, 200)

    t0 = time.time()
    for i in range(n):
        ctensor1 = tensor1.cuda()
        ctensor2 = tensor2.cuda()
        ctensor3 = ctensor1.matmul(ctensor2)
        tensor1 = ctensor3.cpu()

        del ctensor1
        del ctensor2
        del ctensor3

    t1 = time.time()

    print(f'data shipping took {t1-t0}')

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise ValueError('a GPU is required for these examples')

    _data = datasets.MNIST('/tmp/data', train=True, download=True)

    # if you want to look at some images...
    # view_number(_data.train_data[10], str(_data.train_labels[10]))

    data_shipping_experiment(100000)
    this_wont_work_dummy(_data.train_data, _data.train_labels)