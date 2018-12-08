---
layout: post
title: "PyTorch Errors Series: ValueError: optimizer got an empty parameter list"
categories:
  - "PyTorch Errors Series"
not_main: true
hidden: true
tags:
  - pytorch
  - error
  - multilayer perceptron
  - object oriented
author: "Saqib Shamsi"
last_modified_at: 2018-11-07T15:56:27+05:30
---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" async>
</script>

We are going to write a flexible fully connected network, also called a dense network. We will use it to solve XOR. That is, its input will be four boolean values and the output will be their XOR.

Our network will consist of repeated sequences of a fully connected (linear) layer followed by a pointwise non-linear operation (aka activation function). However, rather than fixing the number of layers in our model's class defintion, we will provide it with arguments to let our class know how many layers to create. 

Also, to avoid writing duplicate code, we will create a unit (a `torch` Module) of a linear layer followed by an activation layer. This would be our basic Lego block. We will combine these Lego blocks as per our need, to create a network of desired width (number of neurons in each layer) and depth (number of layers). The example would also demonstrate the ease with which one can create modular structures in an Object Oriented fashion using `PyTorch`. 

Let's get started.

```python
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable


xs = np.array([[0., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 1., 0.], [0., 0., 1., 1.],
              [0., 1., 0., 0.], [0., 1., 0., 1.], [0., 1., 1., 0.], [0., 1., 1., 1.],
              [1., 0., 0., 0.], [1., 0., 0., 1.], [1., 0., 1., 0.], [1., 0., 1., 1.],
              [1., 1., 0., 0.], [1., 1., 0., 1.], [1., 1., 1., 0.], [1., 1., 1., 1.]], dtype=float)

ys = np.array([[0.], [1.], [1.], [0.], [1.], [0.], [0.], [1.], [1.], [0.], [0.], [1.],
               [0.], [1.], [1.], [0.]], dtype=float)

x_var = Variable(Tensor(xs), requires_grad=False)
y_var = Variable(Tensor(ys), requires_grad=False)

EPOCHS = 1000


# Helper function to train the network
def train(model, x_train, y_train, criterion, optimizer, epochs):

    for i in range(epochs):
        # Make predictions (forward propagation)
        y_pred = model(x_train)

        # Compute and print the loss every hundred epochs
        loss = criterion(y_pred, y_train)
        if i % 100 == 0:
            print('Loss:', loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Create a reusable module
# PyTorch makes writing modular OO code extremely easy
class LinearBlock(nn.Module):

    def __init__(self, in_nums, out_nums, activation):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_nums, out_nums)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.linear(x))


class FullyConnectedNet(nn.Module):

    def __init__(self, input_size, neurons, activations):
        super(FullyConnectedNet, self).__init__()

        # For now, we will have a linear layer followed by an activation function
        assert len(neurons) == len(activations), 'Number of activations must be equal to the number of activations'

        # We will need a list of blocks cascaded one after the other, so we keep them in a list
        self.blocks = list()

        previous = input_size
        for i in range(len(neurons)):
            self.blocks.append(LinearBlock(previous, neurons[i], activations[i]))
            previous = neurons[i]

    def forward(self, x):
        "Pass the input through each block"
        for block in self.blocks:
            x = block(x)

        return x


# Crete a network with 2 hidden layers and 1 output layer, with sigmoid activations
fcnet01 = FullyConnectedNet(4, # We have a four dimensional input
                            [4, 4, 1], # We two hidden layers with 4 neurons each, and an output layer
                                       # with 1 neuron
                            [nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid()] # Using sigmoid for activation
                            )

# Since it's a 0-1 problem, we will use Binary Cross Entropy as our loss function
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(fcnet01.parameters(), lr=0.01)

# Then, our usual training loop
train(fcnet01, x_var, y_var, criterion, optimizer, EPOCHS)
```

When we run the code above, we run into the error below:

```python
Traceback (most recent call last):
  File "/media/saqib/ni/Projects/PyTorch_Practice/pytorch-errors-series/MultiLayerNet.py", line 87, in <module>
    optimizer = torch.optim.SGD(fcnet01.parameters(), lr=0.01)
  File "/usr/local/lib/python3.6/site-packages/torch/optim/sgd.py", line 64, in __init__
    super(SGD, self).__init__(params, defaults)
  File "/usr/local/lib/python3.6/site-packages/torch/optim/optimizer.py", line 38, in __init__
    raise ValueError("optimizer got an empty parameter list")
ValueError: optimizer got an empty parameter list
```

What happend here? We certainly created those blocks, so our network must have parameters. So, why does the optimizer say that there are none? Well, it's because we put them in a Python list. In order for those blocks to be detected, we need to use `torch.nn.ModuleList`. It's a container provided by `PyTorch`, which acts just like a Python list would. However, the modules put inside it would become a part of the model, and their parameters can be optimized.

So, let's make the fix and run the code, shall we?


```python
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable


xs = np.array([[0., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 1., 0.], [0., 0., 1., 1.],
              [0., 1., 0., 0.], [0., 1., 0., 1.], [0., 1., 1., 0.], [0., 1., 1., 1.],
              [1., 0., 0., 0.], [1., 0., 0., 1.], [1., 0., 1., 0.], [1., 0., 1., 1.],
              [1., 1., 0., 0.], [1., 1., 0., 1.], [1., 1., 1., 0.], [1., 1., 1., 1.]], dtype=float)

ys = np.array([[0.], [1.], [1.], [0.], [1.], [0.], [0.], [1.], [1.], [0.], [0.], [1.],
               [0.], [1.], [1.], [0.]], dtype=float)

x_var = Variable(Tensor(xs), requires_grad=False)
y_var = Variable(Tensor(ys), requires_grad=False)

EPOCHS = 1000


# Helper function to train the network
def train(model, x_train, y_train, criterion, optimizer, epochs):

    for i in range(epochs):
        # Make predictions (forward propagation)
        y_pred = model(x_train)

        # Compute and print the loss every hundred epochs
        loss = criterion(y_pred, y_train)
        if i % 100 == 0:
            print('Loss:', loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Create a reusable module
# PyTorch makes writing modular OO code extremely easy
class LinearBlock(nn.Module):

    def __init__(self, in_nums, out_nums, activation):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_nums, out_nums)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.linear(x))


class FullyConnectedNet(nn.Module):

    def __init__(self, input_size, neurons, activations):
        super(FullyConnectedNet, self).__init__()

        # For now, we will have a linear layer followed by an activation function
        assert len(neurons) == len(activations), 'Number of activations must be equal to the number of activations'

        # We will need a list of blocks cascaded one after the other, so we keep them in a ModuleList instead of a Python list
        self.blocks = nn.ModuleList()

        previous = input_size
        for i in range(len(neurons)):
            self.blocks.append(LinearBlock(previous, neurons[i], activations[i]))
            previous = neurons[i]

    def forward(self, x):
        "Pass the input through each block"
        for block in self.blocks:
            x = block(x)

        return x


# Crete a network with 2 hidden layers and 1 output layer, with sigmoid activations
fcnet01 = FullyConnectedNet(4, # We have a four dimensional input
                            [4, 4, 1], # We two hidden layers with 4 neurons each, and an output layer
                                       # with 1 neuron
                            [nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid()] # Using sigmoid for activation
                            )

# Since it's a 0-1 problem, we will use Binary Cross Entropy as our loss function
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(fcnet01.parameters(), lr=0.01)

# Then, our usual training loop
train(fcnet01, x_var, y_var, criterion, optimizer, EPOCHS)
```

We changed the `list()` initializer on line 62 to `nn.ModuleList()` and things started working:
```python
Loss: tensor(0.7260, grad_fn=<BinaryCrossEntropyBackward>)
Loss: tensor(0.7048, grad_fn=<BinaryCrossEntropyBackward>)
Loss: tensor(0.6972, grad_fn=<BinaryCrossEntropyBackward>)
Loss: tensor(0.6946, grad_fn=<BinaryCrossEntropyBackward>)
Loss: tensor(0.6936, grad_fn=<BinaryCrossEntropyBackward>)
Loss: tensor(0.6933, grad_fn=<BinaryCrossEntropyBackward>)
Loss: tensor(0.6932, grad_fn=<BinaryCrossEntropyBackward>)
Loss: tensor(0.6932, grad_fn=<BinaryCrossEntropyBackward>)
Loss: tensor(0.6932, grad_fn=<BinaryCrossEntropyBackward>)
Loss: tensor(0.6932, grad_fn=<BinaryCrossEntropyBackward>)
```
Now, as we can see above, the loss doesn't seem to go down very much even after training for 1000 epochs. We can train it for more epochs, but there are loads of others things we can try out as well. We can experiment with networks of different widths and depths. We can try different activations as well. The good news is that since we took the time to write our model in the above manner, creating various architectures is now a breeze.

Want a four layered deep network with more width? Here you go:
```python
fcnet02 = FullyConnectedNet(4, [8, 10, 10, 8, 1],
                            [nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid(), 
                             nn.Sigmoid(), nn.Sigmoid()])
```

Want ReLUs instead? Easy.
```python
fcnet03 = FullyConnectedNet(4, [8, 10, 10, 8, 1], 
                               [nn.ReLU(), nn.ReLU(), nn.ReLU(), 
                                nn.ReLU(), nn.ReLU()])
```

"But I want a sigmoid for the output layer," I hear you saying.
```python
fcnet04 = FullyConnectedNet(4, [8, 10, 10, 8, 1], 
                               [nn.ReLU(), nn.ReLU(), nn.ReLU(), 
                                nn.ReLU(), nn.Sigmoid()])
```

Before we end, I would like to mention that we can get rid the ugly `for` loop on line 71 too. Since all we are doing is feeding the output of one block to another is a sequential manner, we can put our blocks in `PyTorch`'s `torch.nn.Sequential` container. I will cover it in a post later.


