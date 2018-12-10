---
layout: post
title: "PyTorch Errors Series: AssertionError: nn criterions don't compute the gradient w.r.t. targets"
categories:
  - "PyTorch Errors Series"
not_main: true
hidden: true
tags:
  - pytorch
  - error
  - linear regression
last_modified_at: 2018-10-30T18:29:52+05:30
---

Let's write a Linear Regression using `PyTorch`. We will train it on some dummy data.

```python
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim


# Preparing data for regression
x = Variable(torch.Tensor([[-1. ], [-0.9], [-0.8], [-0.7], [-0.6], [-0.5], 
                           [-0.4], [-0.3], [-0.2], [-0.1], [ 0. ], [ 0.1], 
                           [ 0.2], [ 0.3], [ 0.4], [ 0.5], [ 0.6], [ 0.7], 
                           [ 0.8], [ 0.9], [ 1. ]]), requires_grad=False)
y_true = Variable(torch.Tensor([[32.6 ], [31.62], [30.64], [29.66], [28.68], 
                                [27.7 ], [26.72], [25.74], [24.76], [23.78],
                                [22.8 ], [21.82], [20.84], [19.86], [18.88], 
                                [17.9 ], [16.92], [15.94], [14.96], [13.98],
                            [13.  ]]), requires_grad=False)

# Creating a Linear Regression Module
# which just involves multiplying a weight
# with the feature value and adding a bias term
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        # Since there is only a single feature value,
        # and the output is a single value, we use
        # the Linear module with dimensions 1 X 1.
        # It adds a bias term by default
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


linreg = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(linreg.parameters(), lr=0.1)

for i in range(10):
    y_pred = linreg(x)
    loss = criterion(y_true, y_pred)
    print('Loss:', loss.data, 'Parameters:', 
          list(map(lambda x: x.data, linreg.parameters())))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

When we run this code, we get the follwing stack trace:
```python
Traceback (most recent call last):
  File "/media/saqib/ni/Projects/PyTorch_Practice/LinearRegression-01.py", line 41, in <module>
    loss = criterion(y_true, y_pred)
  File "/usr/local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 491, in __call__
    result = self.forward(*input, **kwargs)
  File "/usr/local/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 371, in forward
    _assert_no_grad(target)
  File "/usr/local/lib/python3.6/site-packages/torch/nn/modules/loss.py", line 12, in _assert_no_grad
    "nn criterions don't compute the gradient w.r.t. targets - please " \
AssertionError: nn criterions don't compute the gradient w.r.t. targets - please mark these tensors as not requiring gradients
```
So, it seems that the error is in line 41. Hmm, but we did say `requires_grad=False` when we declared `y_true`. So, why did the error occur? Well, let's try swapping `y_true` and `y_pred`.

```python
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim


# Preparing data for regression
x = Variable(torch.Tensor([[-1. ], [-0.9], [-0.8], [-0.7], [-0.6], [-0.5],
                           [-0.4], [-0.3], [-0.2], [-0.1], [ 0. ], [ 0.1],
                           [ 0.2], [ 0.3], [ 0.4], [ 0.5], [ 0.6], [ 0.7],
                           [ 0.8], [ 0.9], [ 1. ]]), requires_grad=False)
y_true = Variable(torch.Tensor([[32.6 ], [31.62], [30.64], [29.66], [28.68],
                                [27.7 ], [26.72], [25.74], [24.76], [23.78],
                                [22.8 ], [21.82], [20.84], [19.86], [18.88],
                                [17.9 ], [16.92], [15.94], [14.96], [13.98],
                            [13.  ]]), requires_grad=False)

# Creating a Linear Regression Module
# which just involves multiplying a weight
# with the feature value and adding a bias term
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        # Since there is only a single feature value,
        # and the output is a single value, we use
        # the Linear module with dimensions 1 X 1.
        # It adds a bias term by default
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


linreg = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(linreg.parameters(), lr=0.1)

for i in range(10):
    y_pred = linreg(x)
    loss = criterion(y_pred, y_true)
    print('Loss:', loss.data, 'Parameters:',
          list(map(lambda x: x.data, linreg.parameters())))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Output:
```python
Loss: tensor(543.1290) Parameters: [tensor([[ 0.9388]]), tensor([ 0.4204])]
Loss: tensor(356.8506) Parameters: [tensor([[ 0.1513]]), tensor([ 4.8964])]
Loss: tensor(236.3258) Parameters: [tensor([[-0.5785]]), tensor([ 8.4771])]
Loss: tensor(158.0679) Parameters: [tensor([[-1.2547]]), tensor([ 11.3417])]
Loss: tensor(107.0193) Parameters: [tensor([[-1.8814]]), tensor([ 13.6333])]
Loss: tensor(73.5209) Parameters: [tensor([[-2.4621]]), tensor([ 15.4667])]
Loss: tensor(51.3714) Parameters: [tensor([[-3.0002]]), tensor([ 16.9333])]
Loss: tensor(36.5856) Parameters: [tensor([[-3.4989]]), tensor([ 18.1067])]
Loss: tensor(26.5988) Parameters: [tensor([[-3.9609]]), tensor([ 19.0453])]
Loss: tensor(19.7574) Parameters: [tensor([[-4.3891]]), tensor([ 19.7963])]
```
It works! It seems that the second argument in criterion expects a `Variable` with `requires_grad=False`. Since the `Variable` computed by by our model, `linreg` requires gradients for backpropagation, passing it as a second argument to our `MSELoss` criterion caused the error. 


