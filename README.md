<center>
<img src="README_files/ConvDO.png"/>
</center>

## About

ConvDO is a Convolutional Differential Operator designed for Physics-based deep learning study.
ConvDO is PyTorch-based and only supports 2D fields at the moment.

## A Quick How To

Here is a simple case illustrating how to use the `ConvDo` package to calculate the residual of a physical field.

[Run in colab](https://colab.research.google.com/github/qiauil/ConvDO/blob/main/README.ipynb)


### Kolmogorov flow
Kolmogorov flow is a classical turbulent flow; a snapshot of the velocity is illustrated as follows:


```python
import torch
import matplotlib.pyplot as plt

flow= torch.load('./kf_flow.pt')
channel_names=["u","v"]

figs, axs = plt.subplots(1, 2)
for i, ax in enumerate(axs):
    ax.imshow(flow[i].numpy())
    ax.set_title(channel_names[i])
plt.show()
```


![png](README_files/README_2_1.png)
    


The flow field satisfies the divergence free condition, which is

$$
\nabla \cdot \mathbf{u}=\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}=0
$$

To calculate the divergence of the field, we can simply build a `Divergence` operator:


```python
from ConvDO import *

class Divergence(FieldOperations):
    
    def __init__(self, order, device="cpu", dtype=torch.float32) -> None:
        super().__init__(order, device, dtype)
        
    def __call__(self, u:torch.Tensor,v:torch.Tensor,domain_u:Domain,domain_v:Domain):
        velocity=VectorValue(ScalarField(u,domain_u),ScalarField(v,domain_v))
        return self.nabla@velocity
        
```

The `order` here refers to the order of the finite difference scheme used to calculate the derivate. For periodic boundary conditions, supported orders include `2,4,6,8`. For un-periodic boundaries, only a second-order scheme is available.

The `device` and `dtype` should be inconsistent with the tensor used for the operation.

The `domain_u` and `domain_v` should be instances of the `Domain` class, which gives the boundary condition of the fields.

We use the `VectorValue` class to generate the velocity; the component of the vector is two `ScalarField` initialized with the velocity tensor and the corresponding domain. The `VectorValue` class also supports numerical numbers as its component, and we can perform some simple operations on the `VectorValue`: 


```python
a=VectorValue(1,2)
b=VectorValue(3,4)
c=a+b
d=a-b
e=5*a
f=a@b
print(c,d,e,f)
```

    (4,6) (-2,-2) (5,10) 11


`self.nabla` is the nabla operator, $\nabla$. Other supported operator includes the Laplacian operator `self.nabla2`= $\nabla^2$ and gradient operators `self.grad_x`=$\partial/\partial x$ and `self.grad_y`=$\partial/\partial y$. For instance, the `Divergence` operator can also be written as


```python
class Divergence_new(FieldOperations):
    
    def __init__(self, order, device="cpu", dtype=torch.float32) -> None:
        super().__init__(order, device, dtype)
        
    def __call__(self, u:torch.Tensor,v:torch.Tensor,domain_u:Domain,domain_v:Domain):
        return self.grad_x*ScalarField(u,domain_u)+self.grad_y*ScalarField(v,domain_v)
```

If you want to use these operators out of the `FieldOperations` class, you can also simply create them as follows:


```python
nabla = HONabla(order=2)
nabla2 = HOLaplacian(order=2)
grad_x = HOGrad(order=2,direction="x")
grad_y = HOGrad(order=2,direction="y")
```

Then, we can set up the computational domain of the field and calculate the divergence. The boundary condition of Kolmogorov flow is periodic thus we will set up a periodic domain. Here, `delta_x` and `delta_y` is the physical distance between mesh cells in the field. Besides `PeriodicDomain`, you can directly use the `Domain` class to add different boundary conditions and even obstacles inside the field. 

Now, let's calculate the divergence!


```python
u=flow[0]
v=flow[1]
divergence=Divergence(order=6)
domain=PeriodicDomain(delta_x=2*3.14/64,delta_y=2*3.14/64)
residual=divergence(u,v,domain,domain).value.numpy().squeeze().squeeze()
plt.imshow(residual)
plt.colorbar()
plt.show()
```


​    
![png](README_files/README_13_1.png)
​    


Since the operation is based on the convolution provided by PyTorch, it supports batched input and output. If your input of a `ScalarField` only contains 2 dimensions $H \times W$, it will automatically convert to a new tensor with the shape of $B\times C \times H \times W$. The shape of the output is always $B\times C \times H \times W$.

Also, thanks to PyTorch, all the operation is differentiable. For example, we can directly use gradient descent method to make the flow field more divergence free.


```python
losses=[]
optimizer = torch.optim.Adam([u,v],lr=0.001)
u.requires_grad=True
v.requires_grad=True
for i in range(100):
    optimizer.zero_grad()
    residual=divergence(u,v,domain,domain).value
    loss=torch.mean(residual**2)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    
plt.plot(losses)
plt.show()
plt.imshow(residual.detach().numpy().squeeze().squeeze())
plt.colorbar()
plt.title("Final residual")
plt.show()
fig, axs = plt.subplots(1, 2)
for i, ax in enumerate(axs):
    ax.imshow(flow[i].detach().numpy())
    ax.set_title(channel_names[i])
plt.show()
```


​    
![png](README_files/README_15_0.png)
​    




![png](README_files/README_15_1.png)
    




![png](README_files/README_15_2.png)
    

