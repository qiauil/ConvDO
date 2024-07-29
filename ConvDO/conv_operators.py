from .helpers import *
from .boundaries import *
from .domain import *
from .schemes import *
import math
from typing import Optional

class ScalarField(CommutativeValue):
    r"""ScalarField is a class for scalar fields.

    Args:
        value (Optional[torch.Tensor], optional): The value of the scalar field. Defaults to None.
            It can be changed by calling the `register_value` method.
            The shape of the tensor should be (1,1,H,W) or (H,W).
        domain (Optional[Domain], optional): The domain of the scalar field. Defaults to UnconstrainedDomain().
    """
    
    def __init__(self,value:Optional[torch.Tensor]=None,
                 domain:Optional[Domain]=UnconstrainedDomain()) -> None:
        self.value=None
        if value is not None:
            self.register_value(value)
        self.domain=domain
        
    def register_value(self,value: torch.Tensor):
        r"""Register the value of the scalar field.
        
        Args:
            value (torch.Tensor): The value of the scalar field.
        """
    
        if len(value.shape)==2:
            value=value.unsqueeze(0).unsqueeze(0)
        self.value=value
    
    def __add__(self, other):
        if isinstance(other,ScalarField):
            return ScalarField(self.value+other.value,self.domain+other.domain)
        else:
            return ScalarField(self.value+other,self.domain+other)

    def __mul__(self, other):
        if isinstance(other,ScalarField):
            return ScalarField(self.value*other.value,self.domain*other.domain)
        else:
            return ScalarField(self.value*other,self.domain*other)   

    def __pow__(self, other):
        return ScalarField(self.value**other,self.domain**other)
    
    def __truediv__(self, other):
        if isinstance(other,ScalarField):
            return ScalarField(self.value/other.value,self.domain/other.domain)
        else:
            try:
                return ScalarField(self.value/other,self.domain/other)
            except:
                return NotImplemented
    
    def __rtruediv__(self, other):
        if isinstance(other,ScalarField):
            return ScalarField(other.value/self.value,other.domain/self.domain)
        else:
            try:
                return ScalarField(other/self.value,other/self.domain)
            except:
                return NotImplemented
        
class ConvOperator():
    def __init__(self, scheme, direction="x", derivative=1, device="cpu", dtype=torch.float32) -> None:
        self.direction = direction
        if direction == "x":
            self.kernel = scheme.kernel_dx.to(device=device, dtype=dtype)
        elif direction == "y":
            self.kernel = scheme.kernel_dy.to(device=device, dtype=dtype)
        self.pad = scheme.pad
        self.derivative = derivative
        if scheme.kernel_weights.shape[0] > 3:
            self.high_order = True
        else:
            self.high_order = False
    
    def allow_highorder (self, domain):
        if self.direction == "x":
            return isinstance(domain.left_boundary, PeriodicBoundary) 
        else:
            return isinstance(domain.top_boundary, PeriodicBoundary)
        #c_2= domain.obstacles == []
        #return c_1 and c_2

    def __mul__(self, other):
        if isinstance(other, ScalarField):
            if (not self.high_order) or (self.high_order and self.allow_highorder(other.domain)):
                domain = other.domain
                scalar_field = other.value
                if self.direction == "x":
                    delta = math.pow(other.domain.delta_x, self.derivative)
                else:
                    delta = math.pow(other.domain.delta_y, self.derivative)
                if (self.direction == "x" and isinstance(domain.left_boundary, PeriodicBoundary)) or (self.direction == "y" and isinstance(domain.top_boundary, PeriodicBoundary)):
                    # periodic boundary
                    paded = F.pad(scalar_field, (self.pad, self.pad,
                                self.pad, self.pad), mode="circular")
                else:
                    # non-periodic boundary
                    if self.direction == "x":
                        paded = F.pad(scalar_field, (self.pad, self.pad,
                                0,0), mode="constant", value=0)
                        paded=other.domain.left_boundary.correct_left(paded,scalar_field,delta)
                        paded=other.domain.right_boundary.correct_right(paded,scalar_field,delta)
                        # the following pad is due to the shape issue if Neumann boundary. 
                        # TODO: modify neumann boundary to avoid this padding
                        paded = F.pad(paded, (0, 0,
                                self.pad,self.pad), mode="constant", value=0)
                    else:
                        paded = F.pad(scalar_field, (0, 0,
                                self.pad,self.pad), mode="constant", value=0)
                        paded=other.domain.top_boundary.correct_top(paded,scalar_field,delta)
                        paded=other.domain.bottom_boundary.correct_bottom(paded,scalar_field,delta)
                        paded = F.pad(paded, (self.pad, self.pad,
                                0,0), mode="constant", value=0)
                for obstacle in domain.obstacles:
                    if self.direction == "x":
                        paded = obstacle.correct_left(paded, scalar_field, delta)
                        paded = obstacle.correct_right(paded, scalar_field, delta)
                    else:
                        paded = obstacle.correct_top(paded, scalar_field, delta)
                        paded = obstacle.correct_bottom(paded, scalar_field, delta)
                operated = F.conv2d(paded, self.kernel/delta, padding=0)
                if not self.high_order:
                    for obstacle in domain.obstacles:
                        operated = obstacle.fill_internal_field(operated)
                return ScalarField(
                    operated,
                    Domain(
                        boundaries=[
                            PeriodicBoundary() if isinstance(boundary, PeriodicBoundary) else UnConstrainedBoundary() for boundary in [domain.left_boundary,
                                                                                                                                       domain.right_boundary,
                                                                                                                                       domain.top_boundary,
                                                                                                                                       domain.bottom_boundary]
                        ],
                        delta_x=other.domain.delta_x, delta_y=other.domain.delta_y,
                        obstacles=[])
                )
            else:
                raise ValueError(
                    "High order gradient only support PeriodicBoundary with no obstacles inside.")
        else:
            raise NotImplementedError("Operation not supported")


def ConvGrad(order: int=2, direction: str="x", device="cpu", dtype=torch.float32):
    r"""
    Gradient operator $\partial / \partial x$ or $\partial / \partial y$ for a scalar.
    
    Examples:
        ```python
        p=ScalarField(torch.rand(1,1,10,10))
        grad_x = ConvGrad(order=2, direction="x", device="cpu", dtype=torch.float32)
        grad_y = ConvGrad(order=2, direction="y", device="cpu", dtype=torch.float32)
        grad_x*p # $\partial p / \partial x$ 
        grad_y*p # $\partial p / \partial y$
        ```

    Args:
        order (int): The order of the central interpolation scheme (default is 2).
        direction (str): The direction of the gradient operator, ("x" or "y", default is "x").
        device (str, optional): The device to use for computation (default is "cpu").
        dtype (torch.dtype, optional): The data type to use for computation (default is torch.float32).

    Returns:
        ConvOperator (ConvOperator): The convolutional gradient operator.

    """
    return ConvOperator(CENTRAL_INTERPOLATION_SCHEMES[order], direction=direction, derivative=1, device=device, dtype=dtype)


def ConvGrad2(order: int=2, direction="x", device="cpu", dtype=torch.float32):
    r"""
    Second order gradient operator $\partial^2 / \partial x^2$ or $\partial^2 / \partial y^2$ for a scalar.
    
    Examples:
        ```python
        p=ScalarField(torch.rand(1,1,10,10))
        grad_x = ConvGrad2(order=2, direction="x", device="cpu", dtype=torch.float32)
        grad_y = ConvGrad2(order=2, direction="y", device="cpu", dtype=torch.float32)
        grad_x*p # $\partial^2 p / \partial x^2$ 
        grad_y*p # $\partial^2 p / \partial y^2$
        ```
        
    Args:
        order (int): The order of the central Laplacian scheme (default is 2).
        direction (str): The direction of the gradient operator, ("x" or "y", default is "x").
        device (str, optional): The device to use for computation (default is "cpu").
        dtype (torch.dtype, optional): The data type to use for computation (default is torch.float32).
    
    Returns:
        ConvOperator (ConvOperator): The convolutional gradient operator.
    """
    return ConvOperator(CENTRAL_LAPLACIAN_SCHEMES[order], direction=direction, derivative=2, device=device, dtype=dtype)


def ConvNabla(order: int, device="cpu", dtype=torch.float32):
    r"""
    $\nabla=(\partial p / \partial x,\partial p / \partial y)$ operator. 
    Can be used to compute the gradient of a scalar field or the divergence of a vector field.
    
    Examples:
    
        Gradient of a scalar field:
        ```python
        p=ScalarField(torch.rand(1,1,10,10))
        nabla = ConvNabla(order=2, device="cpu", dtype=torch.float32)
        nabla*p # $\nabla p$
        ```
        
        Divergence of a vector field:
        ```python
        u=VectorValue(ScalarField(torch.rand(1,1,10,10)),ScalarField(torch.rand(1,1,10,10)))
        nabla = ConvNabla(order=2, device="cpu", dtype=torch.float32)
        nabla@u # $\nabla \cdot u$
        ```
    Args:
        order (int): The order of the central interpolation scheme (default is 2).
        device (str, optional): The device to use for computation (default is "cpu").
        dtype (torch.dtype, optional): The data type to use for computation (default is torch.float32).
    
    Returns:
        ConvOperator (ConvOperator): The convolutional gradient operator.
    """
    return VectorValue(
        ConvOperator(CENTRAL_INTERPOLATION_SCHEMES[order], direction='x', derivative=1, device=device, dtype=dtype), 
        ConvOperator(CENTRAL_INTERPOLATION_SCHEMES[order], direction='y', derivative=1, device=device, dtype=dtype)
        )

class ConvLaplacian():
    r"""
    Laplacian operator. 
    For scalar field, it is defined as $\nabla^2 p = \partial^2 p / \partial x^2 + \partial^2 p / \partial y^2$.
    For vector field, it is defined as $\nabla \cdot (\nabla \mathbf{u}) = (\frac{\partial u_x }{ \partial x}+\frac{\partial u_x }{ \partial y},\frac{\partial u_y }{ \partial x}+\frac{\partial u_y }{ \partial y})$.
    
    Examples:
    
        Gradient of a scalar field:
        ```python
        p=ScalarField(torch.rand(1,1,10,10))
        nabla2 = ConvLaplacian(order=2, device="cpu", dtype=torch.float32)
        nabla2*p # $\nabla^2 p$
        ```
        
        Divergence of a vector field:
        ```python
        u=VectorValue(ScalarField(torch.rand(1,1,10,10)),ScalarField(torch.rand(1,1,10,10)))
        nabla2 = ConvLaplacian(order=2, device="cpu", dtype=torch.float32)
        nabla*u # $\nabla \cdot (\nabla \mathbf{u})$
        ```
        
    Args:
        order (int): The order of the central Laplacian scheme (default is 2).
        device (str, optional): The device to use for computation (default is "cpu").
        dtype (torch.dtype, optional): The data type to use for computation (default is torch.float32).
    
    Returns:
        ConvOperator (ConvOperator): The convolutional gradient operator.
    """
    def __init__(self, order: int, device="cpu", dtype=torch.float32) -> None:
        self.op_x = ConvOperator(
            CENTRAL_LAPLACIAN_SCHEMES[order], direction='x', derivative=2, device=device, dtype=dtype)
        self.op_y = ConvOperator(
            CENTRAL_LAPLACIAN_SCHEMES[order], direction='y', derivative=2, device=device, dtype=dtype)

    def __mul__(self, other):
        if isinstance(other, ScalarField):
            return self.op_x*other+self.op_y*other
        elif isinstance(other, VectorValue):
            return VectorValue(
                self.op_x*other.ux+self.op_y*other.ux,
                self.op_x*other.uy+self.op_y*other.uy
            )