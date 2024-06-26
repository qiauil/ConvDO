from .helpers import *
from .boundaries import *
from .domain import *
from .schemes import *
import math

class ScalarField(CommutativeValue):
    
    def __init__(self,value,domain=UnconstrainedDomain()) -> None:
        if isinstance(value,torch.Tensor):
            if len(value.shape)==2:
                value=value.unsqueeze(0).unsqueeze(0)
                # add batch and channel dimension
        self.value=value
        self.domain=domain
    
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
            c_1= isinstance(domain.left_boundary, PeriodicBoundary) 
        else:
            c_1= isinstance(domain.top_boundary, PeriodicBoundary)
        c_2= domain.obstacles == []
        return c_1 and c_2

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
            try:
                return self*other
            except TypeError:
                raise NotImplementedError("Operation not supported")


def HOGrad(order: int, direction, device="cpu", dtype=torch.float32):
    return ConvOperator(CENTRAL_INTERPOLATION_SCHEMES[order], direction=direction, derivative=1, device=device, dtype=dtype)


def HOGrad2(order: int, direction, device="cpu", dtype=torch.float32):
    return ConvOperator(CENTRAL_LAPLACIAN_SCHEMES[order], direction=direction, derivative=2, device=device, dtype=dtype)


def HONabla(order: int, device="cpu", dtype=torch.float32):
    return VectorValue(
        ConvOperator(CENTRAL_INTERPOLATION_SCHEMES[order], direction='x', derivative=1, device=device, dtype=dtype), 
        ConvOperator(CENTRAL_INTERPOLATION_SCHEMES[order], direction='y', derivative=1, device=device, dtype=dtype)
        )

class HOLaplacian():
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
