# usr/bin/python3
# -*- coding: UTF-8 -*-

import torch
import torch.nn.functional as F
from . import *
from .domain import *
from .obstacles import *
from .boundaries import *
from .operator import *
import math


class FDScheme():

    def __init__(self, kernel) -> None:
        if not torch.is_tensor(kernel):
            kernel = torch.tensor(kernel)
        self.kernel_dx, self.kernel_dy = self.gen_kernel(kernel)
        self.pad = int((kernel.shape[0]-1)/2)

    def gen_kernel(self, kernel: torch.Tensor):
        len_kernel = len(kernel)
        dx = torch.zeros((len_kernel, len_kernel))
        dx[int((len_kernel-1)/2)] = kernel
        return dx.unsqueeze(0).unsqueeze(0), torch.flip(dx.T, dims=(0,)).unsqueeze(0).unsqueeze(0)


CENTRAL_INTERPOLATION_SCHEMES = {
    2: FDScheme([-1/2, 0, 1/2]),
    4: FDScheme([1/12, -2/3, 0, 2/3, -1/12]),
    6: FDScheme([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]),
    8: FDScheme([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]),
}

CENTRAL_LAPLACIAN_SCHEMES = {
    2: FDScheme([1, -2, 1]),
    4: FDScheme([-1/12, 4/3, -5/2, 4/3, -1/12]),
    6: FDScheme([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]),
    8: FDScheme([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]),
}


class ConvOperator():
    def __init__(self, scheme, direction="x", derivative=1, device="cpu", dtype=torch.float32) -> None:
        self.direction = direction
        if direction == "x":
            self.kernel = scheme.kernel_dx.to(device=device, dtype=dtype)
        elif direction == "y":
            self.kernel = scheme.kernel_dy.to(device=device, dtype=dtype)
        self.pad = scheme.pad
        self.derivative = derivative

    def __mul__(self, other):
        if isinstance(other, ScalarField):
            if isinstance(other.domain.left_boundary, PeriodicBoundary) and isinstance(other.domain.right_boundary, PeriodicBoundary) and other.domain.obstacles == []:
                if self.direction == "x":
                    delta = math.pow(other.domain.delta_x, self.derivative)
                else:
                    delta = math.pow(other.domain.delta_y, self.derivative)
                domain = other.domain
                scalar_field = other.value
                paded = F.pad(scalar_field, (self.pad, self.pad,
                              self.pad, self.pad), mode="circular")
                operated = F.conv2d(paded, self.kernel/delta, padding=0)
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
                ValueError(
                    "High order gradient only support PeriodicBoundary with no obstacles inside.")
        else:
            try:
                return self*other
            except TypeError:
                return NotImplemented


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


class FieldOperations():

    def __init__(self, order,device="cpu", dtype=torch.float32) -> None:
        self.nabla = HONabla(order, device=device, dtype=dtype)
        self.nabla2 = HOLaplacian(order, device=device, dtype=dtype)
        self.grad_x = HOGrad(order, direction='x', device=device, dtype=dtype)
        self.grad_y = HOGrad(order, direction='y', device=device, dtype=dtype)


class HOResTrainsientForce(FieldOperations):

    def __init__(self, order,device="cpu", dtype=torch.float32) -> None:
        super().__init__(order, device=device, dtype=dtype)

    def __call__(self, u_0, v_0, p_0, u_1, v_1, p_1, force_x, force_y, domain_u, domain_v, domain_p, domain_force_x, domain_force_y, viscosity, dt):
        p_0 = ScalarField(p_0, domain_p)
        p_1 = ScalarField(p_1, domain_p)
        u_0 = VectorValue(ScalarField(u_0, domain_u),
                          ScalarField(v_0, domain_v))
        u_1 = VectorValue(ScalarField(u_1, domain_u),
                          ScalarField(v_1, domain_v))
        force = VectorValue(ScalarField(force_x, domain_force_x),
                            ScalarField(force_y, domain_force_y))
        u_inter = (u_1+u_0)*0.5
        transient = (u_1-u_0)*(1/dt)
        advection = u_inter@(self.nabla*u_inter)
        pressure = self.nabla*p_0
        vis = -1*viscosity*(self.nabla2*u_inter)
        ns_res = transient+advection+pressure+vis-force
        con_res = (self.nabla2*p_1)+((self.grad_x*u_1.ux)*(self.grad_x*u_1.ux)+2*(self.grad_y*u_1.ux)
                                     * (self.grad_x*u_1.uy)+(self.grad_y*u_1.uy)*(self.grad_y*u_1.uy))-self.nabla@force
        divergence = self.nabla@u_1
        return torch.cat([ns_res.ux.value, ns_res.uy.value, con_res.value, divergence.value], dim=-3)


class HOResTrainsient(FieldOperations):
    def __init__(self, order,device="cpu", dtype=torch.float32) -> None:
        super().__init__(order, device=device, dtype=dtype)

    def __call__(self, u_0, v_0, p_0, u_1, v_1, p_1, domain_u, domain_v, domain_p, viscosity, dt):
        p_0 = ScalarField(p_0, domain_p)
        p_1 = ScalarField(p_1, domain_p)
        u_0 = VectorValue(ScalarField(u_0, domain_u),
                          ScalarField(v_0, domain_v))
        u_1 = VectorValue(ScalarField(u_1, domain_u),
                          ScalarField(v_1, domain_v))
        u_inter = (u_1+u_0)*0.5
        transient = (u_1-u_0)*(1/dt)
        advection = u_inter@(self.nabla*u_inter)
        pressure = self.nabla*p_0
        vis = -1*viscosity*(self.nabla2*u_inter)
        ns_res = transient+advection+pressure+vis
        con_res = (self.nabla2*p_1)+((self.grad_x*u_1.ux)*(self.grad_x*u_1.ux)+2 *
                                     (self.grad_y*u_1.ux)*(self.grad_x*u_1.uy)+(self.grad_y*u_1.uy)*(self.grad_y*u_1.uy))
        divergence = self.nabla@u_1
        return torch.cat([ns_res.ux.value, ns_res.uy.value, con_res.value, divergence.value], dim=-3)


class HOResNSPressureVelocityForce(FieldOperations):

    def __init__(self, order,device="cpu", dtype=torch.float32) -> None:
        super().__init__(order, device=device, dtype=dtype)

    def __call__(self, u, v, p, domain_u, domain_v, domain_p, force_x, force_y, domain_force_x, domain_force_y,):
        force = VectorValue(ScalarField(force_x, domain_force_x),
                            ScalarField(force_y, domain_force_y))
        p_1 = ScalarField(p, domain_p)
        u_1 = VectorValue(ScalarField(u, domain_u), ScalarField(v, domain_v))
        poisson = (self.nabla2*p_1)+((self.grad_x*u_1.ux)*(self.grad_x*u_1.ux)+2*(self.grad_y*u_1.ux)
                                     * (self.grad_x*u_1.uy)+(self.grad_y*u_1.uy)*(self.grad_y*u_1.uy))-self.nabla@force
        divergence = self.nabla@u_1
        return torch.cat([poisson.value, divergence.value], dim=-3)


class HOResNSPressureVelocity(FieldOperations):

    def __init__(self, order,device="cpu", dtype=torch.float32) -> None:
        super().__init__(order, device=device, dtype=dtype)

    def __call__(self, u, v, p, domain_u, domain_v, domain_p):
        p_1 = ScalarField(p, domain_p)
        u_1 = VectorValue(ScalarField(u, domain_u), ScalarField(v, domain_v))
        poisson = (self.nabla2*p_1)+((self.grad_x*u_1.ux)*(self.grad_x*u_1.ux)+2 *
                                     (self.grad_y*u_1.ux)*(self.grad_x*u_1.uy)+(self.grad_y*u_1.uy)*(self.grad_y*u_1.uy))
        divergence = self.nabla@u_1
        return torch.cat([poisson.value, divergence.value], dim=-3)
