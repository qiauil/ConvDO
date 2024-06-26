from .helpers import *
from .schemes import *
from .domain import *
from .conv_operators import *

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