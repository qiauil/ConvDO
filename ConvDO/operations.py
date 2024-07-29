from .helpers import *
from .schemes import *
from .domain import *
from .conv_operators import *

class FieldOperations():
    r"""
    A class that performs various operations on fields.

    Args:
        order (int): The order of the operations.
        device (str, optional): The device to perform the operations on. Defaults to "cpu".
        dtype (torch.dtype, optional): The data type of the operations. Defaults to torch.float32.
        
    Attributes:
        nabla (ConvNabla): The gradient operator.
        nabla2 (ConvLaplacian): The Laplacian operator.
        grad_x (ConvGrad): The gradient operator in the x direction.
        grad_y (ConvGrad): The gradient operator in the y direction.
    """

    def __init__(self, order:int, device="cpu", dtype=torch.float32) -> None:
        self.nabla = ConvNabla(order, device=device, dtype=dtype)
        self.nabla2 = ConvLaplacian(order, device=device, dtype=dtype)
        self.grad_x = ConvGrad(order, direction='x', device=device, dtype=dtype)
        self.grad_y = ConvGrad(order, direction='y', device=device, dtype=dtype)
        self.grad2_x = ConvGrad2(order, direction='x', device=device, dtype=dtype)
        self.grad2_y = ConvGrad2(order, direction='y', device=device, dtype=dtype)


class TransientNSWithForce(FieldOperations):
    r"""
    Class representing the transient Navier-Stokes equations with external force.
        Returns a three channel tensor containing the residual of momentum equation in x and y directions and the divergence of the velocity field.   
        Defining $\mathbf{u}_{int}=(\mathbf{u}_0+\mathbf{u}_1)/2=((u_{x,0}+u_{x,1})/2,(u_{y,0}+u_{y,1})/2)$, the residual is given by:
        $\begin{matrix}\frac{\mathbf{u}_1-\mathbf{u}_0}{\Delta t}+\mathbf{u}_{int}\cdot\nabla\mathbf{u}_{int}+\nabla\left(\frac{p_0+p_1}{2}\right)-\nu\nabla^2\mathbf{u}_{int}-\mathbf{f}\\(\nabla\cdot\mathbf{u}_{int}+\nabla\cdot\mathbf{u}_1)/2\end{matrix}$


    Args:
        domain_u (Domain): The domain for the x-velocity component.
        domain_v (Domain): The domain for the y-velocity component.
        domain_p (Domain): The domain for the pressure component.
        force_x (torch.Tensor): The x-component of the external force.
        force_y (torch.Tensor): The y-component of the external force.
        domain_force_x (Domain): The domain for the x-component of the external force.
        domain_force_y (Domain): The domain for the y-component of the external force.
        viscosity (float): The viscosity coefficient.
        dt (float): The time step size.
        order (int): The order of accuracy for the finite difference scheme.
        device (str, optional): The device to use for computation (default: "cpu").
        dtype (torch.dtype, optional): The data type to use for computation (default: torch.float32).
    """

    def __init__(self, 
                 domain_u: Domain,
                 domain_v: Domain, 
                 domain_p: Domain, 
                 force_x: torch.Tensor,
                 force_y: torch.Tensor,
                 domain_force_x: Domain, 
                 domain_force_y: Domain, 
                 viscosity: float, 
                 dt: float,
                 order:int,
                 device="cpu", 
                 dtype=torch.float32,
                 ) -> None:
        super().__init__(order, device=device, dtype=dtype)
        self.p_0 = ScalarField(domain=domain_p)
        self.p_1 = ScalarField(domain=domain_p)
        self.velocity_0 = VectorValue(ScalarField(domain=domain_u), ScalarField(domain=domain_v))
        self.velocity_1 = VectorValue(ScalarField(domain=domain_u), ScalarField(domain=domain_v))
        self.force = VectorValue(ScalarField(force_x, domain=domain_force_x),
                                 ScalarField(force_y, domain=domain_force_y))
        self.viscosity = viscosity
        self.dt = dt
        

    def __call__(self, u_0, v_0, p_0, u_1, v_1, p_1):
        """
        Compute the solution of the transient Navier-Stokes equations with external force.

        Args:
            u_0 (torch.Tensor): The x-velocity component at time step t.
            v_0 (torch.Tensor): The y-velocity component at time step t.
            p_0 (torch.Tensor): The pressure component at time step t.
            u_1 (torch.Tensor): The x-velocity component at time step t+1.
            v_1 (torch.Tensor): The y-velocity component at time step t+1.
            p_1 (torch.Tensor): The pressure component at time step t+1.

        Returns:
            residual (torch.Tensor): The concatenated tensor of the x-velocity, y-velocity, and divergence components.
        """
        self.velocity_0.ux.register_value(u_0)
        self.velocity_0.uy.register_value(v_0)
        self.p_0.register_value(p_0)
        self.velocity_1.ux.register_value(u_1)
        self.velocity_1.uy.register_value(v_1)
        self.p_1.register_value(p_1)
        u_inter = (self.velocity_0 + self.velocity_1) * 0.5
        transient = (self.velocity_1 - self.velocity_0) / self.dt
        advection = u_inter @ (self.nabla * u_inter)
        pressure = self.nabla * ((self.p_0 + self.p_1) * 0.5)
        vis = -1 * self.viscosity * (self.nabla2 * u_inter)
        ns_res = transient + advection + pressure + vis - self.force
        divergence = ((self.nabla @ u_inter) + (self.nabla @ self.velocity_1)) * 0.5
        return torch.cat([ns_res.ux.value, ns_res.uy.value, divergence.value], dim=-3)


class TransientNS(FieldOperations):
    r"""
    Class representing the transient Navier-Stokes equations.
        Returns a three channel tensor containing the residual of momentum equation in x and y directions and the divergence of the velocity field.
        Defining $\mathbf{u}_{int}=(\mathbf{u}_0+\mathbf{u}_1)/2=((u_{x,0}+u_{x,1})/2,(u_{y,0}+u_{y,1})/2)$, the residual is given by:
        $\begin{matrix}\frac{\mathbf{u}_1-\mathbf{u}_0}{\Delta t}+\mathbf{u}_{int}\cdot\nabla\mathbf{u}_{int}+\nabla\left(\frac{p_0+p_1}{2}\right)-\nu\nabla^2\mathbf{u}_{int}\\(\nabla\cdot\mathbf{u}_{int}+\nabla\cdot\mathbf{u}_1)/2\end{matrix}$

    Args:
        domain_u (Domain): The domain for the x-velocity component.
        domain_v (Domain): The domain for the y-velocity component.
        domain_p (Domain): The domain for the pressure component.
        viscosity (float): The viscosity coefficient.
        dt (float): The time step size.
        order (int): The order of accuracy for the finite difference scheme.
        device (str, optional): The device to use for computation (default: "cpu").
        dtype (torch.dtype, optional): The data type to use for computation (default: torch.float32).
    """

    def __init__(self, 
                 domain_u: Domain,
                 domain_v: Domain, 
                 domain_p: Domain, 
                 viscosity: float, 
                 dt: float,
                 order:int,
                 device="cpu", 
                 dtype=torch.float32,
                 ) -> None:
        super().__init__(order, device=device, dtype=dtype)
        self.p_0 = ScalarField(domain=domain_p)
        self.p_1 = ScalarField(domain=domain_p)
        self.velocity_0 = VectorValue(ScalarField(domain=domain_u), ScalarField(domain=domain_v))
        self.velocity_1 = VectorValue(ScalarField(domain=domain_u), ScalarField(domain=domain_v))
        self.viscosity = viscosity
        self.dt = dt
        

    def __call__(self, u_0, v_0, p_0, u_1, v_1, p_1):
        """
        Compute the solution of the transient Navier-Stokes equations with external force.

        Args:
            u_0 (torch.Tensor): The x-velocity component at time step t.
            v_0 (torch.Tensor): The y-velocity component at time step t.
            p_0 (torch.Tensor): The pressure component at time step t.
            u_1 (torch.Tensor): The x-velocity component at time step t+1.
            v_1 (torch.Tensor): The y-velocity component at time step t+1.
            p_1 (torch.Tensor): The pressure component at time step t+1.

        Returns:
            residual (torch.Tensor): The concatenated tensor of the x-velocity, y-velocity, and divergence components.
        """
        self.velocity_0.ux.register_value(u_0)
        self.velocity_0.uy.register_value(v_0)
        self.p_0.register_value(p_0)
        self.velocity_1.ux.register_value(u_1)
        self.velocity_1.uy.register_value(v_1)
        self.p_1.register_value(p_1)
        u_inter = (self.velocity_0 + self.velocity_1) * 0.5
        transient = (self.velocity_1 - self.velocity_0) / self.dt
        advection = u_inter @ (self.nabla * u_inter)
        pressure = self.nabla * ((self.p_0 + self.p_1) * 0.5)
        vis = -1 * self.viscosity * (self.nabla2 * u_inter)
        ns_res = transient + advection + pressure + vis
        divergence = ((self.nabla @ u_inter) + (self.nabla @ self.velocity_1)) * 0.5
        return torch.cat([ns_res.ux.value, ns_res.uy.value, divergence.value], dim=-3)


class PoissonDivergenceWithForce(FieldOperations):
    r"""
    Class representing the Poisson equation for pressure and the divergence of the velocity field.
        Returns a two channel tensor containing the residual of the Poisson equation and the divergence of the velocity field.
        Residual of Poisson equation:
        $\left( {{\partial^2 p} \over {\partial x^2}} + {{\partial^2 p} \over {\partial y^2}} \right) = \left( {{\partial u} \over {\partial x}} \right)^2 + 2 {{\partial u} \over {\partial y}} {{\partial v} \over {\partial x}}+ \left( {{\partial v} \over {\partial y}} \right)^2$
        Residual of continuity:$\nabla\cdot\mathbf{u}$
        
    Args:
        domain_u (Domain): The domain for the x-velocity component.
        domain_v (Domain): The domain for the y-velocity component.
        domain_p (Domain): The domain for the pressure component.
        force_x (torch.Tensor): The x-component of the external force.
        force_y (torch.Tensor): The y-component of the external force.
        domain_force_x (Domain): The domain for the x-component of the external force.
        domain_force_y (Domain): The domain for the y-component of the external force.
        order (int): The order of accuracy for the finite difference scheme.
        device (str, optional): The device to use for computation (default: "cpu").
        dtype (torch.dtype, optional): The data type to use for computation (default: torch.float32).
    """

    def __init__(self, 
                 domain_u: Domain,
                 domain_v: Domain, 
                 domain_p: Domain, 
                 force_x: torch.Tensor,
                 force_y: torch.Tensor,
                 domain_force_x: Domain, 
                 domain_force_y: Domain,
                 order: int,
                 device="cpu", 
                 dtype=torch.float32) -> None:
        super().__init__(order, device=device, dtype=dtype)
        self.pressure = ScalarField(domain=domain_p)
        self.velocity = VectorValue(ScalarField(domain=domain_u), ScalarField(domain=domain_v))
        self.force = VectorValue(ScalarField(force_x, domain=domain_force_x),ScalarField(force_y, domain=domain_force_y))

    def __call__(self, u, v, p):
        """
        Compute the solution of the Poisson equation for pressure and the divergence of the velocity field.
        
        Args:
            u (torch.Tensor): The x-velocity component.
            v (torch.Tensor): The y-velocity component.
            p (torch.Tensor): The pressure component.
        
        Returns:
            residual (torch.Tensor): The concatenated tensor of the Poisson equation and the divergence components.
        """
        self.velocity.ux.register_value(u)
        self.velocity.uy.register_value(v)
        self.pressure.register_value(p)
        poisson = (self.nabla2*self.pressure)+(self.grad_x*self.velocity.ux)**2+2*(self.grad_y*self.velocity.ux)*(self.grad_x*self.velocity.uy)+(self.grad_y*self.velocity.uy)**2-self.nabla@self.force
        divergence = self.nabla@self.velocity
        return torch.cat([poisson.value, divergence.value], dim=-3)


class PoissonDivergence(FieldOperations):
    r"""
    Class representing the Poisson equation for pressure and the divergence of the velocity field.
        Returns a two channel tensor containing the residual of the Poisson equation and the divergence of the velocity field.
        Residual of Poisson equation:$\left( {{\partial^2 p} \over {\partial x^2}} + {{\partial^2 p} \over {\partial y^2}} \right) = \left( {{\partial u} \over {\partial x}} \right)^2 + 2 {{\partial u} \over {\partial y}} {{\partial v} \over {\partial x}}+ \left( {{\partial v} \over {\partial y}} \right)^2$
        Residual of continuity:$\nabla\cdot\mathbf{u}$
        
    Args:
        domain_u (Domain): The domain for the x-velocity component.
        domain_v (Domain): The domain for the y-velocity component.
        domain_p (Domain): The domain for the pressure component.
        order (int): The order of accuracy for the finite difference scheme.
        device (str, optional): The device to use for computation (default: "cpu").
        dtype (torch.dtype, optional): The data type to use for computation (default: torch.float32).
    """

    def __init__(self, 
                 domain_u: Domain,
                 domain_v: Domain, 
                 domain_p: Domain, 
                 order: int,
                 device="cpu", 
                 dtype=torch.float32) -> None:
        super().__init__(order, device=device, dtype=dtype)
        self.pressure = ScalarField(domain=domain_p)
        self.velocity = VectorValue(ScalarField(domain=domain_u), ScalarField(domain=domain_v))

    def __call__(self, u, v, p):
        """
        Compute the solution of the Poisson equation for pressure and the divergence of the velocity field.
        
        Args:
            u (torch.Tensor): The x-velocity component.
            v (torch.Tensor): The y-velocity component.
            p (torch.Tensor): The pressure component.
            
        Returns:
            residual (torch.Tensor): The concatenated tensor of the Poisson equation and the divergence components.
        """
        self.velocity.ux.register_value(u)
        self.velocity.uy.register_value(v)
        self.pressure.register_value(p)
        poisson = (self.nabla2*self.pressure)+(self.grad_x*self.velocity.ux)**2+2*(self.grad_y*self.velocity.ux)*(self.grad_x*self.velocity.uy)+(self.grad_y*self.velocity.uy)**2
        divergence = self.nabla@self.velocity
        return torch.cat([poisson.value, divergence.value], dim=-3)