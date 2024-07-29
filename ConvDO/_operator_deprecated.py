#usr/bin/python3
# -*- coding: UTF-8 -*-
import torch
from . import *
from .domain import *
from .obstacles import *
from .boundaries import *
from typing import Optional

print("Warning: This is an old version of operator which will be deprecated soon. Please use 'operator_HO' instead.")

class ScalarField(CommutativeValue):
    
    def __init__(self,value:torch.Tensor,
                 domain:Optional[Domain]=UnconstrainedDomain()) -> None:
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

class GradX():
    
    def __mul__(self,other):
        if isinstance(other,ScalarField):
                domain=other.domain
                scalar_field=other.value
                face_value=(scalar_field[...,0:-1]+scalar_field[...,1:])/2
                padded_face=nn.functional.pad(face_value,(1,1,0,0),"constant",0)
                padded_face=domain.left_boundary.correct_left(padded_face,scalar_field,domain.delta_x)
                padded_face=domain.right_boundary.correct_right(padded_face,scalar_field,domain.delta_x)
                for obstacle in domain.obstacles:
                    padded_face=obstacle.correct_left(padded_face,scalar_field,domain.delta_x)
                    padded_face=obstacle.correct_right(padded_face,scalar_field,domain.delta_x)
                gradient=(padded_face[...,1:]-padded_face[...,0:-1])/domain.delta_x
                for obstacle in domain.obstacles:
                    gradient=obstacle.fill_internal_field(gradient)
                return ScalarField(gradient,Domain(
                                                boundaries=[
                                                    PeriodicBoundary() if isinstance(boundary,PeriodicBoundary) else UnConstrainedBoundary() for boundary in [domain.left_boundary,
                                                                                                                                                            domain.right_boundary,
                                                                                                                                                            domain.top_boundary,
                                                                                                                                                            domain.bottom_boundary]
                                                ],
                                                delta_x=other.domain.delta_x,delta_y=other.domain.delta_y,
                                                obstacles=other.domain.obstacles)
                                    )
        else:
            try:
                return self*other
            except Exception:
                return NotImplemented    
            
class GradY():

    def __mul__(self,other):
        if isinstance(other,ScalarField):
                domain=other.domain
                scalar_field=other.value
                face_value=(scalar_field[...,0:-1,:]+scalar_field[...,1:,:])/2
                padded_face=nn.functional.pad(face_value,(0,0,1,1),"constant",0)
                padded_face=domain.top_boundary.correct_top(padded_face,scalar_field,domain.delta_y)
                padded_face=domain.bottom_boundary.correct_bottom(padded_face,scalar_field,domain.delta_y)
                for obstacle in domain.obstacles:
                    padded_face=obstacle.correct_top(padded_face,scalar_field,domain.delta_y)
                    padded_face=obstacle.correct_bottom(padded_face,scalar_field,domain.delta_y)
                gradient=(padded_face[...,0:-1,:]-padded_face[...,1:,:])/domain.delta_y
                for obstacle in domain.obstacles:
                    gradient=obstacle.fill_internal_field(gradient)
                return ScalarField(gradient,Domain(
                                                boundaries=[
                                                    PeriodicBoundary() if isinstance(boundary,PeriodicBoundary) else UnConstrainedBoundary() for boundary in [domain.left_boundary,
                                                                                                                                                            domain.right_boundary,
                                                                                                                                                            domain.top_boundary,
                                                                                                                                                            domain.bottom_boundary]
                                                ],
                                                delta_x=other.domain.delta_x,delta_y=other.domain.delta_y,
                                                obstacles=other.domain.obstacles)
                                    )
        else:
            try:
                return self*other
            except Exception:
                return NotImplemented    

class LaplacianX():
    
    def __mul__(self,other):
        if isinstance(other,ScalarField):
                domain=other.domain
                scalar_field=other.value
                face_value=(scalar_field[...,0:-1]+scalar_field[...,1:])/2
                padded_face=nn.functional.pad(face_value,(1,1,0,0),"constant",0)
                padded_face=domain.left_boundary.correct_left(padded_face,scalar_field,domain.delta_x)
                padded_face=domain.right_boundary.correct_right(padded_face,scalar_field,domain.delta_x)
                for obstacle in domain.obstacles:
                    padded_face=obstacle.correct_left(padded_face,scalar_field,domain.delta_x)
                    padded_face=obstacle.correct_right(padded_face,scalar_field,domain.delta_x)
                laplacian=(2*(padded_face[...,1:]+padded_face[...,0:-1])-4*scalar_field)/(domain.delta_x*domain.delta_x)
                for obstacle in domain.obstacles:
                    laplacian=obstacle.fill_internal_field(laplacian)
                return ScalarField(laplacian,Domain(
                                                boundaries=[
                                                    PeriodicBoundary() if isinstance(boundary,PeriodicBoundary) else UnConstrainedBoundary() for boundary in [domain.left_boundary,
                                                                                                                                                            domain.right_boundary,
                                                                                                                                                            domain.top_boundary,
                                                                                                                                                            domain.bottom_boundary]
                                                ],
                                                delta_x=other.domain.delta_x,delta_y=other.domain.delta_y,
                                                obstacles=other.domain.obstacles)
                                    )
        else:
            try:
                return self*other
            except Exception:
                return NotImplemented     

class LaplacianY():

    def __mul__(self,other):
        if isinstance(other,ScalarField):
                domain=other.domain
                scalar_field=other.value
                face_value=(scalar_field[...,0:-1,:]+scalar_field[...,1:,:])/2
                padded_face=nn.functional.pad(face_value,(0,0,1,1),"constant",0)
                padded_face=domain.top_boundary.correct_top(padded_face,scalar_field,domain.delta_y)
                padded_face=domain.bottom_boundary.correct_bottom(padded_face,scalar_field,domain.delta_y)
                for obstacle in domain.obstacles:
                    padded_face=obstacle.correct_top(padded_face,scalar_field,domain.delta_y)
                    padded_face=obstacle.correct_bottom(padded_face,scalar_field,domain.delta_y)
                laplacian=(2*(padded_face[...,0:-1,:]+padded_face[...,1:,:])-4*scalar_field)/(domain.delta_y*domain.delta_y)
                for obstacle in domain.obstacles:
                    laplacian=obstacle.fill_internal_field(laplacian)
                return ScalarField(laplacian,Domain(
                                                boundaries=[
                                                    PeriodicBoundary() if isinstance(boundary,PeriodicBoundary) else UnConstrainedBoundary() for boundary in [domain.left_boundary,
                                                                                                                                                            domain.right_boundary,
                                                                                                                                                            domain.top_boundary,
                                                                                                                                                            domain.bottom_boundary]
                                                ],
                                                delta_x=other.domain.delta_x,delta_y=other.domain.delta_y,
                                                obstacles=other.domain.obstacles)
                                    )
        else:
            try:
                return self*other
            except Exception:
                return NotImplemented   

class Laplacian():
    def __init__(self) -> None:
        self.op_x=LaplacianX()
        self.op_y=LaplacianY()

    def __mul__(self,other):
        if isinstance(other,ScalarField):
            return self.op_x*other+self.op_y*other
        elif isinstance(other,VectorValue):
            return VectorValue(
                self.op_x*other.ux+self.op_y*other.ux,
                self.op_x*other.uy+self.op_y*other.uy
            )         
    

grad_x=GradX()

grad_y=GradY()

grad2_x=LaplacianX()

grad2_y=LaplacianY()

nabla=VectorValue(grad_x,grad_y) 

nabla2=Laplacian()

'''
def res_NS_transient(u_0,v_0,p_0,u_1,v_1,p_1,force_x,force_y,domain_u,domain_v,domain_p,domain_force_x,domain_force_y,viscosity,dt):
    p_0=ScalarField(p_0,domain_p)
    p_1=ScalarField(p_1,domain_p)
    u_0=VectorValue(ScalarField(u_0,domain_u),ScalarField(v_0,domain_v))
    u_1=VectorValue(ScalarField(u_1,domain_u),ScalarField(v_1,domain_v))
    force=VectorValue(ScalarField(force_x,domain_force_x),ScalarField(force_y,domain_force_y))
    transient=(u_1-u_0)*(1/dt)
    advection=u_0@(nabla*u_0)
    pressure=nabla*p_1
    vis=-1*viscosity*(nabla2*u_0)
    ns_res=transient+advection+pressure+vis-force
    con_res=nabla@u_1
    return torch.cat([ns_res.ux.value,ns_res.uy.value,con_res.value],dim=1)
'''
from foxutils.plotter.field import show_each_channel

def res_NS_transient_force(u_0,v_0,p_0,u_1,v_1,p_1,force_x,force_y,domain_u,domain_v,domain_p,domain_force_x,domain_force_y,viscosity,dt):
    p_0=ScalarField(p_0,domain_p)
    p_1=ScalarField(p_1,domain_p)
    u_0=VectorValue(ScalarField(u_0,domain_u),ScalarField(v_0,domain_v))
    u_1=VectorValue(ScalarField(u_1,domain_u),ScalarField(v_1,domain_v))
    force=VectorValue(ScalarField(force_x,domain_force_x),ScalarField(force_y,domain_force_y))
    u_inter=(u_1+u_0)*0.5
    transient=(u_1-u_0)*(1/dt)
    advection=u_inter@(nabla*u_inter)
    pressure=nabla*p_0
    vis=-1*viscosity*(nabla2*u_inter)
    ns_res=transient+advection+pressure+vis-force
    con_res=(nabla2*p_1)+((grad_x*u_1.ux)*(grad_x*u_1.ux)+2*(grad_y*u_1.ux)*(grad_x*u_1.uy)+(grad_y*u_1.uy)*(grad_y*u_1.uy))-nabla@force
    divergence=nabla@u_1
    return torch.cat([ns_res.ux.value,ns_res.uy.value,con_res.value,divergence.value],dim=-3)

def res_NS_transient(u_0,v_0,p_0,u_1,v_1,p_1,domain_u,domain_v,domain_p,viscosity,dt):
    p_0=ScalarField(p_0,domain_p)
    p_1=ScalarField(p_1,domain_p)
    u_0=VectorValue(ScalarField(u_0,domain_u),ScalarField(v_0,domain_v))
    u_1=VectorValue(ScalarField(u_1,domain_u),ScalarField(v_1,domain_v))
    u_inter=(u_1+u_0)*0.5
    transient=(u_1-u_0)*(1/dt)
    advection=u_inter@(nabla*u_inter)
    pressure=nabla*p_0
    vis=-1*viscosity*(nabla2*u_inter)
    ns_res=transient+advection+pressure+vis
    con_res=(nabla2*p_1)+((grad_x*u_1.ux)*(grad_x*u_1.ux)+2*(grad_y*u_1.ux)*(grad_x*u_1.uy)+(grad_y*u_1.uy)*(grad_y*u_1.uy))
    divergence=nabla@u_1
    return torch.cat([ns_res.ux.value,ns_res.uy.value,con_res.value,divergence.value],dim=-3)

def res_NS_pressure_velocity(u,v,p,domain_u,domain_v,domain_p):
    p_1=ScalarField(p,domain_p)
    u_1=VectorValue(ScalarField(u,domain_u),ScalarField(v,domain_v))
    poisson=(nabla2*p_1)+((grad_x*u_1.ux)*(grad_x*u_1.ux)+2*(grad_y*u_1.ux)*(grad_x*u_1.uy)+(grad_y*u_1.uy)*(grad_y*u_1.uy))
    divergence=nabla@u_1
    return torch.cat([poisson.value,divergence.value],dim=-3)

def res_NS_pressure_velocity_force(u,v,p,domain_u,domain_v,domain_p,force_x,force_y,domain_force_x,domain_force_y,):
    force=VectorValue(ScalarField(force_x,domain_force_x),ScalarField(force_y,domain_force_y))
    p_1=ScalarField(p,domain_p)
    u_1=VectorValue(ScalarField(u,domain_u),ScalarField(v,domain_v))
    poisson=(nabla2*p_1)+((grad_x*u_1.ux)*(grad_x*u_1.ux)+2*(grad_y*u_1.ux)*(grad_x*u_1.uy)+(grad_y*u_1.uy)*(grad_y*u_1.uy))-nabla@force
    divergence=nabla@u_1
    return torch.cat([poisson.value,divergence.value],dim=-3)