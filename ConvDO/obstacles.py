#usr/bin/python3
# -*- coding: UTF-8 -*-
from .helpers import *
from .faces import *
from .domain import *
from .conv_operators import *
from .meta_type import *
import torch
from typing import Union,Sequence,Optional

def is_shape_equal(shape_filed1:ScalarField,shape_field2:ScalarField):
    return torch.sum(shape_filed1.value-shape_field2.value)==0

def generate_circle_2D(center_x,center_y,radius,length_x,length_y,dx=1,dy=1):
    X, Y=np.ogrid[0.5:int(length_x/dx)+0.5, 0.5:int(length_y/dy)+0.5]
    X=X*dx
    Y=Y*dy
    dist_from_center = torch.tensor(np.sqrt((X - center_x)**2 + (Y-center_y)**2))
    return torch.where(dist_from_center < radius,0.0,1.0)

class Obstacle(CommutativeValue):
    """
    A base class to represent an obstacle.
    
    Args:
        shape_field (Union[torch.Tensor,ScalarField]): A 2D tensor or a ScalarField object representing the shape field of the obstacle.
            Note that the shape field is a binary field where 0 represents the obstacle region.
        lrbt_region (Optional[Sequence], optional): A sequence of four elements representing the left, right, bottom, and top regions of the obstacle. Defaults to None.
    """
    
    def __init__(self,shape_field:Union[torch.Tensor,ScalarField],
                 lrbt_region:Optional[Sequence]=None) -> None:
        if isinstance(shape_field,torch.Tensor):
            shape_domain=UnconstrainedDomain()
            self.shape_field=ScalarField(shape_field,domain=shape_domain) #01 field where 0 inside the obstacle
        elif isinstance(shape_field,ScalarField):
            if not isinstance(shape_field.domain,UnconstrainedDomain):
                raise ValueError("The domain of the shape field must be unconstrained")
            self.shape_field=shape_field
        if lrbt_region is not None:
            if len(lrbt_region)!=4:
                raise ValueError("The lrbt_region must be a sequence of 4 elements")
            self.x_left=lrbt_region[0]
            self.x_right=lrbt_region[1]
            self.y_bottom=lrbt_region[2]
            self.y_top=lrbt_region[3]
        else:
            gradx=ConvGrad(order=2,device=self.shape_field.value.device,direction='x')
            grady=ConvGrad(order=2,device=self.shape_field.value.device,direction='y')
            #dx_mask=nn.functional.pad((shape_field[...,1:]-shape_field[...,0:-1]),(1,1,0,0),"constant",0)
            dx_mask=nn.functional.pad((gradx*self.shape_field).value,(1,1,1,1),"constant",0)
            #NOTE: right is the right corresponding to the internal cell, that is right is the left side of the obstacle
            self.x_right=torch.where(dx_mask < -0.5, 1.0, 0.0)
            self.x_left=torch.where(dx_mask > 0.5, 1.0, 0.0)
            #dy_mask=nn.functional.pad((shape_field[...,0:-1,:]-shape_field[...,1:,:]),(0,0,1,1),"constant",0)
            dy_mask=nn.functional.pad((grady*self.shape_field).value,(1,1,1,1),"constant",0)
            self.y_bottom=torch.where(dy_mask > 0.5, 1.0, 0.0)
            self.y_top=torch.where(dy_mask < -0.5, 1.0, 0.0)
    
    def correct_left(self,padded_face,ori_field,delta):
        raise NotImplementedError
    def correct_right(self,padded_face,ori_field,delta):
        raise NotImplementedError

    def correct_top(self,padded_face,ori_field,delta):
        raise NotImplementedError

    def correct_bottom(self,padded_face,ori_field,delta):
       raise NotImplementedError

    def fill_internal_field(self,target_field):
        return target_field*self.shape_field.value

class DirichletObstacle(Obstacle):
    """
    A class to represent a Dirichlet obstacle.
    
    Args:
        shape_field (Union[torch.Tensor,ScalarField]): A 2D tensor or a ScalarField object representing the shape field of the obstacle.
            Note that the shape field is a binary field where 0 represents the obstacle region.
        boundary_value (float): The boundary value of the Dirichlet obstacle.
    """
    
    def __init__(self, shape_field:Union[torch.Tensor,ScalarField],boundary_value: float) -> None:
        super().__init__(shape_field)
        self.boundary_face=DirichletFace(boundary_value)

    def correct_left(self,padded_face,ori_field,delta):
        return torch.where(
            self.x_left>0.5,
            self.boundary_face.correct_inward_padding(padded_face),
            padded_face)
    
    def correct_right(self,padded_face,ori_field,delta):
        return torch.where(self.x_right>0.5,
                           self.boundary_face.correct_outward_padding(padded_face),
                           padded_face)

    def correct_top(self,padded_face,ori_field,delta):
        return torch.where(self.y_top>0.5,
                           self.boundary_face.correct_outward_padding(padded_face),
                           padded_face)
        
    def correct_bottom(self,padded_face,ori_field,delta):
        return torch.where(self.y_bottom>0.5,
                           self.boundary_face.correct_inward_padding(padded_face),
                           padded_face)

    # + ： 
    def __add__(self, other):
        if isinstance(other,Obstacle):
            if not is_shape_equal(self.shape_field,other.shape_field):
                raise ValueError("The two obstacles don't have same shape field.")
        if isinstance(other,DirichletObstacle):
            # Dirichlet+Dirichlet=Dirichlet
            return DirichletObstacle(self.shape_field, self.boundary_face.face_value+other.boundary_face.face_value)
        elif isinstance(other,Obstacle):
            return UnConstrainedObstacle(self.shape_field)
        else:
            try:
                # Dirichlet+number=Dirichlet
                return DirichletObstacle(self.boundary_face.face_value+other)
            except Exception:
                return NotImplemented

    # *           
    def __mul__(self,other):
        if isinstance(other,Obstacle):
            if not is_shape_equal(self.shape_field,other.shape_field):
                raise ValueError("The two obstacles don't have same shape field.")
        if isinstance(other,DirichletObstacle):
            # Dirichlet*Dirichlet=Dirichlet
            return DirichletObstacle(self.shape_field, self.boundary_face.face_value*other.boundary_face.face_value)
        elif isinstance(other,Obstacle):
            # Dirichlet*otherboundary=uncontrainedBoundary
            return UnConstrainedObstacle(self.shape_field)
        else:
            try:
                # Dirichlet*number=Dirichlet
                return DirichletObstacle(self.boundary_face.face_value*other)
            except Exception:
                return NotImplemented
            
    def __truediv__(self,other):
        if isinstance(other,Obstacle):
            if not is_shape_equal(self.shape_field,other.shape_field):
                raise ValueError("The two obstacles don't have same shape field.")
        if isinstance(other,DirichletObstacle):
            return DirichletObstacle(self.shape_field, self.boundary_face.face_value/other.boundary_face.face_value)
        elif isinstance(other,Obstacle):
            return UnConstrainedObstacle(self.shape_field)
        else:
            try:
                return DirichletObstacle(self.shape_field, self.boundary_face.face_value/other)
            except Exception:
                return NotImplemented
            
    def __rtruediv__(self,other):
        if isinstance(other,Obstacle):
            if not is_shape_equal(self.shape_field,other.shape_field):
                raise ValueError("The two obstacles don't have same shape field.")
        if isinstance(other,DirichletObstacle):
            return DirichletObstacle(self.shape_field, other.boundary_face.face_value/self.boundary_face.face_value)
        elif isinstance(other,Obstacle):
            return UnConstrainedObstacle(self.shape_field)
        else:
            try:
                return DirichletObstacle(self.shape_field, other/self.boundary_face.face_value)
            except Exception:
                return NotImplemented
    
    def __pow__(self,other):
        return DirichletObstacle(self.shape_field, self.boundary_face.face_value**other)
    
class NeumannObstacle(Obstacle):
    """
    A class to represent a Neumann obstacle.
    
    Args:
        shape_field (Union[torch.Tensor,ScalarField]): A 2D tensor or a ScalarField object representing the shape field of the obstacle.
            Note that the shape field is a binary field where 0 represents the obstacle region.
        boundary_gradient (float): The boundary gradient of the Neumann obstacle.
        
    
    """
    def __init__(self, shape_field:Union[torch.Tensor,ScalarField],boundary_gradient:float) -> None:
        super().__init__(shape_field)
        self.boundary_face=NeumannFace(boundary_gradient)
        
    def correct_left(self,padded_face,ori_field,delta):
        return torch.where(
            self.x_left>0.5,
            self.boundary_face.correct_inward_padding(padded_face,delta)
            ,padded_face)
    
    def correct_right(self,padded_face,ori_field,delta):
        return torch.where(self.x_right>0.5,
                           self.boundary_face.correct_outward_padding(padded_face,delta),
                           padded_face)

    def correct_top(self,padded_face,ori_field,delta):
        return torch.where(self.y_top>0.5,
                           self.boundary_face.correct_outward_padding(padded_face,delta),
                           padded_face)
        
    def correct_bottom(self,padded_face,ori_field,delta):
        return torch.where(self.y_bottom>0.5,
                           self.boundary_face.correct_inward_padding(padded_face,delta),
                           padded_face)

    # + ： 
    def __add__(self, other):
        if isinstance(other,Obstacle):
            if not is_shape_equal(self.shape_field,other.shape_field):
                raise ValueError("The two obstacles don't have same shape field.")
        if isinstance(other,NeumannObstacle):
            return NeumannObstacle(self.shape_field, 
                                     self.boundary_face.face_gradient+other.boundary_face.face_gradient
                                     )
        elif isinstance(other,Obstacle):
            return UnConstrainedObstacle(self.shape_field)
        else:
            try:
                # Neumann+number=Neumann
                return NeumannObstacle(self.boundary_face.face_value)
            except Exception:
                return NotImplemented
    
    def __mul__(self,other):
        if isinstance(other,Obstacle):
            if not is_shape_equal(self.shape_field,other.shape_field):
                raise ValueError("The two obstacles don't have same shape field.")
            # Neumann*otherboundary=uncontrainedBoundary
            return UnConstrainedObstacle(self.shape_field)
        else:
            try:
                # Dirichlet*number=Dirichlet
                return NeumannObstacle(self.boundary_face.face_gradient*other)
            except Exception:
                return NotImplemented
    
    def __truediv__(self,other):
        if isinstance(other,Obstacle):
            if not is_shape_equal(self.shape_field,other.shape_field):
                raise ValueError("The two obstacles don't have same shape field.")
            return UnConstrainedObstacle(self.shape_field)
        else:
            try:
                return NeumannObstacle(self.shape_field, self.boundary_face.face_gradient/other)
            except Exception:
                return NotImplemented
    
    def __rtruediv__(self,other):
        if isinstance(other,Obstacle):
            if not is_shape_equal(self.shape_field,other.shape_field):
                raise ValueError("The two obstacles don't have same shape field.")
            return UnConstrainedObstacle(self.shape_field)
        else:
            try:
                return NeumannObstacle(self.shape_field, other/self.boundary_face.face_gradient)
            except Exception:
                return NotImplemented
    
    def __pow__(self,other):
        return UnConstrainedBoundary()
    
class UnConstrainedObstacle(Obstacle):
    """
    A class to represent an unconstrained obstacle.
    
    Args:
        shape_field (Union[torch.Tensor,ScalarField]): A 2D tensor or a ScalarField object representing the shape field of the obstacle.
            Note that the shape field is a binary field where 0 represents the obstacle region.
    """

    def __init__(self, shape_field:Union[torch.Tensor,ScalarField]) -> None:
        super().__init__(shape_field)
        self.boundary_face=UnConstrainedFace()
        
    def correct_left(self,padded_face,ori_field,delta):
        return torch.where(self.x_left>0.5,
                           nn.functional.pad(
                            self.boundary_face.correct_inward_padding(nn.functional.pad(ori_field,(0,1,0,0),"constant",0),
                                                                  nn.functional.pad(ori_field[...,1:],(0,2,0,0),"constant",0),
                                                                  nn.functional.pad(ori_field[...,2:],(0,3,0,0),"constant",0)),
                            (1,0,1,1),"constant",0),
                           padded_face)
    
    def correct_right(self,padded_face,ori_field,delta):
        return torch.where(self.x_right>0.5,
                           nn.functional.pad(
                            self.boundary_face.correct_outward_padding(nn.functional.pad(ori_field,(1,0,0,0),"constant",0),
                                                                   nn.functional.pad(ori_field,(2,0,0,0),"constant",0)[...,:-1],
                                                                   nn.functional.pad(ori_field,(3,0,0,0),"constant",0)[...,:-2],),
                            (0,1,1,1),"constant",0),
                           padded_face)

    def correct_top(self,padded_face,ori_field,delta):
        return torch.where(self.y_top>0.5,
                           nn.functional.pad(
                           self.boundary_face.correct_outward_padding(nn.functional.pad(ori_field,(0,0,0,1),"constant",0),
                                                                   nn.functional.pad(ori_field[...,1:,:],(0,0,0,2),"constant",0),
                                                                   nn.functional.pad(ori_field[...,2:,:],(0,0,0,3),"constant",0)
                                                                   ),
                            (1,1,1,0),"constant",0),
                           padded_face)
        
    def correct_bottom(self,padded_face,ori_field,delta):
        return torch.where(self.y_bottom>0.5,
                           nn.functional.pad(
                            self.boundary_face.correct_inward_padding(nn.functional.pad(ori_field,(0,0,1,0),"constant",0),
                                                                  nn.functional.pad(ori_field,(0,0,2,0),"constant",0)[...,:-1,:],
                                                                  nn.functional.pad(ori_field,(0,0,3,0),"constant",0)[...,:-2,:]),
                            (1,1,0,1),"constant",0),
                           padded_face)

    # + ： 
    def __add__(self, other):
        if isinstance(other,Obstacle):
            if not is_shape_equal(self.shape_field,other.shape_field):
                raise ValueError("The two obstacles don't have same shape field.")
        return UnConstrainedObstacle(self.shape_field)

    
    def __mul__(self,other):
        if isinstance(other,Obstacle):
            if not is_shape_equal(self.shape_field,other.shape_field):
                raise ValueError("The two obstacles don't have same shape field.")
        return UnConstrainedObstacle(self.shape_field)
    
    def __truediv__(self,other):
        if isinstance(other,Obstacle):
            if not is_shape_equal(self.shape_field,other.shape_field):
                raise ValueError("The two obstacles don't have same shape field.")
        return UnConstrainedObstacle(self.shape_field)

    def __rtruediv__(self,other):
        if isinstance(other,Obstacle):
            if not is_shape_equal(self.shape_field,other.shape_field):
                raise ValueError("The two obstacles don't have same shape field.")
        return UnConstrainedObstacle(self.shape_field)

    def __pow__(self,other):
        return UnConstrainedBoundary()