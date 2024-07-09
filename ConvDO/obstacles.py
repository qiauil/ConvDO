#usr/bin/python3
# -*- coding: UTF-8 -*-
from .helpers import *
from .faces import *
from .domain import *
from .conv_operators import *

class Obstacle():
    
    def __init__(self,shape_field) -> None:
        shape_domain=UnconstrainedDomain()
        gradx=HOGrad(order=2,device=shape_field.device,direction='x')
        grady=HOGrad(order=2,device=shape_field.device,direction='y')
        shape=ScalarField(shape_field,domain=shape_domain)
        self.shape_field=shape_field #01 field where 0 inside the obstacle
        #dx_mask=nn.functional.pad((shape_field[...,1:]-shape_field[...,0:-1]),(1,1,0,0),"constant",0)
        dx_mask=nn.functional.pad((gradx*shape).value,(1,1,1,1),"constant",0)
        #NOTE: right is the right corresponding to the internal cell, that is right is the left side of the obstacle
        self.x_right=torch.where(dx_mask < -0.5, 1.0, 0.0)
        self.x_left=torch.where(dx_mask > 0.5, 1.0, 0.0)
        #dy_mask=nn.functional.pad((shape_field[...,0:-1,:]-shape_field[...,1:,:]),(0,0,1,1),"constant",0)
        dy_mask=nn.functional.pad((grady*shape).value,(1,1,1,1),"constant",0)
        self.y_bottom=torch.where(dy_mask > 0.5, 1.0, 0.0)
        self.y_top=torch.where(dy_mask < -0.5, 1.0, 0.0)
    
    def correct_left(self,padded_face,ori_field,delta):
        pass

    def correct_right(self,padded_face,ori_field,delta):
        pass

    def correct_top(self,padded_face,ori_field,delta):
        pass

    def correct_bottom(self,padded_face,ori_field,delta):
        pass

    def fill_internal_field(self,target_field):
        return target_field*self.shape_field

class DirichletObstacle(Obstacle):
    
    def __init__(self, shape_field,boundary_value) -> None:
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
    
class NeumannObstacle(Obstacle):
    
    def __init__(self, shape_field,boundary_gradient) -> None:
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
    
class UnConstrainedObstacle(Obstacle):

    def __init__(self, shape_field) -> None:
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

def generate_circle_2D(center_x,center_y,radius,length_x,length_y,dx=1,dy=1):
    X, Y=np.ogrid[0.5:int(length_x/dx)+0.5, 0.5:int(length_y/dy)+0.5]
    X=X*dx
    Y=Y*dy
    dist_from_center = torch.tensor(np.sqrt((X - center_x)**2 + (Y-center_y)**2))
    return torch.where(dist_from_center < radius,0.0,1.0)