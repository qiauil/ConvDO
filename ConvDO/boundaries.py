#usr/bin/python3
# -*- coding: UTF-8 -*-
from .helpers import *
from .faces import *
from .meta_type import *

class Boundary(CommutativeValue):
    
    def __init__(self) -> None:
        pass
    
    def correct_top(self,padded_face,ori_field,delta):
        pass

    def correct_right(self,padded_face,ori_field,delta):
        pass

    def correct_bottom(self,padded_face,ori_field,delta):
        pass
        
    def correct_left(self,padded_face,ori_field,delta):
        pass
  
class DirichletBoundary(Boundary):

    def __init__(self,boundary_value) -> None:
        super().__init__()
        self.boundary_value=boundary_value
        self.face_calculator=DirichletFace(boundary_value)
    
    def correct_top(self,padded_face,ori_field,delta):
        padded_face[...,0,:]=self.face_calculator.correct_outward_face(padded_face[...,0,:])
        return padded_face

    def correct_right(self,padded_face,ori_field,delta):
        padded_face[...,:,-1]=self.face_calculator.correct_outward_face(padded_face[...,:,-1])  
        return padded_face

    def correct_bottom(self,padded_face,ori_field,delta):
        padded_face[...,-1,:]=self.face_calculator.correct_inward_face(padded_face[...,-1,:])  
        return padded_face
        
    def correct_left(self,padded_face,ori_field,delta):
        padded_face[...,:,0]=self.face_calculator.correct_inward_face(padded_face[...,:,0]) 
        return padded_face   
    
    # + ： 
    def __add__(self, other):
        if isinstance(other,DirichletBoundary):
            # Dirichlet+Dirichlet=Dirichlet
            return DirichletBoundary(self.boundary_value+other.boundary_value)
        elif isinstance(other,Boundary):
            # Dirichlet+otherboundary=uncontrainedBoundary
            return UnConstrainedBoundary()
        else:
            try:
                # Dirichlet+number=Dirichlet
                return DirichletBoundary(self.boundary_value*other)
            except TypeError:
                return NotImplemented

    # *           
    def __mul__(self,other):
        if isinstance(other,DirichletBoundary):
            # Dirichlet*Dirichlet=Dirichlet
            return DirichletBoundary(self.boundary_value*other.boundary_value)
        elif isinstance(other,Boundary):
            # Dirichlet*otherboundary=uncontrainedBoundary
            return UnConstrainedBoundary()
        else:
            try:
                # Dirichlet*number=Dirichlet
                return DirichletBoundary(self.boundary_value*other)
            except TypeError:
                return NotImplemented

class NeumannBoundary(Boundary):

    def __init__(self,face_gradient) -> None:
        super().__init__()
        self.face_gradient=face_gradient
        self.face_calculator=NeumannFace(face_gradient)
    
    def correct_top(self,padded_face,ori_field,delta):
        padded_face[...,0,:]=self.face_calculator.correct_outward_face(ori_field[...,0,:],delta)
        return padded_face

    def correct_right(self,padded_face,ori_field,delta):
        padded_face[...,:,-1]=self.face_calculator.correct_outward_face(ori_field[...,:,-1],delta)  
        return padded_face

    def correct_bottom(self,padded_face,ori_field,delta):
        padded_face[...,-1,:]=self.face_calculator.correct_inward_face(ori_field[...,-1,:],delta)  
        return padded_face
        
    def correct_left(self,padded_face,ori_field,delta):
        padded_face[...,:,0]=self.face_calculator.correct_inward_face(ori_field[...,:,0],delta) 
        return padded_face    

    # + ： 
    def __add__(self, other):
        if isinstance(other,NeumannBoundary):
            # Neumann+Neumann=Dirichlet
            return NeumannBoundary(self.face_gradient+other.face_gradient)
        elif isinstance(other,Boundary):
            # Dirichlet+otherboundary=uncontrainedBoundary
            return UnConstrainedBoundary()
        else:
            try:
                # Neumann+number=NeumannBoundary
                return NeumannBoundary(self.face_gradient)
            except TypeError:
                return NotImplemented

    # *           
    def __mul__(self,other):
        if isinstance(other,Boundary):
            # Dirichlet*otherboundary=uncontrainedBoundary
            return UnConstrainedBoundary()
        else:
            try:
                # Neumann*number=Neumann
                return DirichletBoundary(self.boundary_value*other)
            except TypeError:
                return NotImplemented


class UnConstrainedBoundary(Boundary):

    def __init__(self) -> None:
        super().__init__()
        self.face_calculator=UnConstrainedFace()
    
    def correct_top(self,padded_face,ori_field,delta):
        padded_face[...,0,:]=self.face_calculator.correct_outward_face(ori_field[...,0,:],ori_field[...,1,:],ori_field[...,2,:])
        return padded_face

    def correct_right(self,padded_face,ori_field,delta):
        padded_face[...,:,-1]=self.face_calculator.correct_outward_face(ori_field[...,:,-1],ori_field[...,:,-2],ori_field[...,:,-3])  
        return padded_face

    def correct_bottom(self,padded_face,ori_field,delta):
        padded_face[...,-1,:]=self.face_calculator.correct_outward_face(ori_field[...,-1,:],ori_field[...,-2,:],ori_field[...,-3,:])
        return padded_face
        
    def correct_left(self,padded_face,ori_field,delta):
        padded_face[...,:,0]=self.face_calculator.correct_outward_face(ori_field[...,:,0],ori_field[...,:,1],ori_field[...,:,2]) 
        return padded_face    

    # + ： 
    def __add__(self, other):
        return UnConstrainedBoundary()
    # *           
    def __mul__(self,other):
        return UnConstrainedBoundary()
            
    
class PeriodicBoundary(Boundary):
    
    def __init__(self) -> None:
        super().__init__()
    
    def correct_top(self,padded_face,ori_field,delta):
        padded_face[...,0,:]=(ori_field[...,0,:]+ori_field[...,-1,:])/2
        return padded_face

    def correct_right(self,padded_face,ori_field,delta):
        padded_face[...,:,-1]=(ori_field[...,:,0]+ori_field[...,:,-1])/2
        return padded_face

    def correct_bottom(self,padded_face,ori_field,delta):
        padded_face[...,-1,:]=(ori_field[...,0,:]+ori_field[...,-1,:])/2
        return padded_face
        
    def correct_left(self,padded_face,ori_field,delta):
        padded_face[...,:,0]=(ori_field[...,:,0]+ori_field[...,:,-1])/2
        return padded_face    

    # + ： 
    def __add__(self, other):
        if isinstance(other,PeriodicBoundary):
            return PeriodicBoundary()
        elif isinstance(other,Boundary):
            return UnConstrainedBoundary()
        else:
            return PeriodicBoundary()

    # *           
    def __mul__(self,other):
        if isinstance(other,PeriodicBoundary):
            return PeriodicBoundary()
        elif isinstance(other,Boundary):
            return UnConstrainedBoundary()
        else:
            return PeriodicBoundary()
