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
    '''
    DirichletBoundary is a boundary condition that the value of the field is fixed at the boundary.
    '''

    def __init__(self,boundary_value: float) -> None:
        super().__init__()
        self.boundary_face=DirichletFace(boundary_value)
    
    def correct_top(self,padded_face,ori_field,delta):
        padded_face[...,0,:]=self.boundary_face.correct_outward_padding(ori_field[...,0,:])
        return padded_face

    def correct_right(self,padded_face,ori_field,delta):
        padded_face[...,:,-1]=self.boundary_face.correct_outward_padding(ori_field[...,:,-1])  
        return padded_face

    def correct_bottom(self,padded_face,ori_field,delta):
        padded_face[...,-1,:]=self.boundary_face.correct_inward_padding(ori_field[...,-1,:])  
        return padded_face
        
    def correct_left(self,padded_face,ori_field,delta):
        padded_face[...,:,0]=self.boundary_face.correct_inward_padding(ori_field[...,:,0]) 
        return padded_face   
    
    # + ： 
    def __add__(self, other):
        if isinstance(other,DirichletBoundary):
            # Dirichlet+Dirichlet=Dirichlet
            return DirichletBoundary(self.boundary_face.face_value+other.boundary_face.face_value)
        elif isinstance(other,Boundary):
            # Dirichlet+otherboundary=uncontrainedBoundary
            return UnConstrainedBoundary()
        else:
            try:
                # Dirichlet+number=Dirichlet
                return DirichletBoundary(self.boundary_face.face_value+other)
            except Exception:
                return NotImplemented

    # *           
    def __mul__(self,other):
        if isinstance(other,DirichletBoundary):
            # Dirichlet*Dirichlet=Dirichlet
            return DirichletBoundary(self.boundary_face.face_value*other.boundary_face.face_value)
        elif isinstance(other,Boundary):
            # Dirichlet*otherboundary=uncontrainedBoundary
            return UnConstrainedBoundary()
        else:
            try:
                # Dirichlet*number=Dirichlet
                return DirichletBoundary(self.boundary_face.face_value*other)
            except Exception:
                return NotImplemented
            
    def __truediv__(self, other):
        if isinstance(other,DirichletBoundary):
            return DirichletBoundary(self.boundary_face.face_value/other.boundary_face.face_value)
        elif isinstance(other,Boundary):
            return UnConstrainedBoundary()
        else:
            try:
                return DirichletBoundary(self.boundary_face.face_value/other)
            except Exception:
                return NotImplemented
        
    def __rtruediv__(self, other):
        if isinstance(other,DirichletBoundary):
            return DirichletBoundary(other.boundary_face.face_value/self.boundary_face.face_value)
        elif isinstance(other,Boundary):
            return UnConstrainedBoundary()
        else:
            try:
                return DirichletBoundary(other/self.boundary_face.face_value)
            except Exception:
                return NotImplemented

    def __pow__(self, other):
        return DirichletBoundary(self.boundary_face.face_value**other)

class NeumannBoundary(Boundary):
    '''
    NeumannBoundary is a boundary condition that the gradient of the field is fixed at the boundary.
    '''

    def __init__(self,face_gradient: float) -> None:
        super().__init__()
        self.boundary_face=NeumannFace(face_gradient)
    
    def correct_top(self,padded_face,ori_field,delta):
        padded_face[...,0,:]=self.boundary_face.correct_outward_padding(ori_field[...,0,:],delta)
        return padded_face

    def correct_right(self,padded_face,ori_field,delta):
        padded_face[...,:,-1]=self.boundary_face.correct_outward_padding(ori_field[...,:,-1],delta)  
        return padded_face

    def correct_bottom(self,padded_face,ori_field,delta):
        padded_face[...,-1,:]=self.boundary_face.correct_inward_padding(ori_field[...,-1,:],delta)  
        return padded_face
        
    def correct_left(self,padded_face,ori_field,delta):
        padded_face[...,:,0]=self.boundary_face.correct_inward_padding(ori_field[...,:,0],delta) 
        return padded_face    

    # + ： 
    def __add__(self, other):
        if isinstance(other,NeumannBoundary):
            # Neumann+Neumann=Dirichlet
            return NeumannBoundary(self.boundary_face.face_gradient+other.boundary_face.face_gradient)
        elif isinstance(other,Boundary):
            # Neumann+otherboundary=uncontrainedBoundary
            return UnConstrainedBoundary()
        else:
            try:
                # Neumann+number=Neumann
                return NeumannBoundary(self.boundary_face.face_gradient)
            except Exception:
                return NotImplemented

    # *           
    def __mul__(self,other):
        if isinstance(other,Boundary):
            # Dirichlet*otherboundary=uncontrainedBoundary
            return UnConstrainedBoundary()
        else:
            try:
                # Neumann*number=Neumann*number
                return DirichletBoundary(self.boundary_face.face_value*other)
            except Exception:
                return NotImplemented

    def __truediv__(self, other):
        if isinstance(other,Boundary):
            return UnConstrainedBoundary()
        else:
            try:
                return DirichletBoundary(self.boundary_face.face_value/other)
            except Exception:
                return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other,Boundary):
            return UnConstrainedBoundary()
        else:
            try:
                return DirichletBoundary(other/self.boundary_face.face_value)
            except Exception:
                return NotImplemented

    def __pow__(self, other):
        return UnConstrainedBoundary()


class UnConstrainedBoundary(Boundary):
    '''
    UnConstrainedBoundary is a boundary condition that the value at the boundary is calculated by the value of the neighbour cells.
    If you are not sure about the boundary condition, you can use UnConstrainedBoundary.
    The 
    '''

    def __init__(self) -> None:
        super().__init__()
        self.boundary_face=UnConstrainedFace()
    
    def correct_top(self,padded_face,ori_field,delta):
        padded_face[...,0,:]=self.boundary_face.correct_outward_padding(ori_field[...,0,:],ori_field[...,1,:],ori_field[...,2,:])
        return padded_face

    def correct_right(self,padded_face,ori_field,delta):
        padded_face[...,:,-1]=self.boundary_face.correct_outward_padding(ori_field[...,:,-1],ori_field[...,:,-2],ori_field[...,:,-3])  
        return padded_face

    def correct_bottom(self,padded_face,ori_field,delta):
        padded_face[...,-1,:]=self.boundary_face.correct_outward_padding(ori_field[...,-1,:],ori_field[...,-2,:],ori_field[...,-3,:])
        return padded_face
        
    def correct_left(self,padded_face,ori_field,delta):
        padded_face[...,:,0]=self.boundary_face.correct_outward_padding(ori_field[...,:,0],ori_field[...,:,1],ori_field[...,:,2]) 
        return padded_face    

    # + ： 
    def __add__(self, other):
        return UnConstrainedBoundary()
    # *           
    def __mul__(self,other):
        return UnConstrainedBoundary()

    def __pow__(self, other):
        return UnConstrainedBoundary()
    
    def __truediv__(self, other):
        return UnConstrainedBoundary()
            
    def __rtruediv__(self, other):
        return UnConstrainedBoundary()
    
class PeriodicBoundary(Boundary):
    '''
    Periodic boundary conditions.
    '''
    
    def __init__(self) -> None:
        super().__init__()
    
    def correct_top(self,padded_face,ori_field,delta):
        print("Warning: correct only works for 2nd scheme, if you are using higher order, please directly pad the field with 'circular' model")
        padded_face[...,0,:]=(ori_field[...,0,:]+ori_field[...,-1,:])/2
        return padded_face

    def correct_right(self,padded_face,ori_field,delta):
        print("Warning: correct only works for 2nd scheme, if you are using higher order, please directly pad the field with 'circular' model")
        padded_face[...,:,-1]=(ori_field[...,:,0]+ori_field[...,:,-1])/2
        return padded_face

    def correct_bottom(self,padded_face,ori_field,delta):
        print("Warning: correct only works for 2nd scheme, if you are using higher order, please directly pad the field with 'circular' model")
        padded_face[...,-1,:]=(ori_field[...,0,:]+ori_field[...,-1,:])/2
        return padded_face
        
    def correct_left(self,padded_face,ori_field,delta):
        print("Warning: correct only works for 2nd scheme, if you are using higher order, please directly pad the field with 'circular' model")
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
        
    def __pow__(self, other):
        return PeriodicBoundary()
    
    def __truediv__(self, other):
        if isinstance(other,PeriodicBoundary):
            return PeriodicBoundary()
        else:
            return UnConstrainedBoundary()
        
    def __rtruediv__(self, other):
        if isinstance(other,PeriodicBoundary):
            return PeriodicBoundary()
        else:
            return UnConstrainedBoundary()