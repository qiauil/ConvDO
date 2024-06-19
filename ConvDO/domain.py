#usr/bin/python3
# -*- coding: UTF-8 -*-
from . import *
from .meta_type import *
from .boundaries import *

class Domain():
    def __init__(self,boundaries,obstacles=[],delta_x=1,delta_y=1) -> None:
        self.delta_x=delta_x
        self.delta_y=delta_y
        if isinstance(boundaries,collections.Sequence):
            if len(boundaries)== 4:
                self.left_boundary=boundaries[0]
                self.right_boundary=boundaries[1]
                self.top_boundary=boundaries[2]
                self.bottom_boundary=boundaries[3]
            else:
                raise Exception("The length of boundaries need to be 4: '[left_boundary, right_boundary, top_boundary, bottom_boundary]'")          
        else:
            raise Exception("Boundaries need to be set as a sequence type of '[left_boundary, right_boundary, top_boundary, bottom_boundary]'")                  
        self.set_obstacles(obstacles=obstacles)   
    
    def set_obstacles(self,obstacles):
        if isinstance(obstacles,collections.Sequence):
            self.obstacles=obstacles
        else:
            raise Exception("obstacles need to be a sequence type.")          

    def __add__(self, other):
        if isinstance(other,Domain):
            return Domain(
                [
                    self.left_boundary+other.left_boundary,
                    self.right_boundary+other.right_boundary,
                    self.top_boundary+other.top_boundary,
                    self.bottom_boundary+other.bottom_boundary
                ],
                obstacles=self.obstacles,
                delta_x=self.delta_x,
                delta_y=self.delta_y
            )
        else:
            return Domain(
                [
                    self.left_boundary+other,
                    self.right_boundary+other,
                    self.top_boundary+other,
                    self.bottom_boundary+other
                ],
                obstacles=self.obstacles,
                delta_x=self.delta_x,
                delta_y=self.delta_y
            )            

    def __radd__(self, other):
        return self+other
    
    def __iadd__(self,other):
        return self+other

    def __sub__(self,other):
        try:
            return self+(-1*other)
        except TypeError:
            return NotImplemented

    def __rsub__(self,other):
        try:
            return other+(-1*self)
        except TypeError:
            return NotImplemented
        
    def __isub__(self,other):
        return self-other    

    def __mul__(self, other):
        if isinstance(other,Domain):
            return Domain(
                [
                    self.left_boundary*other.left_boundary,
                    self.right_boundary*other.right_boundary,
                    self.top_boundary*other.top_boundary,
                    self.bottom_boundary*other.bottom_boundary
                ],
                obstacles=self.obstacles,
                delta_x=self.delta_x,
                delta_y=self.delta_y
            )
        else:
            return Domain(
                [
                    self.left_boundary*other,
                    self.right_boundary*other,
                    self.top_boundary*other,
                    self.bottom_boundary*other
                ],
                obstacles=self.obstacles,
                delta_x=self.delta_x,
                delta_y=self.delta_y
            )   
    
    def __rmul__(self,other):
        return self*other       

    def __imul__(self,other):
        return self*other
  

def UnconstrainedDomain(obstacles=[],delta_x=1,delta_y=1):
    return Domain(boundaries=[UnConstrainedBoundary()]*4,obstacles=obstacles,delta_x=delta_x,delta_y=delta_y)

def PeriodicDomain(obstacles=[],delta_x=1,delta_y=1):
    return Domain(boundaries=[PeriodicBoundary()]*4,obstacles=obstacles,delta_x=delta_x,delta_y=delta_y)