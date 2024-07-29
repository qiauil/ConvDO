#usr/bin/python3
# -*- coding: UTF-8 -*-
from .helpers import *
from .meta_type import *
from .boundaries import *
from typing import Sequence

def _obstacles_add(self_obstacle,other_obstacle):
    if len(self_obstacle)!=len(other_obstacle):
        raise Exception("The number of obstacles in two domain need to be the same.")
    return [self_obstacle[i]+other_obstacle[i] for i in range(len(self_obstacle))]

def _obstacles_mul(self_obstacle,other_obstacle):
    if len(self_obstacle)!=len(other_obstacle):
        raise Exception("The number of obstacles in two domain need to be the same.")
    return [self_obstacle[i]*other_obstacle[i] for i in range(len(self_obstacle))]

def _obstacles_div(self_obstacle,other_obstacle):
    if len(self_obstacle)!=len(other_obstacle):
        raise Exception("The number of obstacles in two domain need to be the same.")
    return [self_obstacle[i]/other_obstacle[i] for i in range(len(self_obstacle))]

class Domain(CommutativeValue):
    """
    A class to represent a domain.
    
    Args:
        boundaries (Sequence): A sequence of four boundary objects representing the boundary conditions. 
            The order is [left_boundary, right_boundary, top_boundary, bottom_boundary].
        obstacles (Sequence, optional): A sequence of obstacle objects representing the solid obstacles inside the domain.
            Defaults to [].
        delta_x (float, optional): The grid spacing in the x direction. Defaults to 1.0.
        delta_y (float, optional): The grid spacing in the y direction. Defaults to 1.0.
    """
    def __init__(self,
                 boundaries:Sequence,obstacles=[],
                 delta_x:float=1.0,
                 delta_y:float=1.0) -> None:
        self.delta_x=delta_x
        self.delta_y=delta_y
        if isinstance(boundaries,Sequence):
            if len(boundaries)== 4:
                self.left_boundary=boundaries[0]
                self.right_boundary=boundaries[1]
                self.top_boundary=boundaries[2]
                self.bottom_boundary=boundaries[3]
            else:
                raise Exception("The length of boundaries need to be 4: '[left_boundary, right_boundary, top_boundary, bottom_boundary]'")          
        else:
            raise Exception("Boundaries need to be set as a sequence type of '[left_boundary, right_boundary, top_boundary, bottom_boundary]'")                  
        
        c_1=isinstance(self.left_boundary,PeriodicBoundary) and not isinstance(self.right_boundary,PeriodicBoundary)
        c_2=isinstance(self.right_boundary,PeriodicBoundary) and not isinstance(self.left_boundary,PeriodicBoundary)
        c_3=isinstance(self.top_boundary,PeriodicBoundary) and not isinstance(self.bottom_boundary,PeriodicBoundary)
        c_4=isinstance(self.bottom_boundary,PeriodicBoundary) and not isinstance(self.top_boundary,PeriodicBoundary)
        if c_1 or c_2 or c_3 or c_4:
            raise Exception("Periodic boundary should be set in pairs.")    
        
        self.set_obstacles(obstacles=obstacles)   
    
    def set_obstacles(self,obstacles):
        if isinstance(obstacles,Sequence):
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
                obstacles=_obstacles_add(self.obstacles,other.obstacles),
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

    def __mul__(self, other):
        if isinstance(other,Domain):
            return Domain(
                [
                    self.left_boundary*other.left_boundary,
                    self.right_boundary*other.right_boundary,
                    self.top_boundary*other.top_boundary,
                    self.bottom_boundary*other.bottom_boundary
                ],
                obstacles=_obstacles_mul(self.obstacles,other.obstacles),
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
  
    def __pow__(self, other):
        return Domain(
            [
                self.left_boundary**other,
                self.right_boundary**other,
                self.top_boundary**other,
                self.bottom_boundary**other
            ],
            obstacles=[obstacle**other for obstacle in self.obstacles],
            delta_x=self.delta_x,
            delta_y=self.delta_y
        )   

    def __truediv__(self, other):
        if isinstance(other,Domain):
            return Domain(
                [
                    self.left_boundary/other.left_boundary,
                    self.right_boundary/other.right_boundary,
                    self.top_boundary/other.top_boundary,
                    self.bottom_boundary/other.bottom_boundary
                ],
                obstacles=_obstacles_div(self.obstacles,other.obstacles),
                delta_x=self.delta_x,
                delta_y=self.delta_y
            )
        else:
            try:
                return Domain(
                    [
                        self.left_boundary/other,
                        self.right_boundary/other,
                        self.top_boundary/other,
                        self.bottom_boundary/other
                    ],
                    obstacles=[obstacle/other for obstacle in self.obstacles],
                    delta_x=self.delta_x,
                    delta_y=self.delta_y
                )
            except Exception:
                return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other,Domain):
            return Domain(
                [
                    other.left_boundary/self.left_boundary,
                    other.right_boundary/self.right_boundary,
                    other.top_boundary/self.top_boundary,
                    other.bottom_boundary/self.bottom_boundary
                ],
                obstacles=_obstacles_div(other.obstacles,self.obstacles),
                delta_x=self.delta_x,
                delta_y=self.delta_y
            )
        else:
            try:
                return Domain(
                    [
                        other/self.left_boundary,
                        other/self.right_boundary,
                        other/self.top_boundary,
                        other/self.bottom_boundary
                    ],
                    obstacles=[other/obstacle for obstacle in self.obstacles],
                    delta_x=self.delta_x,
                    delta_y=self.delta_y
                )
            except Exception:
                return NotImplemented

def UnconstrainedDomain(obstacles=[],delta_x=1,delta_y=1):
    """
    Create a domain with unconstrained boundary conditions for all boundaries.

    Args:
        obstacles (list, optional): _description_. Defaults to [].
        delta_x (int, optional): _description_. Defaults to 1.
        delta_y (int, optional): _description_. Defaults to 1.
        
    Returns:
        Domain (Domain): A domain object.
    """
    return Domain(boundaries=[UnConstrainedBoundary()]*4,obstacles=obstacles,delta_x=delta_x,delta_y=delta_y)

def PeriodicDomain(obstacles=[],delta_x=1,delta_y=1):
    """
    Create a domain with periodic boundary conditions for all boundaries.

    Args:
        obstacles (list, optional): _description_. Defaults to [].
        delta_x (int, optional): _description_. Defaults to 1.
        delta_y (int, optional): _description_. Defaults to 1.
        
    Returns:
        Domain (Domain): A domain object.

    """
    return Domain(boundaries=[PeriodicBoundary()]*4,obstacles=obstacles,delta_x=delta_x,delta_y=delta_y)