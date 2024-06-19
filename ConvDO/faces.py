#usr/bin/python3
# -*- coding: UTF-8 -*-
from . import *

class DirichletFace():
    
    def __init__(self,face_value) -> None:
        self.face_value=face_value
        
    def correct_inward_face(self,faces):
        return torch.ones_like(faces)*self.face_value

    def correct_outward_face(self,faces):
        return torch.ones_like(faces)*self.face_value

class NeumannFace():
    
    def __init__(self,face_gradient) -> None:
        self.face_gradient=face_gradient

    def correct_inward_face(self,boundary_cells,delta):
        return (2*boundary_cells-self.face_gradient*delta)/2

    def correct_outward_face(self,boundary_cells,delta):
        return (2*boundary_cells+self.face_gradient*delta)/2       
  
class UnConstrainedFace():

    def correct_inward_face(self,boundary_cells,boundary_cells_neighbour,boundary_cells_neighbour_neighbour):
        return (4*boundary_cells-3*boundary_cells_neighbour+boundary_cells_neighbour_neighbour)/2

    def correct_outward_face(self,boundary_cells,boundary_cells_neighbour,boundary_cells_neighbour_neighbour):
        return (4*boundary_cells-3*boundary_cells_neighbour+boundary_cells_neighbour_neighbour)/2   