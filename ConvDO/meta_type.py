#usr/bin/python3
# -*- coding: UTF-8 -*-
import os

class CommutativeValue():
    #a+b=b+a,a*b=b*a
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
    
    def __rmul__(self,other):
        return self*other       

    def __imul__(self,other):
        return self*other

class VectorValue():
    
    def __init__(self,ux,uy) -> None:
        self.ux=ux
        self.uy=uy
 
    def __str__(self) -> str:
        return "({},{})".format(self.ux,self.uy)
    
    # + ： vector+scalar,vector+vector    
    def __add__(self, other):
        if isinstance(other,VectorValue):
            return VectorValue(self.ux+other.ux,self.uy+other.uy)
        else:
            try:
                return VectorValue(self.ux+other,self.uy+other)
            except TypeError:
                return NotImplemented

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

    # * ： vectorvector, scalarvector            
    def __mul__(self,other):
        if isinstance(other,VectorValue):
            return TensorValue(self.ux*other.ux,self.ux*other.uy,self.uy*other.ux,self.uy*other.uy)
        else:
            try:
                return VectorValue(self.ux*other,self.uy*other)
            except TypeError:
                return NotImplemented  
            
    def __rmul__(self,other):
        try:
            return VectorValue(self.ux*other,self.uy*other)
        except TypeError:
            return NotImplemented         

    def __imul__(self,other):
        return self*other

    # @ (dot product): vector@vector, vector@tensor
    
    def __matmul__(self,other):
        if isinstance(other,VectorValue):
            return self.ux*other.ux+self.uy*other.uy
        if isinstance(other,TensorValue):
            return VectorValue(self.ux*other.uxx+self.uy*other.uxy,self.ux*other.uyx+self.uy*other.uyy)
            
class TensorValue():

    def __init__(self,uxx,uyx,uxy,uyy) -> None:
        self.uxx=uxx
        self.uxy=uxy
        self.uyx=uyx
        self.uyy=uyy
        
    def __str__(self) -> str:
        return "({},{}{}{},{})".format(self.uxx,self.uyx,os.linesep,self.uxy,self.uyy)  