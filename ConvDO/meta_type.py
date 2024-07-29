#usr/bin/python3
# -*- coding: UTF-8 -*-
import os

class CommutativeValue():
    """
    A class to represent a commutative value where a+b=b+a,a*b=b*a, a-b=-b+a.
    """
    #a+b=b+a,a*b=b*a, a-b=-b+a
    def __radd__(self, other):
        return self+other
    
    def __iadd__(self,other):
        return self+other

    def __sub__(self,other):
        try:
            return self+(-1*other)
        except Exception:
            return NotImplemented

    def __rsub__(self,other):
        try:
            return other+(-1*self)
        except Exception:
            return NotImplemented
        
    def __isub__(self,other):
        return self-other    
    
    def __rmul__(self,other):
        return self*other       

    def __imul__(self,other):
        return self*other
    

class VectorValue():
    """
    A class to represent a 2D vector.
    Supports basic operations such as addition, subtraction, multiplication, division, and dot product.
    The vector is represented as a tuple (`ux`,`uy`), where `ux` and `uy` are the x and y components of the vector, respectively.
    The x/y components of the vector can be any type that supports the basic operations.
    
    Args:
        ux (Any): x-component of the vector
        uy (Any): y-component of the vector
    """
    
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
            except Exception:
                return NotImplemented

    def __radd__(self, other):
        return self+other
    
    def __iadd__(self,other):
        return self+other

    def __sub__(self,other):
        try:
            return self+(-1*other)
        except Exception:
            return NotImplemented

    def __rsub__(self,other):
        try:
            return other+(-1*self)
        except Exception:
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
            except Exception:
                return NotImplemented  
            
    def __rmul__(self,other):
        try:
            return VectorValue(self.ux*other,self.uy*other)
        except Exception:
            return NotImplemented         

    def __imul__(self,other):
        return self*other
    
    def __pow__(self, other):
        if isinstance(other,int) and other>1:
            return_value=self*self
            for i in range(other-2):
                return_value=return_value*self
            return return_value
        else:
            raise NotImplementedError("Operation not supported")

    # @ (dot product): vector@vector, vector@tensor
    
    def __matmul__(self,other):
        if isinstance(other,VectorValue):
            return self.ux*other.ux+self.uy*other.uy
        if isinstance(other,TensorValue):
            return VectorValue(self.ux*other.uxx+self.uy*other.uxy,self.ux*other.uyx+self.uy*other.uyy)
        
    def __truediv__(self, other):
        try:
            return VectorValue(self.ux/other,self.uy/other)
        except Exception:
            return NotImplemented
            
class TensorValue():
    """
    A class to represent a 2D tensor.
    """

    def __init__(self,uxx,uyx,uxy,uyy) -> None:
        self.uxx=uxx
        self.uxy=uxy
        self.uyx=uyx
        self.uyy=uyy
        
    def __str__(self) -> str:
        return "({},{}{}{},{})".format(self.uxx,self.uyx,os.linesep,self.uxy,self.uyy)  