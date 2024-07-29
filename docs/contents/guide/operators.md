### Available Operators
Operators are the core of `ConvDO`, the spatial derivate with corresponding code representation is listed as follows:

| Operation                            | Operator initialization   | Code of operation |
| ------------------------------------------------------------ | -------------------------------- | ---------- |
| $\frac{\partial p }{ \partial x}$                            | `grad_x=ConvGrad(direction="x")`   | `grad_x * p` |
| $\frac{\partial p }{ \partial y}$                            | `grad_x=ConvGrad(direction="y")`   | `grad_y * p` |
| $\nabla p = (\frac{\partial p }{ \partial p},\frac{\partial p }{ \partial y})$ | `nabla=ConvNabla()`                | `nabla * p`  |
| $\nabla \cdot \mathbf{u}=\frac{\partial u_x }{ \partial x}+\frac{\partial u_y }{ \partial y}$ | `nabla=ConvNabla()`                | `nabla @ u`  |
| $\frac{\partial^2 p }{ \partial x^2}$                        | `grad2_x=ConvGrad2(direction="x")` | `grad2_x * p` |
| $\frac{\partial^2 p }{ \partial y^2}$                        | `grad2_x=ConvGrad2(direction="y")` | `grad2_y * p` |
| $\nabla^2 p = (\frac{\partial^2 p }{ \partial x^2},\frac{\partial^2 p }{ \partial y^2})$ | `nabla2=ConvLaplacian()`  | `nabla * p`  |
|$\nabla \cdot (\nabla \mathbf{u}) = (\frac{\partial u_x }{ \partial x}+\frac{\partial u_x }{ \partial y},\frac{\partial u_y }{ \partial x}+\frac{\partial u_y }{ \partial y})$|`nabla2=ConvLaplacian()`|`nabla * u`|

**Note:** If you use `ConvDO.operations.FieldOperations`, the name of the operators is unchanged, e.g., the corresponding operator of `grad_x` is `self.grad_x`.

### API Guide of Operators

::: ConvDO.conv_operators.ConvGrad

::: ConvDO.conv_operators.ConvNabla

::: ConvDO.conv_operators.ConvGrad2

::: ConvDO.conv_operators.ConvLaplacian