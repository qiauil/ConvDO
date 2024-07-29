### General Operations
Operations are combination of residual operators for given PDEs.

A operation can be designed with inheriting the `FieldOperations` class which provide all the available operators:

::: ConvDO.operations.FieldOperations

### Pre-defined Operations

We also give some pre-defined operations which mainly focus on the flow dynamics:

::: ConvDO.operations.TransientNS
::: ConvDO.operations.TransientNSWithForce
::: ConvDO.operations.PoissonDivergence
::: ConvDO.operations.PoissonDivergenceWithForce