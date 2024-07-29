
### Available Boundaries
| Obstacle                            | Description |
| ------------------------------------------------------------| ---------- |
| `ConvDO.obstacles.DirichletObstacle`   | The value of the field on the obstacle is fixed at the boundary. |
| `ConvDO.obstacles.NeumannObstacle`   | The gradient of the field on the obstacle is fixed at the boundary. |
| `ConvDO.obstacles.UnConstrainedObstacle`   | The value of boundary is calculated by the value of the neighbour cells. If you are not sure about the boundary condition on obstacle, you can use `UnConstrainedObstacle`. |

### Calculation Rule of Obstacles

---

* Dirichlet(a) + Dirichlet(b) = Dirichlet(a+b)

* Dirichlet(a) * Dirichlet(b) = Dirichlet(a*b)

* Dirichlet(a) + float(b) = Dirichlet(a+b)

* Dirichlet(a) * float(b) = Dirichlet(a*b)

---

* Dirichlet(a) + Neumann(b) = UnConstrained

* Dirichlet(a) * Neumann(b) = UnConstrained

---

* Neumann(a) + Neumann(b) = Neumann(a+b)

* Neumann(a) * Neumann(b) = UnConstrained

* Neumann(a) + float(b) = Neumann(a)

* Neumann(a) * float(b) = Neumann(a*b)

---

* UnConstrained + Other = UnConstrained

* UnConstrained * Other = UnConstrained