
### Available Boundaries
| Boundary                            | Description |
| ------------------------------------------------------------| ---------- |
| `ConvDO.boundaries.DirichletBoundary`   | The value of the field is fixed at the boundary. |
| `ConvDO.boundaries.NeumannBoundary`   | The gradient of the field is fixed at the boundary. |
| `ConvDO.boundaries.UnConstrainedBoundary`   | The value of boundary is calculated by the value of the neighbour cells. If you are not sure about the boundary condition, you can use `UnConstrainedBoundary`. |
| `ConvDO.boundaries.PeriodicBoundary`   | The value of boundary is determined by the opposite side of the domain |

### Calculation Rule of Boundaries

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

---

* Periodic + Periodic = Periodic

* Periodic * Periodic = Periodic

* Periodic + Other = UnConstrained

* Periodic * Other = UnConstrained