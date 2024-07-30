<h1 align="center">
  <img src="./docs/assets/logo/ConvDO.png" width="100"/>
  <br>ConvDO<br>
</h1>
<h4 align="center">Convolutional Differential Operators for Physics-based Deep Learning Study</h4>
<h6 align="center">Calculate the spatial derivative differentiablly!</h6>
<p align="center">
  [<a href="https://qiauil.github.io/ConvDO/">📖 Documentation & Examples</a>]
</p>

## Installation

* Install through pip: `pip install ConvDO`
* Install the latest version through pip: `pip install git+https://github.com/qiauil/ConvDO`
* Install locally: Download the repository and run `./install.sh` or `pip install .`

## Feature

Positive😀 and negative🙃 things are all features... 

* PyTorch-based and only supports 2D fields at the moment.
* Powered by convolutional neural network.
* Differentiable and GPU supported (why not? It's PyTorch based!).
* Second order for Dirichlet and Neumann boundary condition.
* Up to 8th order for periodic boundary condition.
* Obstacles inside of the domain is supported.

## Documentations

Check 👉 [here](https://qiauil.github.io/ConvDO/)

## Further Reading

Projects using `ConvDO`:

* [Diffusion-based-Flow-Prediction](https://github.com/tum-pbs/Diffusion-based-Flow-Prediction): Diffusion-based flow prediction (DBFP) with uncertainty for airfoils.
* To be updated... 

If you need to solve more complex PDEs using differentiable functions, please have a check on

* [PhiFlow](https://github.com/tum-pbs/PhiFlow): A differentiable PDE solving framework for machine learning
* [Exponax](https://github.com/Ceyron/exponax): Efficient Differentiable n-d PDE solvers in JAX.

For more research on physics based deep learning research, please visit the website of [our research group at TUM](https://ge.in.tum.de/publications/).
