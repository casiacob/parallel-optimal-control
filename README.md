## Jax implementation of parallel linear quadratic tracker

## Instructions for installing
Clone the repository:

```
$ git clone https://github.com/casiacob/parallel-optimal-control.git
```

Create conda environment:
```
$ conda create --name paroc python=3.10
$ conda activate paroc
$ cd parallel-optimal-control
$ pip install .
```
Two constrained versions are available at

[interior point parallel optimal control](https://github.com/casiacob/ip-parallel-optimal-control)

[admm parallel optimal control](https://github.com/casiacob/admm-parallel-optimal-control)

## Interior Point 

```
$ cd parallel-constrained-optimal-control
$ git clone https://github.com/casiacob/ip-parallel-optimal-control.git
$ cd ip-parallel-optimal-control
$ pip install .
```
Pendulum runtime example
```
$ cd examples
$ python pendulum_runtime.py
```
Cartpole runtime example
```
$ cd examples
$ python cartpole_runtime.py
```
