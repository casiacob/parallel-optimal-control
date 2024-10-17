## Jax implementation of parallel optimal control and constrained parallel optimal control

## Unconstrained parallel optimal control
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
Clone the repository
```
$ cd ..
$ git clone https://github.com/casiacob/ip-parallel-optimal-control.git
$ cd ip-parallel-optimal-control
$ pip install .
```
Constrained pendulum runtime example (requires GPU)
```
$ cd examples
$ python pendulum_runtime.py
```
Constrained cartpole runtime example (requires GPU)
```
$ cd examples
$ python cartpole_runtime.py
```

## ADMM
Clone the repository
```
$ cd ..
$ git clone https://github.com/casiacob/admm-parallel-optimal-control.git
$ cd admm-parallel-optimal-control
$ pip install .
```
Constrained pendulum runtime example (requires GPU)
```
$ cd examples
$ python pendulum_runtime.py
```
Constrained cartpole runtime example (requires GPU)
```
$ cd examples
$ python cartpole_runtime.py
```

