from typing import NamedTuple
import jax.numpy as jnp

class LQT(NamedTuple):
    """
    x[k+1] = F[k] x[k] + c[k] + L[k] u[k]
      J(u) = E{ 1/2 (H[T] x[T] - r[T)].T X[T] (H[T] x[T] - r[T])
        + sum_{k=0}^{T-1} 1/2 (H[k] x[k] - r[k]).T X[k] (H[k] x[k] - r[k])
                        + 1/2 (Z[k] u[k] - s[k]).T U[k] (Z[k] u[k] - s[k])
                            + (H[k] x[k] - r[k]).T M[k] (Z[k] u[k] - s[k]) }
    """
    F: jnp.ndarray   # state transition matrix
    L: jnp.ndarray   # control matrix
    c: jnp.ndarray   # state offset
    XT: jnp.ndarray  # final cost penalty matrix
    HT: jnp.ndarray  # final cost output matrix (measurement model)
    rT: jnp.ndarray  # final state reference
    X: jnp.ndarray   # state penalty matrix
    H: jnp.ndarray   # state cost output matrix (measurement model)
    r: jnp.ndarray   # state reference
    U: jnp.ndarray   # control penalty matrix
    Z: jnp.ndarray   # control cost output matrix
    s: jnp.ndarray   # control reference
    M: jnp.ndarray   # cross term cost




