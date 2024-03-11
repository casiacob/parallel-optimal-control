from jax.scipy import linalg
import jax.numpy as jnp
from jax import lax


def lti_disc(F, L, Qc=None, dt=1.0):
    """Discretize a system of the form
          dx/dt = F x + L w,
        where w is a white noise with spectral density Qc to
          x[k+1] = A x[k] + w_k, w_k ~ N(0,Q)

    Parameters:
        F: State feedback matrix.
        L: Noise gain matrix.
        Qc: Noise covariance matrix (default identity), if scalar qc, then this is qc * eye(1).
        dt: Time step (default 1.0).

    Returns:
        A: Discrete-time state transition matrix.
        Q: Discrete-time process noise covariance matrix.
    """

    n = F.shape[0]
    Qc = jnp.where(Qc is None, jnp.eye(L.shape[1], dtype=F.dtype), Qc)
    Phi = jnp.zeros((2 * n, 2 * n), dtype=F.dtype)
    Phi = Phi.at[0:n, 0:n].set(F)
    Phi = Phi.at[0:n, n:].set(L @ Qc @ L.T)
    Phi = Phi.at[n : 2 * n, n:].set(-F.T)

    EPhi = linalg.expm(Phi * dt)
    A = EPhi[0:n, 0:n]
    Q = EPhi[0:n, n:] @ A.T

    return A, Q


def lti_disc_u(F, L, G=None, Qc=None, dt=1.0):
    """Discretize a system of the form (note that G is the input gain!)
          dx/dt = F x + G u + L w,
        where w is a white noise with spectral density Qc to
          x[k+1] = A x[k] + B u[k] + w_k, w_k ~ N(0,Q)

    Parameters:
        F: State feedback matrix.
        L: Noise gain matrix.
        G: Input gain matrix (default identity).
        Qc: Noise covariance matrix (default identity), if scalar qc, then this is qc * eye(1).
        dt: Time step (default 1.0).

    Returns:
        A: Discrete-time state transition matrix.
        B: Discrete-time input gain matrix.
        Q: Discrete-time process noise covariance matrix.
    """
    A, Q = lti_disc(F, L, Qc, dt)

    n = F.shape[0]
    # G = jnp.where(G is None, jnp.eye(n), G)

    m = G.shape[1]

    Psi = jnp.zeros((n + m, n + m), dtype=F.dtype)
    Psi = lax.dynamic_update_slice(Psi, F, (0, 0))
    Psi = lax.dynamic_update_slice(Psi, G, (0, n))

    EPsi = linalg.expm(Psi * dt)
    B = lax.dynamic_slice(EPsi, (0, n), (n, m))
    return A, B, Q
