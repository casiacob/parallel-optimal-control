from paroc.lqt_problem import LQT
from jax import lax
import jax.scipy as jcp
import jax.numpy as jnp

def seq_bwd_pass(ocp: LQT):
    """Sequential backward pass to compute control laws and value functions.

    Args:
        ocp: optimal control problem in LQT format

    Returns:
        Kx_array: Array of control gains for 0:T-1.
        d_array: List of control offsets for 0:T-1.
        S_list: List of value function matrices for 0:T.
        v_list: List of value function offsets for 0:T.
    """

    ST = ocp.HT.T @ ocp.XT @ ocp.HT
    vT = ocp.HT.T @ ocp.XT @ ocp.rT

    def bwd_step(carry, inp):
        S, v = carry
        F, L, c, X, H, r, U, Z, s, M = inp
        CF, low = jcp.linalg.cho_factor(Z.T @ U @ Z + L.T @ S @ L)
        Kx = jcp.linalg.cho_solve((CF, low), Z.T @ M.T @ H + L.T @ S @ F)
        d = jcp.linalg.cho_solve(
            (CF, low), Z.T @ U @ s + Z.T @ M.T @ r - L.T @ S @ c + L.T @ v
        )
        v = (
                H.T @ X @ r
                - Kx.T @ Z.T @ U @ s
                + H.T @ M @ s
                - Kx.T @ Z.T @ M.T @ r
                + (F - L @ Kx).T @ (v - S @ c)
        )
        S = H.T @ X @ H - H.T @ M @ Z @ Kx + F.T @ S @ (F - L @ Kx)
        return (S, v), (Kx, d, S, v)

    _, (Kx_array, d_array, S_array, v_array) = lax.scan(
        bwd_step,
        (ST, vT),
        (
            ocp.F,
            ocp.L,
            ocp.c,
            ocp.X,
            ocp.H,
            ocp.r,
            ocp.U,
            ocp.Z,
            ocp.s,
            ocp.M,
        ),
        reverse=True,
    )
    S_array = jnp.vstack((S_array, ST.reshape(1, ST.shape[0], ST.shape[0])))
    v_array = jnp.vstack((v_array, vT))
    return Kx_array, d_array, S_array, v_array


def seq_fwd_pass(ocp: LQT, x0: jnp.ndarray, Kx_array: jnp.ndarray, d_array: jnp.ndarray):
    """
    Args:
        ocp: optimal control problem in LQT format
        x0: initial state
        Kx_array: control gains
        d_array:  control feed forward gains

    Returns:
        u_array: optimal control sequence
        x_array: optimal state trajectory
    """

    def fwd_step(carry, inp):
        x = carry
        Kx, d, F, L, c = inp
        u = -Kx @ x + d
        x = F @ x + c + L @ u
        return x, (u, x)

    _, (u_array, x_array) = lax.scan(
        fwd_step, x0, (Kx_array, d_array, ocp.F, ocp.L, ocp.c)
    )
    x_array = jnp.vstack((x0, x_array))
    return u_array, x_array


def seq_state_rollout(ocp: LQT, x0: jnp.ndarray, u_array: jnp.ndarray):
    """ Sequential state rollout given initial state and control sequence
    Args:
        ocp: optimal control problem in LQT format
        x0: initial state
        u_array: control sequence

    Returns:
        x_array: state trajectory
    """

    def body(carry, inp):
        x = carry
        u, F, L, c = inp
        x = F @ x + c + L @ u
        return x, x

    _, x_array = lax.scan(body, x0, (u_array, ocp.F, ocp.L, ocp.c))
    x_array = jnp.vstack((x0, x_array))

    return x_array
