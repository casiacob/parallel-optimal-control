import jax.numpy as jnp
from jax import random, lax
from paroc.discretize import lti_disc_u
from paroc.lqt_problem import LQT


def gen_waypoints(N: int):
    """

    Args:
        N: number of waypoints

    Returns:
        wp: waypoints
    """
    seed = 123
    key = random.PRNGKey(seed)
    x0 = 0.0
    y0 = 0.0
    x1 = 10.0
    y1 = 10.0
    r = 1.0
    s = r / 2.0
    da = 0.5 * jnp.pi

    x_start = x0 + (x1 - x0) / 2.0
    y_start = y0 + (y1 - y0) / 2.0
    angle = 0.0

    noise = da * 2.0 * (random.uniform(key, shape=(N,), minval=0.0, maxval=1.0) - 0.5)

    def body(carry, inp):
        ang, prev_x, prev_y = carry
        ang = ang + inp
        ang = jnp.where(ang < 0.0, ang + 2.0 * jnp.pi, ang)
        ang = jnp.where(ang > 2.0 * jnp.pi, ang - 2.0 * jnp.pi, ang)

        new_x = prev_x + r * jnp.cos(ang)
        new_y = prev_y + r * jnp.sin(ang)

        new_x = jnp.where(
            jnp.logical_or(new_x <= x0 + s, new_x >= x1 - s),
            2.0 * prev_x - new_x,
            new_x,
        )
        new_y = jnp.where(
            jnp.logical_or(new_y <= y0 + s, new_y >= y1 - s),
            2.0 * prev_y - new_y,
            new_y,
        )

        return (ang, new_x, new_y), (new_x, new_y)

    _, (x, y) = lax.scan(body, (angle, x_start, y_start), xs=noise, length=N)
    x = jnp.hstack((x_start, x))
    y = jnp.hstack((y_start, y))
    wp = jnp.vstack((x, y))
    return wp


def generate_lqt(wp: jnp.ndarray, steps: int):
    dt = 1.0 / steps
    N = wp.shape[1]
    T = steps * N

    U = 0.1 * jnp.eye(2)
    Z = jnp.eye(U.shape[0])
    H = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    HT = jnp.eye(4)
    Xl = 100.0 * jnp.eye(2)
    Xn = 1e-6 * jnp.eye(2)
    XT = 1.0 * jnp.eye(4)
    s = jnp.zeros((T, U.shape[0]))

    F = jnp.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    Qc = jnp.diag(jnp.array([1, 1]))
    L = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    G = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    F, L, _ = lti_disc_u(F, L, G, Qc, dt)

    curr_r = wp[:, 0]
    x0 = jnp.array([curr_r[0], curr_r[1], 0.0, 0.0])

    def generate_reference_body(carry, inp):
        ref, i = carry
        k = inp
        XX = jnp.where(k % steps == 0, Xl, Xn)
        ref = jnp.where(k % steps == 0, wp[:, i], ref)
        i = jnp.where(k % steps == 0, i + 1, i)
        return (ref, i), (ref, XX)

    _, (r, X) = lax.scan(generate_reference_body, (curr_r, 0), jnp.arange(T))

    rT = jnp.array([r[-1, 0], r[-1, 0], 0.0, 0.0])
    c = jnp.zeros((T, x0.shape[0]))
    M = jnp.zeros((r.shape[1], U.shape[0]))

    F = jnp.kron(jnp.ones((T, 1, 1)), F)
    L = jnp.kron(jnp.ones((T, 1, 1)), L)
    U = jnp.kron(jnp.ones((T, 1, 1)), U)
    H = jnp.kron(jnp.ones((T, 1, 1)), H)
    Z = jnp.kron(jnp.ones((T, 1, 1)), Z)
    M = jnp.kron(jnp.ones((T, 1, 1)), M)

    lqt = LQT(F, L, c, XT, HT, rT, X, H, r, U, Z, s, M)
    return lqt, x0
