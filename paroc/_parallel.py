import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import lax, vmap
from paroc.lqt_problem import LQT
import jax

def combine_abcej(elem1, elem2):
    """
    Combine two conditional value functions in backward pass of parallel LQT
    Parameters:
        Aij, bij, Cij, etaij, Jij: parameters of conditional value function V_{i->j}{x_i, x_j}
        Ajk, bjk, Cjk, etajk, Jjk: parameters of conditional value function V_{j->k}{x_j, x_k}

    Returns:
        Aik, bik, Cik, etaik, Jik: parameters of conditional value function V_{i->k}{x_i, x_k}
    """
    Ajk, bjk, Cjk, etajk, Jjk = elem1
    Aij, bij, Cij, etaij, Jij = elem2

    I = jnp.eye(Aij.shape[0], dtype=Aij.dtype)
    Aik = jnp.dot(Ajk, jlinalg.solve(I + jnp.dot(Cij, Jjk), Aij))
    bik = jnp.dot(Ajk, jlinalg.solve(I + jnp.dot(Cij, Jjk), bij + jnp.dot(Cij, etajk))) + bjk
    Cik = jnp.dot(Ajk, jlinalg.solve(I + jnp.dot(Cij, Jjk), jnp.dot(Cij, Ajk.T))) + Cjk
    etaik = jnp.dot(Aij.T, jlinalg.solve(I + jnp.dot(Jjk, Cij), etajk - jnp.dot(Jjk, bij))) + etaij
    Jik = jnp.dot(Aij.T, jlinalg.solve(I + jnp.dot(Jjk, Cij), jnp.dot(Jjk, Aij))) + Jij
    return Aik, bik, Cik, etaik, Jik


def combine_fc(elem1, elem2):
    """Combine two functions in forward pass of parallel LQT.

    Parameters:
        Fij, cij: parameters of function f_{i->j}(x_i)
        Fjk, cjk: parameters of function f_{j->k}(x_j)

    Returns:
        Fik, cik: parameters of function f_{i->k}(x_i)
    """
    Fij, cij = elem1
    Fjk, cjk = elem2

    Fik = Fjk @ Fij
    cik = Fjk @ cij + cjk
    return Fik, cik


def par_bwd_pass_scan(elems):
    """Perform LQT backward associative scan to the backward pass elements.

    Parameters:
        elems: list of tuples (A, b, C, eta, J)

    Returns:
        Reverse prefix sums as a list of tuples (A, b, C, eta, J)
    """
    return lax.associative_scan(vmap(combine_abcej), elems, reverse=True)


def par_fwd_pass_scan(elems):
    """Perform LQT forward associative scan to the backward pass elements.

    Parameters:
        elems: list of tuples (F, c)

    Returns:
        Forward prefix sums as a list of tuples (F, c)
    """
    return lax.associative_scan(vmap(combine_fc), elems)


def par_bwd_pass_init(ocp: LQT):
    """Parallel LQT backward pass initialization

    Args:
        ocp: optimal control problem in LQT format

    Returns:
        elems: List of tuples (A, b, C, eta, J) for 0:T.
    """

    def body(F, L, c, X, H, r, U, Z, s, M):
        Ak = F - L @ jlinalg.solve(U @ Z, M.T @ H)
        bk = c + L @ jlinalg.solve(U @ Z, M.T @ r) + L @ jnp.linalg.solve(Z, s)
        Ck = L @ jlinalg.solve(Z.T @ U @ Z, L.T)
        Y = X - M @ jlinalg.solve(U, M.T)
        etak = H.T @ Y @ r
        Jk = H.T @ Y @ H
        return Ak, bk, Ck, etak, Jk

    A, b, C, eta, J = vmap(body)(
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
    )

    AT = jnp.zeros_like(ocp.F[0])
    A = jnp.vstack((A, AT.reshape(1, A.shape[1], A.shape[2])))

    bT = jnp.zeros_like(ocp.F[0][:, 0])
    b = jnp.vstack((b, bT))

    CT = jnp.zeros_like(ocp.F[0])
    C = jnp.vstack((C, CT.reshape(1, C.shape[1], C.shape[2])))

    etaT = ocp.HT.T @ ocp.XT @ ocp.rT
    eta = jnp.vstack((eta, etaT))

    JT = ocp.HT.T @ ocp.XT @ ocp.HT
    J = jnp.vstack((J, JT.reshape(1, J.shape[1], J.shape[2])))

    elems = (A, b, C, eta, J)
    return elems


def par_bwd_pass_extract(ocp: LQT, elems):
    """
    
    Args:
        ocp: optimal control problem in LQT format
        elems: List of tuples (A, b, c, eta, J) for 0:T

    Returns:
        Kx_array: array of control gains
        d_array: array of feed forward gains
        S_array: array of value function hessians
        v_array: array of value function gradients
        predicted_reduction: sum of value function constant terms
        convex_problem: flag for action-value function positive definite Hessians
    """
    A0T, b0T, C0T, eta0T, J0T = elems
    eta0 = eta0T[0]
    J0 = J0T[0]

    def extract(elemsk, F, L, c, H, r, U, Z, s, M):
        A, b, C, eta, J = elemsk
        S = J
        v = eta
        # check if Hessian is positive definite
        Hess = Z.T @ U @ Z + L.T @ S @ L
        Hess = 0.5 * (Hess.T + Hess)
        eigv, _ = jlinalg.eigh(Z.T @ U @ Z + L.T @ S @ L)
        pos_def = jnp.all(eigv > 0)
        # Kx = jlinalg.solve(Hess, Z.T @ M.T @ H + L.T @ S @ F)
        # d = jlinalg.solve(
        #     Hess,
        #     Z.T @ U @ s + Z.T @ M.T @ r - L.T @ S @ c + L.T @ v,
        # )
        # # predicted cost reduction
        # z = (
        #         -d.T @ (Z.T @ U @ s + Z.T @ M.T @ r - L.T @ S @ c + L.T @ v)
        #         + 0.5 * d.T @ (Z.T @ U @ Z + L.T @ S @ L) @ d
        # )
        def hessian_is_pos_def(operands):
            Hess_pd, Z_pd, M_pd, H_pd, L_pd, S_pd, F_pd, U_pd, s_pd, r_pd, c_pd, v_pd = operands
            Kx_pd = jlinalg.solve(Hess_pd, Z_pd.T @ M_pd.T @ H_pd + L_pd.T @ S_pd @ F_pd, assume_a='pos')
            d_pd = jlinalg.solve(
                Hess_pd,
                Z_pd.T @ U_pd @ s_pd + Z_pd.T @ M_pd.T @ r_pd - L_pd.T @ S_pd @ c_pd + L_pd.T @ v_pd,
                assume_a='pos'
            )
            z_pd = (
                    -d_pd.T @ (Z_pd.T @ U_pd @ s_pd + Z_pd.T @ M_pd.T @ r_pd - L_pd.T @ S_pd @ c_pd + L_pd.T @ v_pd)
                    + 0.5 * d_pd.T @ (Z_pd.T @ U_pd @ Z_pd + L_pd.T @ S_pd @ L_pd) @ d_pd
            )
            return Kx_pd, d_pd, z_pd

        def hessian_is_not_pos_def(operands):
            Hess_pd, _, _, _, _, _, F_pd, _, _, _, _, _ = operands
            return jnp.zeros((Hess_pd.shape[0], F_pd.shape[0])), jnp.zeros((Hess_pd.shape[0], )), 1.

        Kx, d, z = jax.lax.cond(pos_def, hessian_is_pos_def, hessian_is_not_pos_def, (Hess, Z, M, H, L, S, F, U, s, r, c, v))
        return Kx, d, S, v, z, pos_def

    Kx_array, d_array, S_array, v_array, z_array, pos_def_array = vmap(
        extract
    )(
        (A0T[1:], b0T[1:], C0T[1:], eta0T[1:], J0T[1:]),
        ocp.F,
        ocp.L,
        ocp.c,
        ocp.H,
        ocp.r,
        ocp.U,
        ocp.Z,
        ocp.s,
        ocp.M,
    )

    S_array = jnp.vstack((J0.reshape(1, J0.shape[0], J0.shape[1]), S_array))
    v_array = jnp.vstack((eta0, v_array))
    predicted_reduction = jnp.sum(z_array)
    convex_problem = jnp.all(pos_def_array)
    return Kx_array, d_array, S_array, v_array, predicted_reduction, convex_problem


def par_bwd_pass(ocp: LQT):
    """Parallel LQT backward pass-

    Args:
        ocp: optimal control problem in LQT format

    Returns:
        Kx: array of control gains
        d: array of feed-forward gains
        S: value function hessians
        v: vlaue function gradients
        dV: predicted cost reduction
        convex: flag that states if the linearized problem is convex
    """

    # Initialize
    elems = par_bwd_pass_init(ocp)

    # Call the associative scan
    elems = par_bwd_pass_scan(elems)

    # extract and return the result
    Kx, d, S, v, dV, convex = par_bwd_pass_extract(ocp, elems)

    return Kx, d, S, v, dV, convex


def par_fwd_pass_init(ocp: LQT, x0: jnp.ndarray, Kx_array: jnp.ndarray, d_array: jnp.ndarray):
    """Parallel LQT forward pass initialization

    Args:
        ocp: optimal control problem in LQT format
        x0: initial state
        Kx_array: array of control gains for 0:T-1
        d_array: array of feed-forward gains 0:T-1

    Returns:
        elems: liast of tuples (F, c) for 0:T
    """

    tF0 = jnp.zeros_like(ocp.F[0])
    tc0 = (
            (ocp.F[0] - ocp.L[0] @ Kx_array[0]) @ x0
            + ocp.c[0]
            + ocp.L[0] @ d_array[0]
    )

    def init(Kx, d, F, L, c):
        return F - L @ Kx, c + L @ d

    tF, tc = vmap(init)(
        Kx_array[1:], d_array[1:], ocp.F[1:], ocp.L[1:], ocp.c[1:]
    )

    tF = jnp.vstack((tF0.reshape(1, tF0.shape[0], tF0.shape[1]), tF))
    tc = jnp.vstack((tc0, tc))
    elems = (tF, tc)
    return elems


def par_fwd_pass_extract(x0: jnp.ndarray, Kx_array: jnp.ndarray, d_array: jnp.ndarray, elems):
    """Parallel LQT forward pass result extraction

    Args:
        x0: Initial state
        Kx_array: control gain array for 0:T-1
        d_array: control feed-forward gain for 0:T-1
        elems: List of tuples (F, c) for 0:T-1

    Returns:

    """
    u0 = -Kx_array[0] @ x0 + d_array[0]

    def extract_x(e):
        _, tc = e
        x = tc
        return x

    def extract_u(tc, Kx, d):
        u = -Kx @ tc + d
        return u

    x_array = vmap(extract_x)(elems)
    x_array = jnp.vstack((x0, x_array))
    u_array = vmap(extract_u)(elems[1][:-1], Kx_array[1:], d_array[1:])
    u_array = jnp.vstack((u0, u_array))
    return u_array, x_array

def par_fwd_pass(ocp: LQT, x0: jnp.ndarray, Kx_array: jnp.ndarray, d_array: jnp.ndarray):
    """Parallel LQT forward pass

    Args:
        ocp: optimal control problem in LQT format
        x0: initial state
        Kx_array: control gain array for 0:T-1
        d_array: control feed-forward gain array for 0:T-1

    Returns:
        u_array: control sequence for 0:T-1
        x_array: state trajectory for 0:T
    """

    # Initialize
    elems = par_fwd_pass_init(ocp, x0, Kx_array, d_array)

    # Call the associative scan
    elems = par_fwd_pass_scan(elems)

    # Extract the result
    u_array, x_array = par_fwd_pass_extract(x0, Kx_array, d_array, elems)

    return u_array, x_array



