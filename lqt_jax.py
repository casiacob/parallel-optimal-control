import jax.numpy as jnp
from jax.scipy import linalg
from jax import lax, vmap
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
    Aij, bij, Cij, etaij, Jij = elem1
    Ajk, bjk, Cjk, etajk, Jjk = elem2

    I = jnp.eye(Aij.shape[0], dtype=Aij.dtype)
    LU, piv = linalg.lu_factor(I + jnp.dot(Cij, Jjk))
    Aik = jnp.dot(Ajk, linalg.lu_solve((LU, piv), Aij))
    bik = jnp.dot(Ajk, linalg.lu_solve((LU, piv), bij + jnp.dot(Cij, etajk))) + bjk
    Cik = jnp.dot(Ajk, linalg.lu_solve((LU, piv), Cij @ Ajk.T)) + Cjk
    LU, piv = linalg.lu_factor(I + jnp.dot(Jjk, Cij))
    etaik = jnp.dot(Aij.T, linalg.lu_solve((LU, piv), etajk - jnp.dot(Jjk, bij))) + etaij
    Jik = jnp.dot(Aij.T, linalg.lu_solve((LU, piv), jnp.dot(Jjk, Aij))) + Jij
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


def par_backward_pass_scan(elems):
    """Perform LQT backward associative scan to the backward pass elements.

    Parameters:
        elems: list of tuples (A, b, C, eta, J)

    Returns:
        Reverse prefix sums as a list of tuples (A, b, C, eta, J)
    """
    return lax.associative_scan(jax.vmap(combine_abcej), elems, reverse=True)


def par_forward_pass_scan(elems):
    """Perform LQT forward associative scan to the backward pass elements.

    Parameters:
        elems: list of tuples (F, c)

    Returns:
        Forward prefix sums as a list of tuples (F, c)
    """
    return lax.associative_scan(jax.vmap(combine_fc), elems, reverse=False)


class LQT:
    """
    A class containing general LQT model parameters and methods for doing useful
    things with it. The model is

    x[k+1] = F[k] x[k] + c[k] + L[k] u[k]
      J(u) = E{ 1/2 (H[T] x[T] - r[T)].T X[T] (H[T] x[T] - r[T])
        + sum_{k=0}^{T-1} 1/2 (H[k] x[k] - r[k]).T X[k] (H[k] x[k] - r[k])
                        + 1/2 (Z[k] u[k] - s[k]).T U[k] (Z[k] u[k] - s[k])
                            + (H[k] x[k] - r[k]).T M[k] (Z[k] u[k] - s[k]) }

    """

    def __init__(self, F, L, X, U, XT, c, H, r, HT, rT, Z, s, M):
        """Create LQT from given matrices. No defaults are applied. See checkAndExpand for more flexible interface.

        Parameters:
            F: State transition matrices
            L: Control matrices
            X: State cost matrices
            U: Control cost matrices
            XT: Terminal state cost matrix
            c: State offsets
            H: State cost output matrices
            r: State cost output offsets (i.e., the reference trajectory)
            HT: Terminal state cost output matrix
            rT: Terminal state cost output offset
            Z: Control cost output matrices
            s: Control cost output offsets
            M: Cross term matrices
        """
        self.F = F
        self.L = L
        self.X = X
        self.U = U
        self.XT = XT

        self.c = c

        self.H = H
        self.r = r
        self.HT = HT
        self.rT = rT

        self.Z = Z
        self.s = s
        self.M = M

    def parBackwardPass_init(self):
        """Parallel LQT backward pass initialization.

        Returns:
            elems: List of tuples (A, b, C, eta, J) for 0:T.
        """

        def parBackwardPass_init_body(X, r):
            LU, piv = linalg.lu_factor(self.U @ self.Z)
            Ak = self.F - self.L @ linalg.lu_solve((LU, piv), self.M.T @ self.H)
            bk = (
                self.c
                + self.L @ linalg.lu_solve((LU, piv), self.M.T @ r)
                + self.L @ linalg.solve(self.Z, self.s)
            )
            Ck = self.L @ linalg.solve(
                self.Z.T @ self.U @ self.Z, self.L.T, assume_a="pos"
            )
            Y = X - self.M @ linalg.solve(self.U, self.M.T)
            etak = self.H.T @ Y @ r
            Jk = self.H.T @ Y @ self.H
            return Ak, bk, Ck, etak, Jk

        A, b, C, eta, J = vmap(parBackwardPass_init_body)(
            self.X,
            self.r,
        )
        AT = jnp.zeros_like(self.F)
        A = jnp.vstack((A, AT.reshape(1, A.shape[1], A.shape[2])))
        bT = jnp.zeros_like(self.F[:, 0])
        b = jnp.vstack((b, bT))
        CT = jnp.zeros_like(self.F)
        C = jnp.vstack((C, CT.reshape(1, C.shape[1], C.shape[2])))
        etaT = self.HT.T @ self.XT @ self.rT
        eta = jnp.vstack((eta, etaT))
        JT = self.HT.T @ self.XT @ self.HT
        J = jnp.vstack((J, JT.reshape(1, J.shape[1], J.shape[2])))
        jax.debug.breakpoint()
        elems = (A, b, C, eta, J)
        return elems

    def parBackwardPass_extract(self, elems):
        """Parallel LQT backward pass result extraction.

        Parameters:
            elems: List of tuples (A, b, C, eta, J) for 0:T.

        Returns:
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.
            S_list: List of value function matrices for 0:T.
            v_list: List of value function offsets for 0:T.
        """

        A_, b_, C_, eta_, J_ = elems
        eta0 = eta_[0]
        J0 = J_[0]

        def parBackwarPass_extract_body(
            elemts,
            r,
        ):
            A, b, C, eta, J = elemts
            S = J
            v = eta
            CF, low = linalg.cho_factor(self.Z.T @ self.U @ self.Z + self.L.T @ S @ self.L)
            Kx = linalg.cho_solve((CF, low), self.Z.T @ self.M.T @ self.H + self.L.T @ S @ self.F)
            d = linalg.cho_solve(
                (CF, low),
                self.Z.T @ self.U @ self.s + self.Z.T @ self.M.T @ r - self.L.T @ S @ self.c + self.L.T @ v,
            )
            return Kx, d, S, v

        Kx_list, d_list, S_list, v_list = vmap(parBackwarPass_extract_body)(
            (A_[1:], b_[1:], C_[1:], eta_[1:], J_[1:]),
            self.r
        )
        S_list = jnp.vstack((J0.reshape(1, J0.shape[0], J0.shape[1]), S_list))
        v_list = jnp.vstack((eta0, v_list))
        return Kx_list, d_list, S_list, v_list

    def parBackwardPass(self):
        """Parallel LQT backward pass.

        Returns:
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.
            S_list: List of value function matrices for 0:T.
            v_list: List of value function offsets for 0:T.
        """

        # Initialize
        elems = self.parBackwardPass_init()

        # Call the associative scan
        elems = par_backward_pass_scan(elems)

        # Extract the results
        return self.parBackwardPass_extract(elems)

    def parForwardPass_init(self, x0, Kx_list, d_list):
        """Parallel LQT forward pass initialization.

        Parameters:
            x0: Initial state.
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.

        Returns:
            elems: List of tuples (F, c) for 0:T.
        """

        tF0 = jnp.zeros_like(self.F)
        tc0 = (
            (self.F - self.L @ Kx_list[0]) @ x0
            + self.c
            + self.L @ d_list[0]
        )
        elems0 = (tF0, tc0)

        def parForwardPass_init_body(Kx, d):
            tF = self.F - self.L @ Kx
            tc = self.c + self.L @ d
            return tF, tc

        tF_, tc_ = vmap(parForwardPass_init_body)(Kx_list, d_list)
        tF_ = jnp.vstack((tF0.reshape(1, tF0.shape[0], tF0.shape[1]), tF_))
        tc_ = jnp.vstack((tc0, tc_))
        elems = (tF_, tc_)
        return elems

    def parForwardPass_extract(self, x0, Kx_list, d_list, elems):
        """Parallel LQT forward pass result extraction.

        Parameters:
            x0: Initial state.
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.
            elems: List of tuples (F, c) for 0:T-1.

        Returns:
            u_list: List of controls for 0:T-1.
            x_list: List of states for 0:T.
        """
        u0 = -Kx_list[0] @ x0 + d_list[0]


        def extract_x(el):
            _, tc = el
            x = tc
            return x

        def extract_u(tc, Kx, d):
            u = -Kx @ tc + d
            return u

        x_list = vmap(extract_x)(elems)
        x_list = jnp.vstack((x0, x_list))

        u_list = vmap(extract_u)(elems[1][:-2], Kx_list[1:], d_list[1:])
        u_list = jnp.vstack((u0, u_list))
        return u_list, x_list

    def parForwardPass(self, x0, Kx_list, d_list):
        """Parallel LQT forward pass.

        Parameters:
            x0: Initial state.
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.

        Returns:
            u_list: List of controls for 0:T-1.
            x_list: List of states for 0:T.
        """
        # Initialize
        elems = self.parForwardPass_init(x0, Kx_list, d_list)

        # Call the associative scan
        elems = par_forward_pass_scan(elems)

        # Extract the results
        return self.parForwardPass_extract(x0, Kx_list, d_list, elems)

    def cost(self, u_list, x_list):
        """Compute the cost of a trajectory.

        Parameters:
            u_list: List of control vectors for 0:T-1.
            x_list: List of state vectors for 0:T.

        Returns:
            res: Cost of the trajectory.
        """
        final_cost = (
            0.5
            * (self.HT @ x_list[-1] - self.rT).T
            @ self.XT
            @ (self.HT @ x_list[-1] - self.rT)
        )

        def stage_cost(H, x, r, X, Z, u, s, U, M):
            ln = 0.5 * (H @ x - r).T @ X @ (H @ x - r)
            ln += 0.5 * (Z @ u - s).T @ U @ (Z @ u - s)
            ln += (H @ x - r).T @ M @ (Z @ u - s)
            return ln

        stage_costs = vmap(stage_cost)(
            self.H, x_list[:-1], self.r, self.X, self.Z, u_list, self.s, self.U, self.M
        )

        return final_cost + jnp.sum(stage_costs)
