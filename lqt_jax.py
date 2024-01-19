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
    Aij, bij, Cij, etaij, Jij = elem2
    Ajk, bjk, Cjk, etajk, Jjk = elem1

    I = jnp.eye(Aij.shape[0], dtype=Aij.dtype)
    LU, piv = linalg.lu_factor(I + Cij @ Jjk)
    Aik = Ajk @ linalg.lu_solve((LU, piv), Aij)
    bik = Ajk @ linalg.lu_solve((LU, piv), bij + Cij @ etajk) + bjk
    Cik = Ajk @ linalg.lu_solve((LU, piv), Cij @ Ajk.T) + Cjk
    LU, piv = linalg.lu_factor(I + Jjk @ Cij)
    etaik = Aij.T @ linalg.lu_solve((LU, piv), etajk - Jjk @ bij) + etaij
    Jik = Aij.T @ linalg.lu_solve((LU, piv), Jjk @ Aij) + Jij
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

    #######################################################################
    # Sequential computation of gains, value functions, states, and controls
    #######################################################################
    def seqBackwardPass(self):
        """Sequential backward pass to compute control laws and value functions.

        Returns:
             Kx_array: Array of control gains for 0:T-1.
             d_array: List of control offsets for 0:T-1.
             S_list: List of value function matrices for 0:T.
             v_list: List of value function offsets for 0:T.
        """
        ST = self.HT.T @ self.XT @ self.HT
        vT = self.HT.T @ self.XT @ self.rT

        def bwd_step(carry, inp):
            S, v = carry
            F, L, X, U, c, H, r, Z, s, M = inp
            CF, low = linalg.cho_factor(Z.T @ U @ Z + L.T @ S @ L)
            Kx = linalg.cho_solve((CF, low), Z.T @ M.T @ H + L.T @ S @ F)
            d = linalg.cho_solve(
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

        _, bwd_pass_out = lax.scan(
            bwd_step,
            (ST, vT),
            (
                self.F,
                self.L,
                self.X,
                self.U,
                self.c,
                self.H,
                self.r,
                self.Z,
                self.s,
                self.M,
            ),
            reverse=True,
        )
        Kx_array, d_array, S_array, v_array = bwd_pass_out
        S_array = jnp.vstack((S_array, ST.reshape(1, ST.shape[0], ST.shape[0])))
        v_array = jnp.vstack((v_array, vT))
        return Kx_array, d_array, S_array, v_array

    def seqForwardPass(self, x0, Kx_array, d_array):
        """
        Args:
            x0: Initial state.
            Kx_array: Array of control gains for 0:T-1
            d_array: Array of control offsets for 0:T-1

        Returns:
            u_array: Array of controls for 0:T-1
            x_array: Array of states for 0:T.
        """

        def fwd_step(carry, inp):
            x = carry
            Kx, d, F, c, L = inp
            u = -Kx @ x + d
            x = F @ x + c + L @ u
            return x, (u, x)

        _, fwd_pass_out = lax.scan(fwd_step, x0, (Kx_array, d_array, self.F, self.c, self.L))
        u_array, x_array = fwd_pass_out
        x_array = jnp.vstack((x0, x_array))
        return u_array, x_array

    def seqSimulation(self, x0, u_array):
        """ Sequential simulation of the system

        Args:
            x0: Initial state.
            u_array: Array of controls for 0:T-1

        Returns:
            x_array: Array of states for 0:T
        """

        def body(carry, inp):
            x = carry
            u, F, c, L = inp
            x = F @ x + c + L @ u
            return x, x

        _, x_array = lax.scan(body, x0, (u_array, self.F, self.c, self.L))
        x_array = jnp.vstack((x0, x_array))

        return x_array

    #######################################################################
    # Parallel computation of gains, value functions, states, and controls
    #######################################################################

    def parBackwardPass_init(self):
        """Parallel LQT backward pass initialization.

        Returns:
            elems: List of tuples (A, b, C, eta, J) for 0:T.
        """

        def parBackwardPass_init_body(F, L, X, U, c, H, r, Z, s, M):
            LU, piv = linalg.lu_factor(U @ Z)
            Ak = F - L @ linalg.lu_solve((LU, piv), M.T @ H)
            bk = c + L @ linalg.lu_solve((LU, piv), M.T @ r) + L @ linalg.solve(Z, s)
            Ck = L @ linalg.solve(Z.T @ U @ Z, L.T, assume_a="pos")
            Y = X - M @ linalg.solve(U, M.T)
            etak = H.T @ Y @ r
            Jk = H.T @ Y @ H
            return Ak, bk, Ck, etak, Jk

        A, b, C, eta, J = vmap(parBackwardPass_init_body)(
            self.F,
            self.L,
            self.X,
            self.U,
            self.c,
            self.H,
            self.r,
            self.Z,
            self.s,
            self.M,
        )
        AT = jnp.zeros_like(self.F[0])
        A = jnp.vstack((A, AT.reshape(1, A.shape[1], A.shape[2])))
        bT = jnp.zeros_like(self.F[0][:, 0])
        b = jnp.vstack((b, bT))
        CT = jnp.zeros_like(self.F[0])
        C = jnp.vstack((C, CT.reshape(1, C.shape[1], C.shape[2])))
        etaT = self.HT.T @ self.XT @ self.rT
        eta = jnp.vstack((eta, etaT))
        JT = self.HT.T @ self.XT @ self.HT
        J = jnp.vstack((J, JT.reshape(1, J.shape[1], J.shape[2])))
        elems = (A, b, C, eta, J)
        return elems

    def parBackwardPass_extract(self, elems):
        """Parallel LQT backward pass result extraction.

        Parameters:
            elems: List of tuples (A, b, C, eta, J) for 0:T.

        Returns:
            Kx_array: List of control gains for 0:T-1.
            d_array: List of control offsets for 0:T-1.
            S_array: List of value function matrices for 0:T.
            v_array: List of value function offsets for 0:T.
        """

        A_, b_, C_, eta_, J_ = elems
        eta0 = eta_[0]
        J0 = J_[0]

        def parBackwarPass_extract_body(elemts, F, L, U, c, H, r, Z, s, M):
            A, b, C, eta, J = elemts
            S = J
            v = eta
            CF, low = linalg.cho_factor(Z.T @ U @ Z + L.T @ S @ L)
            Kx = linalg.cho_solve((CF, low), Z.T @ M.T @ H + L.T @ S @ F)
            d = linalg.cho_solve(
                (CF, low),
                Z.T @ U @ s + Z.T @ M.T @ r - L.T @ S @ c + L.T @ v,
            )
            return Kx, d, S, v

        Kx_array, d_array, S_array, v_array = vmap(parBackwarPass_extract_body)(
            (A_[1:], b_[1:], C_[1:], eta_[1:], J_[1:]),
            self.F,
            self.L,
            self.U,
            self.c,
            self.H,
            self.r,
            self.Z,
            self.s,
            self.M,
        )
        S_array = jnp.vstack((J0.reshape(1, J0.shape[0], J0.shape[1]), S_array))
        v_array = jnp.vstack((eta0, v_array))
        return Kx_array, d_array, S_array, v_array

    def parBackwardPass(self):
        """Parallel LQT backward pass.

        Returns:
            Kx_array: List of control gains for 0:T-1.
            d_array: List of control offsets for 0:T-1.
            S_array: List of value function matrices for 0:T.
            v_array: List of value function offsets for 0:T.
        """

        # Initialize
        elems = self.parBackwardPass_init()

        # Call the associative scan
        elems = par_backward_pass_scan(elems)

        # Extract the results
        return self.parBackwardPass_extract(elems)

    def parForwardPass_init(self, x0, Kx_array, d_array):
        """Parallel LQT forward pass initialization.

        Parameters:
            x0: Initial state.
            Kx_array: List of control gains for 0:T-1.
            d_array: List of control offsets for 0:T-1.

        Returns:
            elems: List of tuples (F, c) for 0:T.
        """

        tF0 = jnp.zeros_like(self.F[0])
        tc0 = (
            (self.F[0] - self.L[0] @ Kx_array[0]) @ x0
            + self.c[0]
            + self.L[0] @ d_array[0]
        )

        def parForwardPass_init_body(Kx, d, F, L, c):
            tF = F - L @ Kx
            tc = c + L @ d
            return tF, tc

        tF_, tc_ = vmap(parForwardPass_init_body)(
            Kx_array[1:], d_array[1:], self.F[1:], self.L[1:], self.c[1:]
        )
        tF_ = jnp.vstack((tF0.reshape(1, tF0.shape[0], tF0.shape[1]), tF_))
        tc_ = jnp.vstack((tc0, tc_))
        elems = (tF_, tc_)
        return elems

    def parForwardPass_extract(self, x0, Kx_array, d_array, elems):
        """Parallel LQT forward pass result extraction.

        Parameters:
            x0: Initial state.
            Kx_array: List of control gains for 0:T-1.
            d_array: List of control offsets for 0:T-1.
            elems: List of tuples (F, c) for 0:T-1.

        Returns:
            u_array: List of controls for 0:T-1.
            x_array: List of states for 0:T.
        """
        u0 = -Kx_array[0] @ x0 + d_array[0]

        def extract_x(el):
            _, tc = el
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

    def parForwardPass(self, x0, Kx_array, d_array):
        """Parallel LQT forward pass.

        Parameters:
            x0: Initial state.
            Kx_array: List of control gains for 0:T-1.
            d_array: List of control offsets for 0:T-1.

        Returns:
            u_array: List of controls for 0:T-1.
            x_array: List of states for 0:T.
        """
        # Initialize
        elems = self.parForwardPass_init(x0, Kx_array, d_array)

        # Call the associative scan
        elems = par_forward_pass_scan(elems)

        # Extract the results
        return self.parForwardPass_extract(x0, Kx_array, d_array, elems)

    def cost(self, u_array, x_array):
        """Compute the cost of a trajectory.

        Parameters:
            u_array: List of control vectors for 0:T-1.
            x_array: List of state vectors for 0:T.

        Returns:
            res: Cost of the trajectory.
        """
        final_cost = (
            0.5
            * (self.HT @ x_array[-1] - self.rT).T
            @ self.XT
            @ (self.HT @ x_array[-1] - self.rT)
        )

        def stage_cost(H, x, r, X, Z, u, s, U, M):
            ln = 0.5 * (H @ x - r).T @ X @ (H @ x - r)
            ln += 0.5 * (Z @ u - s).T @ U @ (Z @ u - s)
            ln += (H @ x - r).T @ M @ (Z @ u - s)
            return ln

        stage_costs = vmap(stage_cost)(
            self.H,
            x_array[:-1],
            self.r,
            self.X,
            self.Z,
            u_array,
            self.s,
            self.U,
            self.M,
        )

        return final_cost + jnp.sum(stage_costs)
