import jax.numpy as jnp
import numpy as np
from paroc.lqt_problem import LQT
from paroc import par_bwd_pass, par_fwd_pass
from paroc import seq_bwd_pass, seq_fwd_pass
import unittest
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


class LQT_jax_UnitTest(unittest.TestCase):
    def setupRndLQT(self):
        #
        # Test by initializing with random matrices
        # while keeping the sizes the default
        #
        rng = np.random.default_rng(123)

        T = 5
        nx = 4
        nu = 2
        nr = 3

        x0 = rng.standard_normal(nx)

        c = T * [0]
        F = T * [0]
        s = T * [0]
        r = T * [0]
        M = T * [0]
        Z = T * [0]
        H = T * [0]
        L = T * [0]
        X = T * [0]
        U = T * [0]

        HT = rng.standard_normal((nr, nx))
        rT = rng.standard_normal(nr)
        XT = rng.standard_normal((nr, 2 * nr))
        XT = XT @ XT.T
        for i in range(T):
            c[i] = rng.standard_normal(nx)
            F[i] = rng.standard_normal((nx, nx))
            s[i] = rng.standard_normal(nu)
            r[i] = rng.standard_normal(nr)
            M[i] = rng.standard_normal((nr, nu))
            Z[i] = rng.standard_normal((nu, nu))
            H[i] = rng.standard_normal((nr, nx))
            L[i] = rng.standard_normal((nx, nu))
            X[i] = rng.standard_normal((nr, 2 * nr))
            X[i] = X[i] @ X[i].T
            U[i] = rng.standard_normal((nu, 2 * nu))
            U[i] = U[i] @ U[i].T + np.eye(nu)
        F = jnp.array(F)
        L = jnp.array(L)
        X = jnp.array(X)
        U = jnp.array(U)
        XT = jnp.array(XT)
        c = jnp.array(c)
        H = jnp.array(H)
        r = jnp.array(r)
        HT = jnp.array(HT)
        rT = jnp.array(rT)
        Z = jnp.array(Z)
        s = jnp.array(s)
        M = jnp.array(M)

        lqt = LQT(F, L, c, XT, HT, rT, X, H, r, U, Z, s, M)
        return lqt, x0

    def test_seq_par_rnd_tracking_1(self):
        lqt, x0 = self.setupRndLQT()

        Kx_par, d_par, S_par, v_par, _, _ = par_bwd_pass(lqt)
        Kx_seq, d_seq, S_seq, v_seq = seq_bwd_pass(lqt)

        u_par, x_par = par_fwd_pass(lqt, x0, Kx_par, d_par)
        u_seq, x_seq = par_fwd_pass(lqt, x0, Kx_seq, d_seq)

        err = jnp.max(jnp.abs(x_par - x_seq))
        self.assertTrue(err < 1e-10)

        err = jnp.max(jnp.abs(u_par - u_seq))
        self.assertTrue(err < 1e-10)
