import jax.numpy as jnp
import numpy as np
import lqt_jax
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

        lqt = lqt_jax.LQT(F, L, X, U, XT, c, H, r, HT, rT, Z, s, M)
        return lqt, x0

    def test_seq_par_rnd_tracking_1(self):
        lqt, x0 = self.setupRndLQT()

        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        Kx_list2, d_list2, S_list2, v_list2 = lqt.parBackwardPass()

        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)
        u_list2, x_list2 = lqt.parForwardPass(x0, Kx_list2, d_list2)

        err = jnp.max(jnp.abs(x_list1 - x_list2))
        self.assertTrue(err < 1e-10)

        err = jnp.max(jnp.abs(u_list1 - u_list2))
        self.assertTrue(err < 1e-10)
