import jax.numpy as jnp
from jax.numpy import linalg
import jax.random as jr
from jax import lax
import lqt_jax
import unittest


class LQT_jax_UnitTest(unittest.TestCase):
    def setupRndLQT(self):
        #
        # Test by initializing with random matrices
        # while keeping the sizes the default
        #

        seed = 123
        key0 = jr.PRNGKey(seed)

        T = 5
        nx = 4
        nu = 2
        nr = 3

        x0 = jr.normal(key0, (nx,))

        HT_rnd = jr.normal(key0, (nr, nx))
        rT_rnd = jr.normal(key0, (nr,))
        XT_rnd = jr.normal(key0, (nr, nr))

        def body(carry, inp):
            key = carry
            key, sub_key = jr.split(key, 2)
            c = jr.normal(sub_key, (nx,))
            F = jr.normal(sub_key, (nx, nx))
            s = jr.normal(sub_key, (nu,))
            r = jr.normal(sub_key, (nr,))
            M = jr.normal(sub_key, (nr, nu))
            Z = jr.normal(sub_key, (nu, nu))
            H = jr.normal(sub_key, (nr, nx))
            L = jr.normal(sub_key, (nx, nu))
            X = jr.normal(sub_key, (nr, 2 * nr))
            X = X @ X.T
            U = jr.normal(sub_key, (nu, 2 * nu))
            U = U @ U.T + jnp.eye(nu)
            return key, (F, L, X, U, c, H, r, Z, s, M)

        _, random_matrices = lax.scan(body, key0, xs=None, length=T)
        (
            F_rnd_array,
            L_rnd_array,
            X_rnd_array,
            U_rnd_array,
            c_rnd_array,
            H_rnd_array,
            r_rnd_array,
            Z_rnd_array,
            s_rnd_array,
            M_rnd_array,
        ) = random_matrices
        lqt = lqt_jax.LQT(
            F_rnd_array,
            L_rnd_array,
            X_rnd_array,
            U_rnd_array,
            XT_rnd,
            c_rnd_array,
            H_rnd_array,
            r_rnd_array,
            HT_rnd,
            rT_rnd,
            Z_rnd_array,
            s_rnd_array,
            M_rnd_array,
        )
        return lqt, x0

    def test_seq_par_rnd_tracking_1(self):
        lqt, x0 = self.setupRndLQT()

        Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
        Kx_list2, d_list2, S_list2, v_list2 = lqt.parBackwardPass()
        u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)
        u_list2, x_list2 = lqt.parForwardPass(x0, Kx_list2, d_list2)

        self.assertTrue(len(x_list1) == len(x_list2))
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(x_list1, x_list2)])
        self.assertTrue(err < 1e-10)

        self.assertTrue(len(u_list1) == len(u_list2))
        err = max([linalg.norm(e1 - e2) for e1, e2 in zip(u_list1, u_list2)])
        self.assertTrue(err < 1e-10)
