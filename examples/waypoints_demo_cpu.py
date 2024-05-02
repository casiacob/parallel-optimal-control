import jax

from paroc.linear_model_jax import gen_waypoints, generate_lqt
from paroc._parallel import par_bwd_pass, par_fwd_pass
from paroc._sequential import seq_bwd_pass, seq_fwd_pass
import matplotlib.pyplot as plt
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

wp = gen_waypoints(10)
# wp = jax.numpy.array([[5., 6., 7.], [5., 5., 6.]])
lqt, x0 = generate_lqt(wp, 10)

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(wp[0, :], wp[1, :], label="Connecting lines", color="b", linestyle="--")
ax.scatter(wp[0, :], wp[1, :], label="Tracked points", color="r")

# import time
#
# start = time.time()

_jitted_par_bwd_pass = jax.jit(par_bwd_pass, backend='cpu')
_jitted_par_fwd_pass = jax.jit(par_fwd_pass, backend='cpu')

Kx_par, d_par, S_par, v_par, _, _ = _jitted_par_bwd_pass(lqt)
print(Kx_par)
print(d_par)
u_par, x_par = _jitted_par_fwd_pass(lqt, x0, Kx_par, d_par)
print(x_par)
# for _ in range(10):
#     Kx_par, d_par, S_par, v_par, _, _ = _jitted_par_bwd_pass(lqt)
#     Kx_par.block_until_ready()
#     u_par, x_par = _jitted_par_fwd_pass(lqt, x0, Kx_par, d_par)
#     u_par.block_until_ready()

# end = time.time()
# print("time: ", end - start)
# Kx_seq, d_seq, S_seq, v_seq = seq_bwd_pass(lqt)
# u_seq, x_seq = seq_fwd_pass(lqt, x0, Kx_par, d_par)

ax.plot(x_par[:, 0], x_par[:, 1], label="pLQT", color="k")
# ax.plot(x_seq[:, 0], x_seq[:, 1], label="sLQT", color="g")
_ = plt.legend()
plt.show()