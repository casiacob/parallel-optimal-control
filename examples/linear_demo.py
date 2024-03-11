from cpoc import linear_model_jax
import matplotlib.pyplot as plt
from jax import config
import numpy
import jax.numpy as jnp

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

model = linear_model_jax.LinearModel()
xy = model.genData(10)
# xy = numpy.load("data.npy")
# xy = jnp.array(xy)
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(xy[0, :], xy[1, :], label="Connecting lines", color="b", linestyle="--")
ax.scatter(xy[0, :], xy[1, :], label="Tracked points", color="r")


lqt, x0 = model.getLQT(xy)
Kx_par, d_par, S_par, v_par = lqt.parBackwardPass()
u_par, x_par = lqt.parForwardPass(x0, Kx_par, d_par)

Kx_seq, d_seq, S_seq, v_seq = lqt.seqBackwardPass()
u_seq, x_seq = lqt.parForwardPass(x0, Kx_seq, d_seq)


ax.plot(x_par[:, 0], x_par[:, 1], label="pLQT", color="k")
# ax.plot(x_seq[:, 0], x_seq[:, 1], label="sLQT", color="g")
# _ = plt.legend()
plt.show()
