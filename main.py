import linear_model_jax
import matplotlib.pyplot as plt
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

model = linear_model_jax.LinearModel()
xy = model.genData(50)
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(xy[0, :], xy[1, :], label="Connecting lines", color="b", linestyle="--")
ax.scatter(xy[0, :], xy[1, :], label="Tracked points", color="r")


lqt, x0 = model.getLQT(xy)
Kx_par, d_par, S_par, v_par = lqt.parBackwardPass()
u_par1, x_par1 = lqt.parForwardPass(x0, Kx_par, d_par)


ax.plot(x_par1[:, 0], x_par1[:, 1], label="pLQT1", color="k")
_ = plt.legend()
plt.show()