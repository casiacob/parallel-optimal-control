import numpy as np
import linear_model_jax
import matplotlib.pyplot as plt
import jax


model = linear_model_jax.LinearModel()
xy = model.genData(50)

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(xy[0, :], xy[1, :], label="Connecting lines", color="b", linestyle="--")
ax.scatter(xy[0, :], xy[1, :], label="Tracked points", color="r")
_ = plt.legend()


lqt, x0 = model.getLQT(xy)
Kx_list_par, d_list_par, S_list_par, v_list_par = lqt.parBackwardPass()
u_list_par1, x_list_par1 = lqt.parForwardPass(x0, Kx_list_par, d_list_par)

ax.scatter(x_list_par1[0, :], x_list_par1[1, :], label="lqt points", color="g")
plt.show()