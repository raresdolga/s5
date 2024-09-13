import jax
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt
from rares_layers import GammaDecayBlockDiagEfficient

jax.config.update("jax_platform_name", "cpu")


ssm_size = 8
input_size = 1
r_min = 0.9
r_max = 0.9999
max_phase = 0.314
bidirectional = False
model_type = "rotssm"
nheads = 4
T = 1000

model_key = jr.PRNGKey(2)

fig, ax = plt.subplots(nheads, 1)
if nheads == 1:
    ax = [ax]
fig.set_figwidth(20)
fig.set_figheight(nheads * 4)

toy_inputs = jnp.concat((jnp.ones(1), jnp.zeros(T - 1)))[:, None]
model = GammaDecayBlockDiagEfficient(
    lru_dim=ssm_size,
    hidden_dim=input_size,
    r_min=r_min,
    r_max=r_max,
    max_phase=max_phase,
    bidirectional=bidirectional,
    nheads=nheads,
)

params = model.init({"params": model_key}, toy_inputs)

outputs = model.apply(params, toy_inputs)  # T H N

for h in range(nheads):
    for n in range(ssm_size // nheads):
        ax[h].plot(outputs[:, h, n], linewidth=3)
    ax[h].get_xaxis().set_ticks([])
    ax[h].get_yaxis().set_ticks([])
plt.savefig("kernel_plot.png")
