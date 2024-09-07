import flax
import jax
import optax
from jax import numpy as jnp
from jax import random as jr
from jax.scipy.linalg import block_diag
from matplotlib import pyplot as plt
from rares_layers import LRU, GammaDecayBlockDiagEfficient
from ssm import S5SSM
from ssm_init import make_DPLR_HiPPO

jax.config.update("jax_platform_name", "cpu")


ssm_size = 256
input_size = 128
r_min = 0
r_max = 1 - 1e-6
max_phase = 6.28
bidirectional = False
model_type = "rotssm"
T = 100000

key = jr.PRNGKey(0)

n_seqs = 1

model_types = ["rotssm", "lru", "s5"]

fig, ax = plt.subplots(n_seqs, 1)
if n_seqs == 1:
    ax = [ax]
fig.set_figwidth(12)
fig.set_figheight(n_seqs * 6)

input_key, model_key = jr.split(key)
inputs = []
targets = []
n_seqs = 100
for i in range(n_seqs):
    input_key, cur_key = jr.split(input_key)
    cur_input_key, target_key_1, target_key_2 = jr.split(cur_key, 3)
    input = jr.normal(cur_input_key, (T - 2, input_size))

    in_0_1 = bool(jr.bernoulli(target_key_1))
    in_0 = jnp.ones((1, input_size)) if in_0_1 else jnp.zeros((1, input_size))
    in_T_1 = bool(jr.bernoulli(target_key_2))
    in_T = jnp.ones((1, input_size)) if in_T_1 else jnp.zeros((1, input_size))

    target = in_0_1 == in_T_1
    input = jnp.concatenate([in_0, input, in_T], axis=0)
    inputs.append(input)
    targets.append(target)


for model_type in model_types:
    if model_type == "lru":
        model = LRU(
            lru_dim=ssm_size,
            hidden_dim=input_size,
            r_min=r_min,
            r_max=r_max,
            max_phase=max_phase,
            bidirectional=bidirectional,
        )
    elif model_type == "rotssm":
        model = GammaDecayBlockDiagEfficient(
            lru_dim=ssm_size,
            hidden_dim=input_size,
            r_min=r_min,
            r_max=r_max,
            max_phase=max_phase,
            bidirectional=bidirectional,
            nheads=32,
        )
    elif model_type == "s5":
        block_size = 16
        blocks = 16
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T
        Lambda = (Lambda * jnp.ones((blocks, block_size))).ravel()
        V = block_diag(*([V] * blocks))
        Vinv = block_diag(*([Vc] * blocks))
        model = S5SSM(
            H=input_size,
            P=ssm_size,
            Lambda_re_init=Lambda.real,
            Lambda_im_init=Lambda.imag,
            V=V,
            Vinv=Vinv,
            C_init="trunc_standard_normal",
            discretization="zoh",
            dt_min=0.001,
            dt_max=0.1,
            bidirectional=bidirectional,
            conj_sym=False,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ssm_params = model.init({"params": model_key}, inputs[0])

    out_layer = flax.linen.Dense(2)
    out_params = out_layer.init(jr.PRNGKey(0), jnp.zeros((ssm_size,)))

    opt = optax.adam(learning_rate=1e-3)
    opt_state = opt.init()

    outputs = model.apply(ssm_params, inputs[0])

    if model_type == "s5":
        output_norm = jnp.sqrt(
            jnp.einsum("...i,...i->...", outputs.real, outputs.real)
            + jnp.einsum("...i,...i->...", outputs.imag, outputs.imag)
        )
    else:
        output_norm = jnp.sqrt(jnp.einsum("...i,...i->...", outputs, outputs))
    if model_type == "rotssm":
        output_norm = output_norm.mean(axis=-1)

    norms.append(output_norm)

for norm, name in zip(norms, model_types):
    ax[n].scatter(range(T), norm, label=name, linewidths=0.1)
    print(name)
    jax.debug.print("{x}", x=norm.max())
ax[0].legend()
plt.savefig("toy.png")
