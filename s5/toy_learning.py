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
from tqdm import tqdm

jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_debug_nans", True)


def loss_fn(params, ssm_layer, out_layer, inputs, targets):
    kernel_outputs, _ = jax.vmap(ssm_layer.apply, (None, 0))(params[0], inputs)
    kernel_outputs = kernel_outputs.mean(axis=1)
    outputs = jax.vmap(out_layer.apply, (None, 0))(params[1], kernel_outputs)
    return optax.losses.softmax_cross_entropy_with_integer_labels(
        outputs, targets
    ).mean()


ssm_size = 256
input_size = 128
r_min = 1e-6
r_max = 1 - 1e-4
max_phase = 6.28
bidirectional = False
model_type = "rotssm"
T = 10000
nsteps = 1000
batch_size = 8
lr = 1e-2

key = jr.PRNGKey(0)

model_types = ["rotssm", "lru", "s5"]

fig, ax = plt.subplots(2, 1)
fig.set_figwidth(12)
fig.set_figheight(12)

input_key, model_key = jr.split(key)
inputs = []
targets = []
n_seqs = 1000
for i in range(n_seqs):
    input_key, cur_key = jr.split(input_key)
    cur_input_key, target_key_1, target_key_2 = jr.split(cur_key, 3)
    input = jr.normal(cur_input_key, (T - 2, input_size))

    in_0_1 = bool(jr.bernoulli(target_key_1))
    in_0 = jnp.ones((1, input_size)) if in_0_1 else jnp.zeros((1, input_size))
    in_T_1 = bool(jr.bernoulli(target_key_2))
    in_T = jnp.ones((1, input_size)) if in_T_1 else jnp.zeros((1, input_size))

    target = int(in_0_1 == in_T_1)
    input = jnp.concatenate([in_0, input, in_T], axis=0)
    inputs.append(input)
    targets.append(target)
inputs = jnp.stack(inputs)
targets = jnp.array(targets)


all_norms = []
all_losses = []
for model_type in model_types:
    norms = []
    losses = []
    if model_type == "lru":
        ssm = LRU(
            lru_dim=ssm_size,
            hidden_dim=input_size,
            r_min=r_min,
            r_max=r_max,
            max_phase=max_phase,
            bidirectional=bidirectional,
        )
    elif model_type == "rotssm":
        ssm = GammaDecayBlockDiagEfficient(
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
        ssm = S5SSM(
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

    ssm_params = ssm.init({"params": model_key}, inputs[0])

    out_layer = flax.linen.Dense(2)
    out_params = out_layer.init(jr.PRNGKey(0), jnp.zeros((input_size,)))

    ssm_opt = optax.adam(learning_rate=lr)
    out_opt = optax.adam(learning_rate=lr)
    ssm_opt_state = ssm_opt.init(ssm_params)
    out_opt_state = out_opt.init(out_params)

    pbar = tqdm(range(nsteps))
    for i in pbar:
        cur_key, input_key = jr.split(input_key)
        in_idxs = jr.choice(input_key, n_seqs, shape=(batch_size,), replace=False)
        ins = inputs[in_idxs]
        targs = targets[in_idxs]
        _, hidden_state = jax.vmap(ssm.apply, (None, 0))(ssm_params, ins)
        loss, (ssm_grad, out_grad) = jax.value_and_grad(loss_fn, argnums=0)(
            (ssm_params, out_params), ssm, out_layer, ins, targs
        )
        ssm_grad, ssm_opt_state = ssm_opt.update(ssm_grad, ssm_opt_state)
        out_grad, out_opt_state = out_opt.update(out_grad, out_opt_state)
        ssm_params = optax.apply_updates(ssm_params, ssm_grad)
        out_params = optax.apply_updates(out_params, out_grad)

        if model_type in ["s5", "lru"]:
            hidden_state_norm = jnp.sqrt(
                jnp.einsum("...i,...i->...", hidden_state.real, hidden_state.real)
                + jnp.einsum("...i,...i->...", hidden_state.imag, hidden_state.imag)
            )
        else:
            hidden_state_norm = jnp.sqrt(
                jnp.einsum("...i,...i->...", hidden_state, hidden_state)
            )
        if model_type == "rotssm":
            hidden_state_norm = hidden_state_norm.mean(axis=-1)
        norm = hidden_state_norm.mean()
        pbar.set_postfix(norm=norm, loss=loss)
        norms.append(norm)
        losses.append(loss)

    all_norms.append(norms)
    all_losses.append(losses)

for loss, norm, name in zip(all_losses, all_norms, model_types):
    ax[0].plot(norm, label=name, linewidth=0.1)
    ax[1].plot(loss, label=name, linewidth=0.1)
    print(name)
ax[0].legend()
plt.savefig("toy_learning.png")
