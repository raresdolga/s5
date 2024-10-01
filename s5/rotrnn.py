import jax
from flax import linen as nn
from jax import numpy as jnp

parallel_scan = jax.lax.associative_scan


class RotRNN(nn.Module):
    rotrnn_dim: int = 64
    model_dim: int = 128
    n_heads: int = 64
    gamma_min: float = 0.9
    gamma_max: float = 0.999
    max_phase: float = 6.28
    bidirectional: bool = False
    step_rescale: float = 0.0

    def theta_init(self, rng_key, n_heads, head_dim, max_phase):
        return jax.random.uniform(
            rng_key, shape=(n_heads, head_dim // 2), maxval=max_phase
        )

    def gamma_log_init(self, rng_key, H):
        u1 = jax.random.uniform(rng_key, shape=(H, 1))
        gamma_log = jnp.log(-jnp.log(self.gamma_max)) + u1 * (
            jnp.log(-jnp.log(self.gamma_min)) - jnp.log(-jnp.log(self.gamma_max))
        )
        return gamma_log

    def input_mat_init(self, rng_key, n_heads, output_dim, input_dim):
        M = jax.random.normal(
            rng_key, shape=(n_heads, output_dim, input_dim)
        ) / jnp.sqrt(self.model_dim + self.rotrnn_dim)
        return M

    def ortho_mat_init(self, rng_key, n_heads, output_dim, input_dim):
        M = jax.random.normal(rng_key, shape=(n_heads, output_dim, input_dim))
        return M

    def output_mat_init(self, rng_key, output_dim, input_dim):
        M = jax.random.normal(rng_key, shape=(output_dim, input_dim)) / jnp.sqrt(
            self.model_dim + self.rotrnn_dim
        )
        return M

    @staticmethod
    def binf(a, b):
        """Binary function for the parallel scan"""
        gamma_i, thetas_i, acc_i = a
        gamma_j, thetas_j, acc_j = b

        # get off diagonal terms [-x2, x1, -x4, x3,...]
        # these will be multiplied by sin(\theta)
        off_diags = jnp.stack([-acc_i[..., 1::2], acc_i[..., 0::2]], axis=-1)
        off_diags = off_diags.reshape(acc_i.shape)

        # duplicate \theta [\theta_1, \theta_1, \theta_2, \theta_2,...]
        theta = jnp.repeat(thetas_j, repeats=2, axis=-1)

        # compute sine and cosine elements of the output
        sin = jnp.sin(theta) * off_diags
        cos = jnp.cos(theta) * acc_i
        acc = gamma_j * (cos + sin)

        return (gamma_i * gamma_j, thetas_i + thetas_j, acc + acc_j)

    @nn.compact
    def __call__(self, input_sequence):
        H, N = self.n_heads, self.rotrnn_dim // self.n_heads
        assert N % 2 == 0, "N should be even"
        thetas = self.param("theta", self.theta_init, H, N, self.max_phase)
        M = self.param("M", self.ortho_mat_init, H, N, N)
        B = self.param("B", self.input_mat_init, H, N, self.model_dim)
        C = self.param("C", self.output_mat_init, self.model_dim, self.rotrnn_dim)
        if self.bidirectional:
            C2 = self.param("C2", self.output_mat_init, self.model_dim, self.rotrnn_dim)
        D = self.param(
            "D", lambda rng, H: jax.random.normal(rng, shape=(H,)), self.model_dim
        )
        gamma_log = self.param("gamma_log", self.gamma_log_init, H)
        gammas = jnp.exp(-jnp.exp(gamma_log))

        T, dim_u = input_sequence.shape

        # compute \xi and normalise B
        B_T_B = jax.vmap(lambda a, b: a @ b)(B.transpose(0, 2, 1), B)
        B_T_B_trace = jnp.trace(B_T_B, axis1=1, axis2=2)
        xi = jnp.sqrt((1 - gammas.squeeze() ** 2) / B_T_B_trace)
        B_norm = jnp.einsum("H, HTD -> HTD", xi, B)

        # create orthogonal matrix P from weight matrix M
        P = jax.scipy.linalg.expm(M - M.transpose(0, 2, 1))

        # project inputs onto heads
        x = jnp.einsum("HDi,Ti->HTD", B_norm, input_sequence)

        # project with P^T
        x = jnp.einsum("HDi, HTi -> HTD", P.transpose(0, 2, 1), x)

        # compute recurrence parallelised over heads
        gammas = jnp.repeat(gammas[:, None], repeats=T, axis=1)
        thetas = jnp.repeat(thetas[:, None], repeats=T, axis=1)
        rec_fn = jax.vmap(
            lambda a, b, c: parallel_scan(self.binf, (a, b, c)),
            in_axes=(0, 0, 0),
            out_axes=0,
        )
        hidden_states = rec_fn(gammas, thetas, x)[2]

        # project back with P
        x = jnp.einsum("HDi, HTi -> HTD", P, hidden_states)

        if self.bidirectional:
            backward = rec_fn(gammas, thetas, hidden_states, reverse=True)[2]
            backward = jnp.einsum("HDi, HTi -> HTD", P, backward)
            x = jnp.concatenate([x, backward], axis=-1)
            C = jnp.concatenate([C, C2], axis=-1)

        # concatenate heads
        x = x.transpose(1, 0, 2).reshape(T, -1)

        # apply output projection/head mixing and skip connection
        y = jax.vmap(lambda a: C @ a)(x) + D * input_sequence
        return y
