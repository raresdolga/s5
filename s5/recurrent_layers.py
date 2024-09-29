import jax
from flax import linen as nn
from jax import numpy as jnp

parallel_scan = jax.lax.associative_scan


class LRU(nn.Module):
    """
    Complex number implementation as in the original LRU paper
    """

    lru_dim: int  # N: state dimension
    hidden_dim: int  # H: model dimension
    r_min: float = 0.9
    r_max: float = 0.999
    max_phase: float = jnp.pi / 10
    bidirectional: bool = True
    step_rescale: float = 1.0

    @staticmethod
    def theta_init(rng_key, lru_dim, max_phase=6.28):
        # print("Max_phase: ", max_phase)
        subkeys = jax.random.split(rng_key, num=3)
        u1 = jax.random.uniform(subkeys[0], shape=(lru_dim,))
        theta_log = jnp.log(max_phase * u1)
        return theta_log

    @staticmethod
    def nu_init(rng_key, lru_dim, r_min=0, r_max=1):
        # print("R_min: ", r_min, "r_max: ",  r_max)
        subkeys = jax.random.split(rng_key, num=3)
        u1 = jax.random.uniform(subkeys[0], shape=(lru_dim,))
        nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        return nu_log

    @staticmethod
    def b_init(rng_key, lru_dim, hidden_dim):
        subkeys = jax.random.split(rng_key, num=2)
        # Glorot initialized Input/Output projection matrices
        B = jax.random.normal(subkeys[0], shape=(lru_dim, hidden_dim)) / jnp.sqrt(
            2 * hidden_dim
        )
        return B

    @staticmethod
    def c_init(rng_key, lru_dim, hidden_dim):
        subkeys = jax.random.split(rng_key, num=2)
        # Glorot initialized Input/Output projection matrices
        C = jax.random.normal(subkeys[0], shape=(hidden_dim, lru_dim)) / jnp.sqrt(
            lru_dim
        )
        return C

    @staticmethod
    def binary_operator_diag(element_i, element_j):
        """Binary operator for parallel scan of linear recurrence."""
        a_i, bu_i = element_i
        a_j, bu_j = element_j
        return a_j * a_i, a_j * bu_i + bu_j

    def mix_sequence(self, Lambda_elements, Bu_elements, reverse=False):
        elements = (Lambda_elements, Bu_elements)
        _, inner_states = parallel_scan(
            self.binary_operator_diag, elements, reverse=reverse
        )  # all x_k  # LBN
        # inner_states = inner_states.transpose(1, 0, 2)  # LBN -> BLN
        return inner_states

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: jnp.array(TD)
        """
        theta_log = self.param(
            "theta_log", self.theta_init, self.lru_dim, self.max_phase
        )
        nu_log = self.param(
            "nu_log", self.nu_init, self.lru_dim, self.r_min, self.r_max
        )
        B_re = self.param("B_re", self.b_init, self.lru_dim, self.hidden_dim)
        B_im = self.param("B_im", self.b_init, self.lru_dim, self.hidden_dim)
        C_re = self.param("C_re", self.c_init, self.lru_dim, self.hidden_dim)
        C_im = self.param("C_im", self.c_init, self.lru_dim, self.hidden_dim)
        if self.bidirectional:
            C_re2 = self.param("C_re2", self.c_init, self.lru_dim, self.hidden_dim)
            C_im2 = self.param("C_im2", self.c_init, self.lru_dim, self.hidden_dim)

        D = self.param(
            "D", lambda rng, H: jax.random.normal(rng, shape=(H,)), self.hidden_dim
        )

        Lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
        gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(Lambda) ** 2))
        B_norm = (B_re + 1j * B_im) * jnp.expand_dims(jnp.exp(gamma_log), axis=-1)
        C = C_re + 1j * C_im
        # Running the LRU + output projection
        # For details on parallel scan, check discussion in Smith et al (2022).
        # Lambda_elements = jnp.repeat(Lambda[None, ...], x.shape[1], axis=0)
        Lambda_elements = jnp.repeat(Lambda[None, ...], x.shape[0], axis=0)
        # Lambda_elements = Lambda_elements.transpose(1, 0, 2)  # BLN -> LBN

        # Bu_elements = jax.vmap(lambda u: u @ B_norm.T)(x) # LBN
        Bu_elements = jnp.einsum("LH,NH->LN", x, B_norm)

        # print(Lambda_elements.shape, Bu_elements.shape)
        inner_states = self.mix_sequence(Lambda_elements, Bu_elements, reverse=False)
        if self.bidirectional:
            C2 = C_re2 + 1j * C_im2
            C = jnp.concatenate([C, C2], axis=-1)
            backward = self.mix_sequence(Lambda_elements, Bu_elements, reverse=True)
            inner_states = jnp.concatenate(
                [inner_states, backward], axis=-1
            )  # BLN -> BL2N

        y = jnp.einsum("HN,LN->LH", C, inner_states).real + jnp.einsum("H,LH->LH", D, x)
        # y = jax.vmap(lambda x, u: (x@C.T).real + D * u)(inner_states, x).transpose(1,0,2) # LBH -> BLH
        return y


def binary_operator_diag(element_i, element_j):
    # Binary operator for parallel scan of linear recurrence.
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j


class SimpleRotRNN(nn.Module):
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
        ) / jnp.sqrt(self.model_dim)
        return M

    def ortho_mat_init(self, rng_key, n_heads, output_dim, input_dim):
        M = jax.random.normal(rng_key, shape=(n_heads, output_dim, input_dim))
        return M

    def output_mat_init(self, rng_key, output_dim, input_dim):
        M = jax.random.normal(rng_key, shape=(output_dim, input_dim)) / jnp.sqrt(
            self.rotrnn_dim
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
        theta = self.param("theta", self.theta_init, H, N, self.max_phase)
        M = self.param("M", self.ortho_mat_init, H, N, N)
        B = self.param("B", self.input_mat_init, H, N, self.model_dim)
        C = self.param("C", self.output_mat_init, self.model_dim, self.rotrnn_dim)
        if self.bidirectional:
            C2 = self.param("C2", self.output_mat_init, self.model_dim, self.rotrnn_dim)
        D = self.param(
            "D", lambda rng, H: jax.random.normal(rng, shape=(H,)), self.model_dim
        )
        gamma_log = self.param("gamma_log", self.gamma_log_init, H)
        gamma = jnp.exp(-jnp.exp(gamma_log))

        T, dim_u = input_sequence.shape

        # compute \xi and normalise B
        B_T_B = jax.vmap(lambda a, b: a @ b)(B.transpose(0, 2, 1), B)
        B_T_B_trace = jnp.trace(B_T_B, axis1=1, axis2=2)
        xi = jnp.sqrt((1 - gamma.squeeze() ** 2) / B_T_B_trace)
        B_norm = xi[..., None, None] * B

        # create orthogonal matrix P from weight matrix M
        P = jax.scipy.linalg.expm(M - M.transpose(0, 2, 1))

        # project inputs onto heads
        x = jnp.einsum("HDi,Ti->HTD", B_norm, input_sequence)

        # multiply from the right by P^T
        x = jax.vmap(lambda a, b: a @ b)(x, P.transpose(0, 2, 1))

        # compute recurrence parallelised over heads
        gamma = jnp.repeat(gamma[:, None], repeats=T, axis=1)
        theta = jnp.repeat(theta[:, None], repeats=T, axis=1)
        multi_head_scan = jax.vmap(
            lambda g, t, x, rev: parallel_scan(self.binf, (g, t, x), rev),
            in_axes=(0, 0, 0, None),
            out_axes=0,
        )
        x = multi_head_scan(gamma, theta, x, False)[2]

        # multiply from the left by P
        x = jnp.einsum("HDi, HTi -> HTD", P, x)

        if self.bidirectional:
            # compute reverse recurrence parallelised over heads
            bwd = multi_head_scan(gamma, theta, x, True)[2]

            # multiply from the left by P
            bwd = jnp.einsum("HDi, HTi -> HTD", P, bwd)

            # concatenate fwd and bwd over dimension
            x = jnp.concatenate([x, bwd], axis=-1)
            C = jnp.concatenate([C, C2], axis=-1)

        # concatenate heads
        x = x.transpose(1, 0, 2).reshape(T, -1)

        # apply output projection/head mixing and skip connection
        x = jax.vmap(lambda a: C @ a)(x) + D * input_sequence
        return x
