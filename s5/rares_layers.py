import jax
import numpy as np
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
            "theta_log",
            self.theta_init,
            self.lru_dim,
            self.max_phase,
        )
        nu_log = self.param(
            "nu_log",
            self.nu_init,
            self.lru_dim,
            self.r_min,
            self.r_max,
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


class LRU2(nn.Module):
    N: int
    H: int
    r_min: float
    r_max: float
    max_phase: float
    bidirectional: bool = False
    step_rescale: float = 0.0

    def theta_init(self, key, N, max_phase):
        u2 = jax.random.uniform(key, shape=(N,))
        theta_log = jnp.log(max_phase * u2)
        return theta_log

    def nu_init(self, key, N, r_max, r_min):
        u1 = jax.random.uniform(key, shape=(N,))
        nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        return nu_log

    def b_init(self, key, N, H):
        B_re = jax.random.normal(key, shape=(N, H)) / np.sqrt(2 * H)
        # B_im = np.random.normal(size=(N,H))/np.sqrt(2*H)
        return B_re

    def c_init(self, key, N, H):
        C_re = jax.random.normal(key, shape=(H, N)) / np.sqrt(N)
        # C_im = np.random.normal(size=(H,N))/np.sqrt(N)
        return C_re

    @nn.compact
    def __call__(self, input_sequence):
        """Forward pass of the LRU layer. Output y and input_sequence are of shape (L, H)."""

        # All LRU parameters
        # nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log = lru_parameters
        theta_log = self.param(
            "theta_log",
            self.theta_init,
            self.N,
            self.max_phase,
        )
        nu_log = self.param(
            "nu_log",
            self.nu_init,
            self.N,
            self.r_min,
            self.r_max,
        )

        B_re = self.param("B_re", self.b_init, self.N, self.H)
        B_im = self.param("B_im", self.b_init, self.N, self.H)
        C_re = self.param("C_re", self.c_init, self.N, self.H)
        C_im = self.param("C_im", self.c_init, self.N, self.H)
        D = self.param("D", jax.random.normal, (self.H,))

        # Normalization factor
        diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
        gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))
        # Materializing the diagonal of Lambda and projections
        Lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
        B_norm = (B_re + 1j * B_im) * jnp.expand_dims(jnp.exp(gamma_log), axis=-1)
        C = C_re + 1j * C_im

        # Running the LRU + output projection
        # For details on parallel scan, check discussion in Smith et al (2022).
        Lambda_elements = jnp.repeat(Lambda[None, ...], input_sequence.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(input_sequence)
        elements = (Lambda_elements, Bu_elements)
        _, inner_states = parallel_scan(binary_operator_diag, elements)  # all x_k
        y = jax.vmap(lambda x, u: (C @ x).real + D * u)(inner_states, input_sequence)

        return y

    def init_lru_parameters(N, H, r_min=0, r_max=1, max_phase=6.28):
        """Initialize parameters of the LRU layer."""
        # N: state dimension, H: model dimension
        # Initialization of Lambda is complex valued distributed uniformly on ring
        # between r_min and r_max, with phase in [0, max_phase].
        u1 = np.random.uniform(size=(N,))
        u2 = np.random.uniform(size=(N,))
        nu_log = np.log(-0.5 * np.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        theta_log = np.log(max_phase * u2)

        # Glorot initialized Input/Output projection matrices
        B_re = np.random.normal(size=(N, H)) / np.sqrt(2 * H)
        B_im = np.random.normal(size=(N, H)) / np.sqrt(2 * H)
        C_re = np.random.normal(size=(H, N)) / np.sqrt(N)
        C_im = np.random.normal(size=(H, N)) / np.sqrt(N)
        D = np.random.normal(size=(H,))
        # Normalization factor
        diag_lambda = np.exp(-np.exp(nu_log) + 1j * np.exp(theta_log))
        gamma_log = np.log(np.sqrt(1 - np.abs(diag_lambda) ** 2))
        return nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log


def binary_operator_diag(element_i, element_j):
    # Binary operator for parallel scan of linear recurrence.
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j


class GammaDecayBlockDiagEfficient(nn.Module):
    lru_dim: int = 64  # devisible by heads
    hidden_dim: int = 128  # devisible by heads
    nheads: int = 64  # apply model in parallel
    r_min: float = 0.9
    r_max: float = 0.999
    max_phase: float = 6.28
    bidirectional: bool = False
    step_rescale: float = 0.0

    def theta_init(self, rng_key, N, max_phase):
        return jnp.log(jax.random.uniform(rng_key, shape=(N, 1), maxval=max_phase))

    def nu_log_init(self, rng_key, H, r_min=0, r_max=1):
        """
        r_min, r_max in (0, 1)
        """
        subkey, rng_key = jax.random.split(rng_key)
        u1 = jax.random.uniform(subkey, shape=(H,))
        # double exponential
        nu_log = jnp.log(-jnp.log(r_max)) + u1 * (
            jnp.log(-jnp.log(r_min)) - jnp.log(-jnp.log(r_max))
        )
        return nu_log

    def mat_init(self, rng_key, lru_dim, hidden_dim):
        subkey, rng_key = jax.random.split(rng_key, num=2)
        # Glorot initialized Input/Output projection matrices
        B = jax.random.normal(subkey, shape=(lru_dim, hidden_dim)) / jnp.sqrt(
            hidden_dim + lru_dim
        )
        return B

    def ortho_mat_init(self, rng_key, lru_dim, hidden_dim):
        subkey, rng_key = jax.random.split(rng_key, num=2)
        # Glorot initialized Input/Output projection matrices
        B = jax.random.normal(subkey, shape=(lru_dim, hidden_dim)) / jnp.sqrt(
            hidden_dim + lru_dim
        )
        return B

    @staticmethod
    def mix_sequence(gamma, R, Us, reverse=False):
        """
        N - per head dimension
        Args:
            gammas: jax.Array(T,)
            As: jax.Array(T,N,N)
            Us: jax.array(T,B,N)
        Returns:
            out: jax.array(T,B,N)
        """

        def binf(a, b):
            gamma_i, thetas_i, acc_i = a
            gamma_j, thetas_j, acc_j = b
            # R_j@acc_i + acc_j
            # get [-x2, x1, -x4, x3,...]
            rotate_half_mat_i = jnp.stack(
                [-acc_i[..., 1::2], acc_i[..., 0::2]], axis=-1
            )
            shapes = list(rotate_half_mat_i.shape)[:-1]
            shapes[-1] *= 2
            rotate_half_mat_i = rotate_half_mat_i.reshape(shapes)
            # duplicate theta [o1, o1, o2, o2,...]
            shapes = list(thetas_j.shape)
            shapes[-1] *= 2
            theta = jnp.repeat(thetas_j[..., None], repeats=2, axis=-1).reshape(
                tuple(shapes)
            )
            sin = jnp.sin(theta)[..., None, :]  # add mock batch dimension
            cos = jnp.cos(theta)[..., None, :]  # add mock batch dimension
            acc = gamma_j[..., None, None] * (cos * acc_i + sin * rotate_half_mat_i)

            return (gamma_i * gamma_j, thetas_i + thetas_j, acc + acc_j)

        T = Us.shape[0]
        gammas = jnp.repeat(gamma[None, ...], repeats=T, axis=0)
        R = jnp.repeat(R[None, ...], repeats=T, axis=0)
        _, _, res = parallel_scan(binf, (gammas, R, Us), reverse=reverse)
        return res

    @nn.compact
    def __call__(self, input_sequence):
        # add dummy batch dimension for code
        x = input_sequence[None, ...]
        # print("Input: ", x.shape)
        # naming shortcut
        H, N = self.nheads, self.lru_dim // self.nheads
        assert N % 2 == 0, "N should be even"
        batch_sz, T, D = x.shape
        # log might not be necessary for theta
        thetas = self.param(
            "theta_log", self.theta_init, self.lru_dim // 2, self.max_phase
        ).reshape(H, N // 2)
        # P = self.param("P", self.ortho_mat_init, self.lru_dim, N).reshape(H, N, N)
        P = jnp.repeat(jnp.eye(N, N)[None], H, axis=0)
        B = self.param("B", self.mat_init, self.lru_dim, self.hidden_dim).reshape(
            H, N, self.hidden_dim
        )
        C = self.param("C", self.mat_init, self.hidden_dim, self.lru_dim)
        if self.bidirectional:
            C2 = self.param("C2", self.mat_init, self.hidden_dim, self.lru_dim)

        D = self.param(
            "D", lambda rng, H: jax.random.normal(rng, shape=(H,)), self.hidden_dim
        )
        gamma_log = self.param(
            "gamma_log", self.nu_log_init, H, r_min=self.r_min, r_max=self.r_max
        )

        # do not forget the double exponential
        gamma_log = -jnp.exp(gamma_log)
        gamma = jnp.exp(gamma_log)

        trace_per_head = jnp.trace(jnp.einsum("HDd,HAd->HDA", B, B), axis1=-2, axis2=-1)
        norm = jnp.sqrt((1 - gamma**2) / trace_per_head)  #  H / H elementwise -> H
        B_norm = jnp.einsum("H,HnD->HnD", norm, B)
        P = jax.scipy.linalg.expm(P - P.transpose(0, 2, 1))
        # apply P.T to Bx_t
        Us = jnp.einsum("HnD,BTD->HTBn", B_norm, x)
        Us = jnp.einsum("HnN,HTBn->HTBN", P.transpose(0, 2, 1), Us)
        # mix per head
        mix_head_fn = jax.vmap(self.mix_sequence, in_axes=(0, 0, 0, None), out_axes=0)
        thetas = jnp.exp(thetas)

        y = mix_head_fn(gamma, thetas, Us, False)  # H T B N
        # multiply P back to \tilde{x}_t
        y = jnp.einsum("HNn,HTBN->HTBn", P, y)

        if self.bidirectional:
            backward = mix_head_fn(gamma, thetas, Us, True)  # H T B N
            # multiply P back to \tilde{x}_t
            backward = jnp.einsum("HNn,HTBN->HTBn", P, backward)
            y = jnp.concatenate([y, backward], axis=-1)
            C = jnp.concatenate([C, C2], axis=-1)

        y = y.transpose(2, 1, 0, 3)  # H T B N -> B T H N
        y = jnp.einsum("Dn,BTn->BTD", C, y.reshape(batch_sz, T, -1)) + D * x
        # squeeze batch dimension
        return y[0]
