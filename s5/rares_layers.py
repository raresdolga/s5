import jax
from jax import numpy as jnp
import numpy as np
from flax import linen as nn
parallel_scan = jax.lax.associative_scan


class LRU(nn.Module):
    """
    Complex number implementation as in the original LRU paper
    """

    lru_dim: int  # N: state dimension
    hidden_dim: int  # H: model dimension
    r_min: float = 0.9
    r_max: float = 0.999
    max_phase: float = jnp.pi/10
    bidirectional: bool = True
    step_rescale: float = 1.0

    @staticmethod
    def theta_init(rng_key, lru_dim, max_phase=6.28):
        #print("Max_phase: ", max_phase)
        subkeys = jax.random.split(rng_key, num=3)
        u1 = jax.random.uniform(subkeys[0], shape=(lru_dim,))
        theta_log = jnp.log(max_phase * u1)
        return theta_log

    @staticmethod
    def nu_init(rng_key, lru_dim, r_min=0, r_max=1):
        #print("R_min: ", r_min, "r_max: ",  r_max)
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
        #inner_states = inner_states.transpose(1, 0, 2)  # LBN -> BLN
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
        #Lambda_elements = jnp.repeat(Lambda[None, ...], x.shape[1], axis=0)
        Lambda_elements = jnp.repeat(Lambda[None, ...], x.shape[0], axis=0)
        #Lambda_elements = Lambda_elements.transpose(1, 0, 2)  # BLN -> LBN

        # Bu_elements = jax.vmap(lambda u: u @ B_norm.T)(x) # LBN
        Bu_elements = jnp.einsum("LH,NH->LN", x, B_norm)

        # print(Lambda_elements.shape, Bu_elements.shape)
        inner_states = self.mix_sequence(Lambda_elements, Bu_elements, reverse=False)
        if self.bidirectional:
            C2 = C_re2 + 1j * C_im2
            C = jnp.concatenate([C, C2], axis=-1)
            backward =  self.mix_sequence(Lambda_elements, Bu_elements, reverse=True)
            inner_states = jnp.concatenate([inner_states, backward], axis=-1) # BLN -> BL2N
        
        y = jnp.einsum("HN,LN->LH", C, inner_states).real + jnp.einsum(
            "H,LH->LH", D, x
        )
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

    def theta_init(self, key,  N, max_phase):
        u2 = jax.random.uniform(key,shape = (N,))
        theta_log = jnp.log(max_phase*u2)
        return theta_log

    def nu_init(self,key, N, r_max, r_min ):
        u1 = jax.random.uniform(key, shape = (N,))
        nu_log = jnp.log(-0.5*jnp.log(u1*(r_max**2-r_min**2) + r_min**2))
        return nu_log
    
    def b_init(self, key, N,H):
        B_re = jax.random.normal(key, shape=(N,H))/np.sqrt(2*H)
        #B_im = np.random.normal(size=(N,H))/np.sqrt(2*H)
        return B_re
    
    def c_init(self, key, N,H):
        C_re = jax.random.normal(key, shape=(H,N))/np.sqrt(N)
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
        diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j*jnp.exp(theta_log))
        gamma_log = jnp.log(jnp.sqrt(1-jnp.abs(diag_lambda)**2))
        # Materializing the diagonal of Lambda and projections
        Lambda = jnp.exp(-jnp.exp(nu_log) + 1j*jnp.exp(theta_log))
        B_norm = (B_re + 1j*B_im) * jnp.expand_dims(jnp.exp(gamma_log), axis=-1)
        C = C_re + 1j*C_im

        # Running the LRU + output projection
        # For details on parallel scan, check discussion in Smith et al (2022).
        Lambda_elements = jnp.repeat(Lambda[None, ...], input_sequence.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(input_sequence)
        elements = (Lambda_elements, Bu_elements)
        _, inner_states = parallel_scan(binary_operator_diag, elements) # all x_k
        y = jax.vmap(lambda x, u: (C @ x).real + D * u)(inner_states, input_sequence)

        return y

    def init_lru_parameters(N, H, r_min=0, r_max=1, max_phase=6.28):
        """Initialize parameters of the LRU layer."""
        # N: state dimension, H: model dimension
        # Initialization of Lambda is complex valued distributed uniformly on ring
        # between r_min and r_max, with phase in [0, max_phase].
        u1 = np.random.uniform(size = (N,))
        u2 = np.random.uniform(size = (N,))
        nu_log = np.log(-0.5*np.log(u1*(r_max**2-r_min**2) + r_min**2))
        theta_log = np.log(max_phase*u2)

        # Glorot initialized Input/Output projection matrices
        B_re = np.random.normal(size=(N,H))/np.sqrt(2*H)
        B_im = np.random.normal(size=(N,H))/np.sqrt(2*H)
        C_re = np.random.normal(size=(H,N))/np.sqrt(N)
        C_im = np.random.normal(size=(H,N))/np.sqrt(N)
        D = np.random.normal(size=(H,))
        # Normalization factor
        diag_lambda = np.exp(-np.exp(nu_log) + 1j*np.exp(theta_log))
        gamma_log = np.log(np.sqrt(1-np.abs(diag_lambda)**2))
        return nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log


def binary_operator_diag(element_i, element_j):
    # Binary operator for parallel scan of linear recurrence.
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j