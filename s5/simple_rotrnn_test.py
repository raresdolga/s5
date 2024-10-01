import jax
import numpy as np
from jax import numpy as jnp

parallel_scan = jax.lax.associative_scan


def forward(rotrnn_params, input_sequence):
    """Forward pass through the RotRNN layer"""

    thetas, gamma_log, M, B, C, D = rotrnn_params
    gammas = jnp.exp(-jnp.exp(gamma_log))

    T, dim_u = input_sequence.shape

    # compute \xi and normalise B
    B_T_B = jax.vmap(lambda a, b: a @ b)(B.transpose(0, 2, 1), B)
    B_T_B_trace = jnp.trace(B_T_B, axis1=1, axis2=2)
    xi = jnp.sqrt((1 - gammas.squeeze() ** 2) / B_T_B_trace)
    # B_norm = jnp.einsum("H, HTD -> HTD", xi, B)
    B_norm = xi[..., None, None] * B

    # create orthogonal matrix P from weight matrix M
    P = jax.scipy.linalg.expm(M - M.transpose(0, 2, 1))

    # project inputs onto heads
    x = jnp.einsum("HDi,Ti->HTD", B_norm, input_sequence)

    # multiply from the right by P^T
    x = jax.vmap(lambda a, b: a @ b)(x, P.transpose(0, 2, 1))

    # compute recurrence parallelised over heads
    gammas = jnp.repeat(gammas[:, None], repeats=T, axis=1)
    thetas = jnp.repeat(thetas[:, None], repeats=T, axis=1)
    rec_fn = jax.vmap(
        lambda a, b, c: parallel_scan(binf, (a, b, c)),
        in_axes=(0, 0, 0),
        out_axes=0,
    )
    x = rec_fn(gammas, thetas, x)[2]

    # multiply from the left by P
    x = jnp.einsum("HDi, HTi -> HTD", P, x)
    return x

    # concatenate heads
    x = x.transpose(1, 0, 2).reshape(T, -1)

    # apply output projection/head mixing and skip connection
    y = jax.vmap(lambda a: C @ a)(x) + D * input_sequence
    return y


def init_params(H, dim_x, dim_u, gamma_min, gamma_max, theta_max):
    """Initialise the learnable parameters"""

    dim_h = dim_x // H

    # random initialisation of \theta in [0, theta_max]
    theta = np.random.uniform(0, theta_max, (H, dim_h // 2))

    # constrained initialisation of \gamma in [gamma_min, gamma_max]
    u1 = np.random.uniform(size=(H, 1))
    gamma_log = jnp.log(
        -0.5 * jnp.log(u1 * (gamma_max**2 - gamma_min**2) + gamma_min**2)
    )

    # Glorot initialised input/output matrices
    B = np.random.normal(size=(H, dim_h, dim_u)) / np.sqrt(dim_u)
    C = np.random.normal(size=(dim_u, dim_x)) / np.sqrt(dim_x)

    # Orthogonal weight matrix M
    M = np.random.normal(size=(H, dim_h, dim_h))

    # D is random vector applied element-wise to u
    D = np.random.normal(size=(dim_u))

    return theta, gamma_log, M, B, C, D


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # initialise parameters
    H = 4
    dim_x = 16
    dim_u = 2
    gamma_min = 0.1
    gamma_max = 0.9
    theta_max = 2 * np.pi

    rotrnn_params = init_params(H, dim_x, dim_u, gamma_min, gamma_max, theta_max)

    # create a random input sequence
    T = 10000
    input_sequence = np.random.normal(size=(T, dim_u))

    # run the forward pass
    y = forward(rotrnn_params, input_sequence)

    y = jnp.linalg.norm(y, axis=-1)
    print(y.mean())
    exit()
    # plot the output
    plt.plot(y)
    plt.show()
