from catalog.metal_shader import kernel


def solve(params, grad, m, v, N, lr, beta1, beta2, eps, weight_decay, t):
    # All parameters share these bias corrections; evaluate the powers once.
    kernel(__file__, "adamw")(params, grad, m, v, lr, beta1, beta2, eps,
                              1 - lr * weight_decay, 1 / (1 - beta1 ** t),
                              1 / (1 - beta2 ** t), threads=N)
