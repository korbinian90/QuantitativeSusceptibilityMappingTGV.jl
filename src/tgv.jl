function qsm_tgv(laplace_phi0, mask, res, omega; alpha=(0.2, 0.1), iterations=1000)

    mask0 = copy(mask)

    # initialize primal variables
    chi = zeros(size(laplace_phi0))
    chi_ = zeros(size(laplace_phi0))

    w = zeros((size(laplace_phi0)..., 3))
    w_ = zeros((size(laplace_phi0)..., 3))

    phi = zeros(size(laplace_phi0))
    phi_ = zeros(size(laplace_phi0))

    # initialize dual variables
    eta = zeros(size(laplace_phi0))
    p = zeros((size(laplace_phi0)..., 3))
    q = zeros((size(laplace_phi0)..., 6))

    # estimate squared norm
    grad_norm_sqr = 4 * (sum(res .^ -2))

    norm_sqr = 2 * grad_norm_sqr^2 + 1

    # set regularization parameters
    alpha1 = alpha[2]
    alpha0 = alpha[1]

    # initialize resolution
    res = abs.(res)

    for k in 1:iterations

        tau = 1 / sqrt(norm_sqr)
        sigma = (1 / norm_sqr) / tau

        #############
        # dual update

        tgv_update_eta!(eta, phi_, chi_, laplace_phi0, mask0, sigma, res, omega)

        tgv_update_p!(p, chi_, w_, mask, mask0, sigma, alpha1, res)

        tgv_update_q!(q, w_, mask0, sigma, alpha0, res)

        #######################
        # swap primal variables

        (phi_, phi) = (phi, phi_)
        (chi_, chi) = (chi, chi_)
        (w_, w) = (w, w_)

        ###############
        # primal update

        tgv_update_phi!(phi, phi_, eta, mask, mask0, tau, res)

        tgv_update_chi!(chi, chi_, eta, p, mask0, tau, res, omega)

        tgv_update_w!(w, w_, p, q, mask, mask0, tau, res)

        ######################
        # extragradient update

        extragradient_update(phi_, phi)
        extragradient_update(chi_, chi)
        extragradient_update(w_, w)
    end

    return chi
end
