function qsm_tgv(laplace_phi0, mask, res, omega; alpha=(0.2, 0.1), iterations=1000, type=Float32, gpu=false)
    device, cu = if gpu
        CUDADevice(), CUDA.cu
    else
        CPU(), identity
    end

    laplace_phi0 = cu(copy(laplace_phi0))

    mask0 = cu(copy(mask))
    mask = cu(mask)
    # initialize primal variables
    chi = cu(zeros(type, size(laplace_phi0)))
    chi_ = cu(zeros(type, size(laplace_phi0)))

    w = cu(zeros(type, (size(laplace_phi0)..., 3)))
    w_ = cu(zeros(type, (size(laplace_phi0)..., 3)))

    phi = cu(zeros(type, size(laplace_phi0)))
    phi_ = cu(zeros(type, size(laplace_phi0)))

    # initialize dual variables
    eta = cu(zeros(type, size(laplace_phi0)))
    p = cu(zeros(type, (size(laplace_phi0)..., 3)))
    q = cu(zeros(type, (size(laplace_phi0)..., 6)))

    res_corr = prod(res)^(-1 / 3)
    res = collect(type.(abs.(res))) * res_corr
    alpha = collect(type.(alpha))
    alpha[2] *= res_corr
    alpha[1] *= res_corr .^ 2
    laplace_phi0 ./= res_corr .^ 2

    omega = type.(omega)
    # estimate squared norm
    grad_norm_sqr = 4 * (sum(res .^ -2))

    norm_sqr = 2 * grad_norm_sqr^2 + 1

    # set regularization parameters
    alpha1 = type(alpha[2])
    alpha0 = type(alpha[1])

    # initialize resolution
    qx_alloc = cu(zeros(type, size(w)))
    qy_alloc = cu(zeros(type, size(w)))
    qz_alloc = cu(zeros(type, size(w)))

    res_inv_dim4 = cu(reshape(res .^ -1, 1, 1, 1, 3))

    synchronize()
    for k in 1:iterations

        tau = 1 / sqrt(norm_sqr)
        sigma = (1 / norm_sqr) / tau # TODO they are always identical

        #############
        # dual update

        thread_eta = tgv_update_eta!(eta, phi_, chi_, laplace_phi0, mask0, sigma, res, omega; cu, device)

        thread_p = tgv_update_p!(p, chi_, w_, mask, mask0, sigma, alpha1, res; cu, device)

        thread_q = tgv_update_q!(q, w_, mask0, sigma, alpha0, res; cu, device)
        wait(thread_eta)
        wait(thread_p)
        wait(thread_q)

        #######################
        # swap primal variables

        (phi_, phi) = (phi, phi_)
        (chi_, chi) = (chi, chi_)
        (w_, w) = (w, w_)

        ###############
        # primal update
        thread_phi = tgv_update_phi!(phi, phi_, eta, mask, mask0, tau, res; cu, device)

        thread_chi = tgv_update_chi!(chi, chi_, eta, p, mask0, tau, res, omega; cu, device)

        thread_w = tgv_update_w!(w, w_, p, q, mask, mask0, tau, res, res_inv_dim4, qx_alloc, qy_alloc, qz_alloc; cu, device)
        wait(thread_phi)
        wait(thread_chi)
        wait(thread_w)

        ######################
        # extragradient update

        @async extragradient_update(phi_, phi)
        @async extragradient_update(chi_, chi)
        @async extragradient_update(w_, w)
        synchronize()
    end

    return chi
end
