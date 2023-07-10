function qsm_full(phase, mask, res; kw...)
    laplace_phi0 = laplacian(phase, res)
    return qsm_tgv(laplace_phi0, mask, res; kw...)
end

function qsm_tgv(laplace_phi0, mask, res; TE, fieldstrength=3, alpha=(0.003, 0.001), iterations=1000, erosions=3, type=Float32, gpu=false, nblocks=32)
    device, cu = if gpu
        println("Using the GPU")
        CUDA.CUDAKernels.CUDABackend(), CUDA.cu
    else
        println("Using $(Threads.nthreads()) CPU cores")
        CPU(), identity
    end

    for _ in 1:erosions
        mask = erode_mask(mask)
    end

    res_chi = zeros(type, size(laplace_phi0))
    # get smaller views for only the area inside the mask
    cut_indices = mask_box_indices(mask)
    laplace_phi0 = view(laplace_phi0, cut_indices...)
    mask = view(mask, cut_indices...)
    res_chi_view = view(res_chi, cut_indices...)

    laplace_phi0 = cu(copy(laplace_phi0))

    mask0 = cu(erode_mask(mask))
    mask = cu(mask)
    # initialize primal variables
    chi = cu(zeros(type, size(laplace_phi0)))
    chi_ = cu(zeros(type, size(laplace_phi0)))

    w = cu(zeros(type, (3, size(laplace_phi0)...)))
    w_ = cu(zeros(type, (3, size(laplace_phi0)...)))

    phi = cu(zeros(type, size(laplace_phi0)))
    phi_ = cu(zeros(type, size(laplace_phi0)))

    # initialize dual variables
    eta = cu(zeros(type, size(laplace_phi0)))
    p = cu(zeros(type, (3, size(laplace_phi0)...)))
    q = cu(zeros(type, (6, size(laplace_phi0)...)))

    res = type.(abs.(res))

    # de-dimensionalize
    res_corr = prod(res)^(-1 / 3)
    res .*= res_corr
    alpha = collect(type.(alpha))
    alpha[2] *= res_corr
    alpha[1] *= res_corr .^ 2
    laplace_phi0 ./= res_corr .^ 2
    laplace_phi0 .-= mean(laplace_phi0[mask])

    # estimate squared norm
    grad_norm_sqr = 4 * (sum(res .^ -2))

    norm_sqr = 2 * grad_norm_sqr^2 + 1

    # set regularization parameters
    alpha1 = type(alpha[2])
    alpha0 = type(alpha[1])

    # initialize resolution
    tau = 1 / sqrt(norm_sqr)
    sigma = (1 / norm_sqr) / tau # TODO they are always identical

    resinv = cu(1 ./ res)
    resinv2 = cu(res .^ -2)
    resinv_d2 = cu(0.5 ./ res)
    wresinv2 = cu([1 / 3, 1 / 3, -2 / 3] ./ (res .^ 2))

    KernelAbstractions.synchronize(device)

    @showprogress 1 "Running $iterations TGV iterations..." for k in 1:iterations

        #############
        # dual update
        tgv_update_eta!(eta, phi_, chi_, laplace_phi0, mask0, sigma, resinv2, wresinv2, device, nblocks)
        # KernelAbstractions.synchronize(device)
        tgv_update_p!(p, chi_, w_, mask, mask0, sigma, alpha1, resinv, device, nblocks)
        # KernelAbstractions.synchronize(device)
        tgv_update_q!(q, w_, mask0, sigma, alpha0, resinv, resinv_d2, device, nblocks)
        KernelAbstractions.synchronize(device)

        #######################
        # swap primal variables
        (phi_, phi) = (phi, phi_)
        (chi_, chi) = (chi, chi_)
        (w_, w) = (w, w_)

        ###############
        # primal update
        tgv_update_phi!(phi, phi_, eta, mask, mask0, tau, resinv2, device, nblocks)
        # KernelAbstractions.synchronize(device)
        tgv_update_chi!(chi, chi_, eta, p, mask0, tau, resinv, wresinv2, device, nblocks)
        # KernelAbstractions.synchronize(device)
        tgv_update_w!(w, w_, p, q, mask, mask0, tau, resinv, device, nblocks)
        KernelAbstractions.synchronize(device)
        ######################
        # extragradient update

        extragradient_update!(phi_, phi)
        extragradient_update!(chi_, chi)
        extragradient_update!(w_, w)
        KernelAbstractions.synchronize(device)
    end

    res_chi_view .= Array(chi) ./ scale(TE, fieldstrength)

    return res_chi
end

function laplacian(phase, res)
    sz = size(phase)
    padded_indices = (0:sz[1]+1, 0:sz[2]+1, 0:sz[3]+1)
    phase_pad = PaddedView(0, phase, padded_indices)

    laplace_phi = rem2pi.(-2.0 * phase_pad[1:end-1, 1:end-1, 1:end-1] +
                          (phase_pad[0:end-2, 1:end-1, 1:end-1]) +
                          (phase_pad[2:end, 1:end-1, 1:end-1]), RoundNearest) / (res[1]^2)

    laplace_phi += rem2pi.(-2.0 * phase_pad[1:end-1, 1:end-1, 1:end-1] +
                           (phase_pad[1:end-1, 0:end-2, 1:end-1]) +
                           (phase_pad[1:end-1, 2:end, 1:end-1]), RoundNearest) / (res[2]^2)

    laplace_phi += rem2pi.(-2.0 * phase_pad[1:end-1, 1:end-1, 1:end-1] +
                           (phase_pad[1:end-1, 1:end-1, 0:end-2]) +
                           (phase_pad[1:end-1, 1:end-1, 2:end]), RoundNearest) / (res[3]^2)
    return laplace_phi
end

function get_laplace_phase3(phase, res)
    #pad phase
    sz = size(phase)
    padded_indices = (0:sz[1]+1, 0:sz[2]+1, 0:sz[3]+1)
    phase = PaddedView(0, phase, padded_indices)

    dx = phase[1:end, 1:end-1, 1:end-1] .- phase[0:end-1, 1:end-1, 1:end-1]
    dy = phase[1:end-1, 1:end, 1:end-1] .- phase[1:end-1, 0:end-1, 1:end-1]
    dz = phase[1:end-1, 1:end-1, 1:end] .- phase[1:end-1, 1:end-1, 0:end-1]

    (Ix, Jx) = get_best_local_h1(dx, axis=1)
    (Iy, Jy) = get_best_local_h1(dy, axis=2)
    (Iz, Jz) = get_best_local_h1(dz, axis=3)

    laplace_phi = (-2.0 * phase[1:end-1, 1:end-1, 1:end-1] +
                   (phase[0:end-2, 1:end-1, 1:end-1] + 2 * pi * Ix) +
                   (phase[2:end, 1:end-1, 1:end-1] + 2 * pi * Jx)) / (res[1]^2)

    laplace_phi += (-2.0 * phase[1:end-1, 1:end-1, 1:end-1] +
                    (phase[1:end-1, 0:end-2, 1:end-1] + 2 * pi * Iy) +
                    (phase[1:end-1, 2:end, 1:end-1] + 2 * pi * Jy)) / (res[2]^2)

    laplace_phi += (-2.0 * phase[1:end-1, 1:end-1, 1:end-1] +
                    (phase[1:end-1, 1:end-1, 0:end-2] + 2 * pi * Iz) +
                    (phase[1:end-1, 1:end-1, 2:end] + 2 * pi * Jz)) / (res[3]^2)

    return laplace_phi
end

function get_best_local_h1(dx; axis=1)
    F_shape = collect(size(dx))
    F_shape[axis] -= 1
    push!(F_shape, 3)
    push!(F_shape, 3)

    F = zeros(eltype(dx), Tuple(F_shape))
    for i in -1:1
        for j in -1:1
            F[:, :, :, i+2, j+2] =
                if axis == 1
                    (dx[1:end-1, :, :] .- (2pi * i)) .^ 2 + (dx[2:end, :, :] .+ (2pi * j)) .^ 2
                elseif axis == 2
                    (dx[:, 1:end-1, :] .- (2pi * i)) .^ 2 + (dx[:, 2:end, :] .+ (2pi * j)) .^ 2
                elseif axis == 3
                    (dx[:, :, 1:end-1] .- (2pi * i)) .^ 2 + (dx[:, :, 2:end] .+ (2pi * j)) .^ 2
                end
        end
    end

    G = argmin(F; dims=(4, 5))
    G = dropdims(G; dims=(4, 5))
    I = getindex.(G, 4) .- 2
    J = getindex.(G, 5) .- 2

    return I, J
end

function erode_mask(mask)
    SE = strel(CartesianIndex, strel_diamond(mask))
    erode_vox(I) = minimum(mask[I+J] for J in SE if checkbounds(Bool, mask, I + J))
    return [erode_vox(I) for I in eachindex(IndexCartesian(), mask)]
end

function scale(TE, fieldstrength)
    GAMMA = 42.5781
    return 2pi * TE * fieldstrength * GAMMA
end

# get indices for the smallest box that contains the mask
function mask_box_indices(mask)
    function get_range(mask, dims)
        mask_projection = dropdims(reduce(max, mask; dims); dims)
        return findfirst(mask_projection):findlast(mask_projection)
    end
    return [get_range(mask, dims) for dims in [(2, 3), (1, 3), (1, 2)]]
end
