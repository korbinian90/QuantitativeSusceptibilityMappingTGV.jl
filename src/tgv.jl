qsm_tgv(phase, mask, res; kw...) = qsm_tgv_laplacian(get_laplace_phase3(phase, res), mask, res; kw...)
function qsm_tgv_laplacian(laplace_phi0, mask, res; TE, omega=[0, 0, 1], fieldstrength=3, alpha=[0.003, 0.001], iterations=3000, erosions=3, type=Float32, gpu=CUDA.functional(), nblocks=32)
    device, cu = select_device(gpu)
    laplace_phi0, res, alpha, fieldstrength, mask = adjust_types(type, laplace_phi0, res, alpha, fieldstrength, mask)

    for _ in 1:erosions
        mask = erode_mask(mask)
    end

    iterations = adjust_iterations(iterations, res)

    laplace_phi0, mask, box_indices, original_size = reduce_to_mask_box(laplace_phi0, mask)

    res, alpha, laplace_phi0 = de_dimensionalize(res, alpha, laplace_phi0)
    
    mask0 = erode_mask(mask) # one additional erosion in mask0

    laplace_phi0 .-= mean(laplace_phi0[mask0]) # mean correction to avoid background artefact

    alphainv, tau, sigma, resinv, laplace_kernel, dipole_kernel = set_parameters(alpha, res, omega, cu)

    laplace_phi0, mask, mask0 = cu(laplace_phi0), cu(mask), cu(mask0) # send to device

    chi, chi_, w, w_, phi, phi_, eta, p, q = initialize_device_variables(type, size(laplace_phi0), cu)

    ndrange = size(laplace_phi0)
    @showprogress 1 "Running $iterations TGV iterations..." for k in 1:iterations

        #############
        # dual update
        KernelAbstractions.synchronize(device)
        update_eta_kernel!(device, nblocks)(eta, phi_, chi_, laplace_phi0, mask0, sigma, laplace_kernel, dipole_kernel; ndrange)
        update_p_kernel!(device, nblocks)(p, chi_, w_, mask, mask0, sigma, alphainv[2], resinv; ndrange)
        update_q_kernel!(device, nblocks)(q, w_, mask0, sigma, alphainv[1], resinv; ndrange)

        #######################
        # swap primal variables
        (phi_, phi) = (phi, phi_)
        (chi_, chi) = (chi, chi_)
        (w_, w) = (w, w_)

        ###############
        # primal update
        KernelAbstractions.synchronize(device)
        update_phi_kernel!(device, nblocks)(phi, phi_, eta, mask, mask0, tau, laplace_kernel; ndrange)
        update_chi_kernel!(device, nblocks)(chi, chi_, eta, p, mask0, tau, resinv, dipole_kernel; ndrange)
        update_w_kernel!(device, nblocks)(w, w_, p, q, mask, mask0, tau, resinv; ndrange)
        #####################
        # extragradient update

        KernelAbstractions.synchronize(device)
        extragradient_update!(phi_, phi)
        extragradient_update!(chi_, chi)
        extragradient_update!(w_, w)
    end

    res_chi = zeros(type, original_size)
    view(res_chi, box_indices...) .= scale(Array(chi), TE, fieldstrength)
    return res_chi
end

function initialize_device_variables(type, sz, cu)
    # initialize primal variables
    chi = cu(zeros(type, sz))
    chi_ = cu(zeros(type, sz))
    w = cu(zeros(type, (3, sz...)))
    w_ = cu(zeros(type, (3, sz...)))
    phi = cu(zeros(type, sz))
    phi_ = cu(zeros(type, sz))
    # initialize dual variables
    eta = cu(zeros(type, sz))
    p = cu(zeros(type, (3, sz...)))
    q = cu(zeros(type, (6, sz...)))

    return chi, chi_, w, w_, phi, phi_, eta, p, q
end

function adjust_iterations(iterations, res)
    iterations = round(Int, iterations / sqrt(prod(res)))
    return iterations
end

function de_dimensionalize(res, alpha, laplace_phi0)
    res_corr = prod(res)^(-1 / 3)
    res = res .* res_corr
    alpha = (alpha[1] * res_corr^2, alpha[2] * res_corr)
    laplace_phi0 ./= res_corr^2

    return res, alpha, laplace_phi0
end

function set_parameters(alpha, res, omega, cu)
    alphainv = 1 ./ alpha

    grad_norm_sqr = 4 * (sum(res .^ -2))
    norm_sqr = 2 * grad_norm_sqr^2 + 1
    tau = 1 / sqrt(norm_sqr)
    sigma = (1 / norm_sqr) / tau # TODO always identical to tau
    
    resinv = cu(1 ./ res)
    laplace_kernel = cu(res .^ -2)
    dipole_kernel = cu((1 / 3 .- omega .^ 2) ./ (res .^ 2))

    return alphainv, tau, sigma, resinv, laplace_kernel, dipole_kernel
end

function adjust_types(type, laplace_phi_0, res, alpha, fieldstrength, mask)
    type.(laplace_phi_0), collect(type.(abs.(res))), collect(type.(alpha)), type(fieldstrength), mask .!= 0
end

function reduce_to_mask_box(laplace_phi0, mask)
    original_size = size(laplace_phi0)
    box_indices = mask_box_indices(mask)
    laplace_phi0 = view(laplace_phi0, box_indices...)
    mask = view(mask, box_indices...)

    return laplace_phi0, mask, box_indices, original_size
end

function select_device(gpu)
    if gpu
        println("Using the GPU")
        return CUDA.CUDAKernels.CUDABackend(), CUDA.cu
    else
        println("Using $(Threads.nthreads()) CPU cores")
        return CPU(), identity
    end
end

function erode_mask(mask)
    SE = strel(CartesianIndex, strel_diamond(mask))
    erode_vox(I) = minimum(mask[I+J] for J in SE if checkbounds(Bool, mask, I + J))
    return [erode_vox(I) for I in eachindex(IndexCartesian(), mask)]
end

function scale(chi, TE, fieldstrength)
    GAMMA = 42.5781
    chi .*= 1 / (2pi * TE * fieldstrength * GAMMA)
    return chi
end

# get indices for the smallest box that contains the mask
function mask_box_indices(mask)
    function get_range(mask, dims)
        mask_projection = dropdims(reduce(max, mask; dims); dims)
        return findfirst(mask_projection):findlast(mask_projection)
    end
    return [get_range(mask, dims) for dims in [(2, 3), (1, 3), (1, 2)]]
end

function get_laplace_phase3(phase, res)
    # pad phase
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

function get_best_local_h1(dx; axis)
    F_shape = [size(dx)..., 3, 3]
    len = F_shape[axis]
    F_shape[axis] -= 1

    F = zeros(eltype(dx), F_shape...)
    for i in -1:1, j in -1:1
        F[:, :, :, i+2, j+2] = (selectdim(dx, axis, 1:len-1) .- (2pi * i)) .^ 2 .+ (selectdim(dx, axis, 2:len) .+ (2pi * j)) .^ 2
    end

    dims = (4, 5)
    G = dropdims(argmin(F; dims); dims)
    I = getindex.(G, 4) .- 2
    J = getindex.(G, 5) .- 2

    return I, J
end
