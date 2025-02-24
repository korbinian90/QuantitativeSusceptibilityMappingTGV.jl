"""
qsm_tgv(phase, mask, res; TE, kwargs...)

Optional arguments:

Example:
    chi = qsm_tgv(randn(20,20,20), trues(20,20,20), [1,1,2]; TE=0.025, fieldstrength=3)

"""
function qsm_tgv(phase, mask, res; TE, B0_dir=[0, 0, 1], fieldstrength=3, regularization=2, alpha=get_default_alpha(regularization), step_size=3, iterations=get_default_iterations(res, step_size), erosions=3, dedimensionalize=false, correct_laplacian=true, laplacian=get_laplace_phase_del, type=Float32, gpu=nothing, nblocks=32, kernel=nothing)
    device, cu = select_device(gpu)
    phase, res, alpha, fieldstrength, mask = adjust_types(type, phase, res, alpha, fieldstrength, mask)

    for _ in 1:erosions
        mask = erode_mask(mask)
    end

    phase, mask, box_indices, original_size = reduce_to_mask_box(phase, mask)

    mask0 = erode_mask(mask) # one additional erosion in mask0
    laplace_phi0 = laplacian(phase, res)

    if dedimensionalize
        res, alpha, laplace_phi0 = de_dimensionalize(res, alpha, laplace_phi0)
    end

    if correct_laplacian
        laplace_phi0 .-= mean(laplace_phi0[mask0]) # mean correction to avoid background artefact
    end

    alphainv, tau, sigma, resinv, laplace_kernel, dipole_kernel = set_parameters(alpha, res, B0_dir, cu; kernel)

    laplace_phi0, mask, mask0 = cu(laplace_phi0), cu(mask), cu(mask0) # send to device

    chi, chi_, w, w_, phi, phi_, eta, p, q = initialize_device_variables(type, size(laplace_phi0), cu)

    ndrange = size(laplace_phi0)

    if iterations isa AbstractArray
        max_iterations = iterations[end]
        ret = []
    else
        max_iterations = iterations
        ret = nothing
    end

    @showprogress dt=1 desc="Running $max_iterations TGV iterations..." for k in 1:max_iterations
        #############
        # dual update
        KernelAbstractions.synchronize(device)
        update_eta_kernel!(device, nblocks)(eta, phi_, chi_, laplace_phi0, mask0, sigma, laplace_kernel, dipole_kernel; ndrange)
        update_p_kernel!(device, nblocks)(p, chi_, w_, mask, mask0, sigma * step_size, alphainv[2], resinv; ndrange)
        update_q_kernel!(device, nblocks)(q, w_, mask0, sigma * step_size, alphainv[1], resinv; ndrange)

        #######################
        # swap primal variables
        (phi_, phi) = (phi, phi_)
        (chi_, chi) = (chi, chi_)
        (w_, w) = (w, w_)

        ###############
        # primal update
        KernelAbstractions.synchronize(device)
        update_phi_kernel!(device, nblocks)(phi, phi_, eta, mask, mask0, tau, laplace_kernel; ndrange)
        update_chi_kernel!(device, nblocks)(chi, chi_, eta, p, mask0, tau * step_size, resinv, dipole_kernel; ndrange)
        update_w_kernel!(device, nblocks)(w, w_, p, q, mask, mask0, tau * step_size, resinv; ndrange)

        #####################
        # extragradient update
        KernelAbstractions.synchronize(device)
        extragradient_update!(phi_, phi)
        extragradient_update!(chi_, chi)
        extragradient_update!(w_, w)

        if iterations isa AbstractArray && k in iterations
            res_chi = zeros(type, original_size)
            view(res_chi, box_indices...) .= scale(Array(chi), TE, fieldstrength)
            push!(ret, res_chi)
        end
    end

    if iterations isa AbstractArray
        return ret
    end

    # Assign result to full size output array
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

function get_default_iterations(res, step_size)
    # Heuristic formula
    it = 3200 # default for res=[1,1,1]
    min_iterations = 1000 # even low res data needs at least 1000 iterations
    it = max(min_iterations, it / prod(res)^0.42)
    it /= step_size^0.6 # increasing the step size reduces the required iterations
    return round(Int, it)
end

function get_default_alpha(regularization)
    # Heuristic formula, regularization=2 is default, 1 is low regularization, 3 is high regularization
    alpha = [0.001, 0.001] + [0.001, 0.002] * (regularization - 1)
    alpha = max.(alpha, 0)
    return alpha
end

function de_dimensionalize(res, alpha, laplace_phi0)
    res_corr = prod(res)^(-1 / 3)
    res = res .* res_corr
    alpha = (alpha[1] * res_corr^2, alpha[2] * res_corr)
    laplace_phi0 ./= res_corr^2

    return res, alpha, laplace_phi0
end

function dipole_kernel_orig(res)
    dipole_kernel = zeros(Float32, 3, 3, 3)
    dipole_kernel[3, 2, 2] = 1/3 / res[1]^2
    dipole_kernel[2, 3, 2] = 1/3 / res[2]^2
    dipole_kernel[2, 2, 3] = -2/3 / res[3]^2
    dipole_kernel[2, 2, 1] = -2/3 / res[3]^2
    dipole_kernel[2, 1, 2] = 1/3 / res[2]^2
    dipole_kernel[1, 2, 2] = 1/3 / res[1]^2    
    dipole_kernel[2, 2, 2] = -sum(dipole_kernel)
    return dipole_kernel
end

function set_parameters(alpha, res, B0_dir, cu; kernel=nothing)
    if kernel isa AbstractArray
        dipole_kernel = kernel

        grad_norm_sqr = 4 * sum(res .^ -2)
        grad_norm = sqrt(grad_norm_sqr)
        wave_norm = sum(abs.(dipole_kernel))
        norm_matrix = [0 grad_norm 1; 0 0 grad_norm; grad_norm_sqr wave_norm 0]
        F = svd(norm_matrix)
        norm_sqr = first(F.S)^2    
    elseif kernel == :original
        dipole_kernel = dipole_kernel_orig(res)
        grad_norm_sqr = 4 * sum(res .^ -2)
        norm_sqr = 2 * grad_norm_sqr^2 + 1
    else # default
        dipole_kernel = stencil(; st=27, direction=B0_dir, res=res)

        grad_norm_sqr = 4 * sum(res .^ -2)
        grad_norm = sqrt(grad_norm_sqr)
        wave_norm = sum(abs.(dipole_kernel))
        norm_matrix = [0 grad_norm 1; 0 0 grad_norm; grad_norm_sqr wave_norm 0]
        F = svd(norm_matrix)
        norm_sqr = first(F.S)^2
    end

    alphainv = 1 ./ alpha

    tau = 1 / sqrt(norm_sqr)
    sigma = (1 / norm_sqr) / tau # always identical to tau

    resinv = cu(1 ./ res)
    laplace_kernel = cu(res .^ -2)

    dipole_kernel = cu(dipole_kernel)

    return alphainv, tau, sigma, resinv, laplace_kernel, dipole_kernel
end

function adjust_types(type, laplace_phi_0, res, alpha, fieldstrength, mask)
    type.(laplace_phi_0), collect(type.(abs.(res))), collect(type.(alpha)), type(fieldstrength), mask .!= 0
end

function reduce_to_mask_box(laplace_phi0, mask)
    if sum(mask) == 0
        error("Mask is empty")
    end
    original_size = size(laplace_phi0)
    box_indices = mask_box_indices(mask)
    laplace_phi0 = view(laplace_phi0, box_indices...)
    mask = view(mask, box_indices...)

    return laplace_phi0, mask, box_indices, original_size
end

function select_device(library)
    if Symbol(library) == :CUDA
        println("Using the GPU via CUDA")
        return library.CUDAKernels.CUDABackend(), library.cu
    elseif Symbol(library) == :AMDGPU
        println("Using the GPU via AMDGPU")
        return library.ROCKernels.ROCBackend(), library.ROCArray
    elseif Symbol(library) == :oneAPI
        println("Using the GPU via oneAPI")
        return library.oneAPIKernels.oneAPIBackend(), library.oneArray
    elseif Symbol(library) == :Metal
        println("Using the GPU via Metal")
        return library.MetalKernels.MetalBackend(), library.MtlArray
    else
        println("Using $(Threads.nthreads()) CPU threads")
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
    function get_range(mask, dim)
        dims = [(2, 3), (1, 3), (1, 2)][dim]
        mask_projection = dropdims(reduce(max, mask; dims); dims)
        first = max(findfirst(mask_projection) - 1, 1)
        last = min(findlast(mask_projection) + 1, size(mask, dim))
        return first:last
    end
    return [get_range(mask, dim) for dim in 1:3]
end
