function qsm_full(phase, mask, res; kw...)
    laplace_phi0 = laplacian(phase, res)
    return qsm_tgv(laplace_phi0, mask, res; kw...)
end

function qsm_tgv(laplace_phi0, tensor, mask, res; TE, fieldstrength=3, omega=[0, 0, 1], alpha=(0.0015, 0.0005), sigma=0.8, iterations=1000, erosions=3, type=Float32, gpu=false)
    device, cu = if gpu
        CUDADevice(), CUDA.cu
    else
        CPU(), identity
    end

    mask_orig = copy(mask)
    for _ in 1:erosions
        mask = erode_mask(mask)
    end

    tensor = cu(tensor)
    laplace_phi0 = cu(copy(laplace_phi0))
    mask0 = cu(erode_mask(mask))
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

    res = type.(abs.(res))
    # res_corr = prod(res)^(-1 / 3)
    # res = collect(type.(abs.(res))) * res_corr
    # alpha = collect(type.(alpha))
    # alpha[2] *= res_corr
    # alpha[1] *= res_corr .^ 2
    # laplace_phi0 ./= res_corr .^ 2

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
    @time for k in 1:iterations

        tau = 1 / sqrt(norm_sqr)
        sigma = (1 / norm_sqr) / tau # TODO they are always identical

        #############
        # dual update

        thread_eta = tgv_update_eta!(eta, phi_, chi_, laplace_phi0, mask0, sigma, res, omega; cu, device)
        thread_p = tgv_update_p!(p, chi_, w_, tensor, mask, mask0, sigma, alpha1, res; cu, device)
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
        thread_chi = tgv_update_chi!(chi, chi_, eta, p, tensor, mask0, tau, res, omega; cu, device)
        thread_w = tgv_update_w!(w, w_, p, q, tensor, mask, mask0, tau, res, res_inv_dim4, qx_alloc, qy_alloc, qz_alloc; cu, device)
        wait(thread_phi)
        wait(thread_chi)
        wait(thread_w)

        ######################
        # extragradient update

        @sync begin
            @async extragradient_update(phi_, phi)
            @async extragradient_update(chi_, chi)
            @async extragradient_update(w_, w)
        end
    end

    return chi ./ scale(TE, fieldstrength)
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

function erode_mask(mask, erosions)
    mask = copy(mask)
    for _ in 1:erosions
        mask = erode_mask(mask)
    end
    return mask
end

function scale(TE, fieldstrength)
    GAMMA = 42.5781
    return 2pi * TE * fieldstrength * GAMMA
end

function st_gauss(magnitude,mask, mask_orig,sigma; type=Float32)
    N=size(magnitude)
    ND=N[1]*N[2]*N[3]
    @time begin
    x_sobel = -Kernel.sobel((true,true,true), 1)[1]
    y_sobel = -Kernel.sobel((true,true,true), 2)[1]
    z_sobel = -Kernel.sobel((true,true,true), 3)[1]
          
    dex = imfilter(magnitude, x_sobel) .* mask
    dey = imfilter(magnitude, y_sobel) .* mask
    dez = imfilter(magnitude, z_sobel) .* mask

    #orientation tensor
    J011=dex.*dex
    J012=dex.*dey
    J013=dex.*dez
    J022=dey.*dey
    J023=dey.*dez
    J033=dez.*dez
    end

    #structure tensor independent dimentions
    @time begin
    gaussian = Kernel.gaussian((sigma, sigma, sigma), (9,9,9))
    J11_smooth=imfilter(J011, gaussian, "symmetric")
    J12_smooth=imfilter(J012, gaussian, "symmetric")
    J13_smooth=imfilter(J013, gaussian, "symmetric")
    J22_smooth=imfilter(J022, gaussian, "symmetric")
    J23_smooth=imfilter(J023, gaussian, "symmetric")
    J33_smooth=imfilter(J033, gaussian, "symmetric")
    end
    #structure tensor in matrix 3x3
    @time begin
    J=zeros(type, (N[1],N[2],N[3],3,3))
    J[:,:,:,1,1]=J11_smooth
    J[:,:,:,1,2]=J12_smooth
    J[:,:,:,1,3]=J13_smooth
    J[:,:,:,2,1]=J12_smooth
    J[:,:,:,2,2]=J22_smooth
    J[:,:,:,2,3]=J23_smooth
    J[:,:,:,3,1]=J13_smooth
    J[:,:,:,3,2]=J23_smooth
    J[:,:,:,3,3]=J33_smooth
    end
    # eigen decomposition for eigenvalues modification
    @time v, w = eigen_decomposition(J)
    # @time eig = mapslices(eigen, J; dims=(4,5))
    # eig = dropdims(eig; dims=(4,5))
    # v = [[e.vectors[i,j] for e in eig] for i in 1:3, j in 1:3]
    # w = [e.values[i] for e in eig[:,:,:], i in 1:3]
    @time l = scale_w_adaptive(w, mask_orig, type)
    @time wmod= @. 1/(1+l*w^4)

    @time begin
    structure_tensor=zeros(type, (size(wmod)[1:3]...,6))
    @. structure_tensor[:,:,:,1]=(((v[:,:,:,1,1]*v[:,:,:,1,1])*wmod[:,:,:,1])+(v[:,:,:,1,2]^2)*wmod[:,:,:,2]+(v[:,:,:,1,3]^2)*wmod[:,:,:,3])   
    @. structure_tensor[:,:,:,2]=(((v[:,:,:,1,1]*v[:,:,:,2,1])*wmod[:,:,:,1])+(v[:,:,:,1,2]*v[:,:,:,2,2])*wmod[:,:,:,2]+(v[:,:,:,1,3]*v[:,:,:,2,3])*wmod[:,:,:,3])
    @. structure_tensor[:,:,:,3]=(((v[:,:,:,1,1]*v[:,:,:,3,1])*wmod[:,:,:,1])+(v[:,:,:,1,2]*v[:,:,:,3,2])*wmod[:,:,:,2]+(v[:,:,:,1,3]*v[:,:,:,3,3])*wmod[:,:,:,3])
    @. structure_tensor[:,:,:,4]=(((v[:,:,:,2,1]^2)*wmod[:,:,:,1])+(v[:,:,:,2,2]^2)*wmod[:,:,:,2]+(v[:,:,:,2,3]^2)*wmod[:,:,:,3])
    @. structure_tensor[:,:,:,5]=(((v[:,:,:,2,1]*v[:,:,:,3,1])*wmod[:,:,:,1])+(v[:,:,:,2,2]*v[:,:,:,3,2])*wmod[:,:,:,2]+(v[:,:,:,2,3]*v[:,:,:,3,3])*wmod[:,:,:,3])
    @. structure_tensor[:,:,:,6]=(((v[:,:,:,3,1]^2)*wmod[:,:,:,1])+(v[:,:,:,3,2]^2)*wmod[:,:,:,2]+(v[:,:,:,3,3]^2)*wmod[:,:,:,3]) 
    end
    return (structure_tensor,wmod,v,l)
end

function scale_w_adaptive(w, mask, type)
    max_tries = 10e3
    grad=zeros(type, (size(mask)...,3))
    wmod = similar(w)
    totpi=sum(mask)
    l = 1e-7
    for _ = 1:max_tries
        
        @. wmod = 1 / (1 + l * (w^4))
        @. grad[:,:,:,1]=(wmod[:,:,:,1]<0.3)*mask
        @. grad[:,:,:,2]=(wmod[:,:,:,2]<0.3)*mask
        @. grad[:,:,:,3]=(wmod[:,:,:,3]<0.3)*mask
        
        wgrad=sum(grad)
        @show wgrad / totpi
        if wgrad>=0.1*totpi
            return l
        end
        l *= 1.1
    end
    return l
end

function eigen_decomposition(J)
    eigvec = zeros(size(J)...);
    eigval = zeros(size(J)[1:4]...);
    for i in axes(J,1), j in axes(J,2)
        Threads.@threads for k in axes(J,3)
            e = eigen(view(J,i,j,k,:,:))
            eigvec[i,j,k,:,:] .= e.vectors
            eigval[i,j,k,:] .= e.values
        end
    end
    return (eigvec, eigval)
end
