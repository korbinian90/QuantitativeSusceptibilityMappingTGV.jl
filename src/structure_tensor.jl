function structure_tensor(mag, mask, mask_orig, sigma)
    if !isnothing(mag)
        return st_gauss(mag, mask, mask_orig, sigma)
    else
        return identity_tensor(size(mask))
    end
end

function st_gauss(magnitude, mask, mask_orig, sigma; type=Float32)
    N = size(magnitude)
    ND = N[1] * N[2] * N[3]
    begin
        x_sobel = -Kernel.sobel((true, true, true), 1)[1]
        y_sobel = -Kernel.sobel((true, true, true), 2)[1]
        z_sobel = -Kernel.sobel((true, true, true), 3)[1]

        dex = imfilter(magnitude, x_sobel) .* mask
        dey = imfilter(magnitude, y_sobel) .* mask
        dez = imfilter(magnitude, z_sobel) .* mask

        #orientation tensor
        J011 = dex .* dex
        J012 = dex .* dey
        J013 = dex .* dez
        J022 = dey .* dey
        J023 = dey .* dez
        J033 = dez .* dez
    end

    #structure tensor independent dimentions
    begin
        gaussian = Kernel.gaussian((sigma, sigma, sigma), (9, 9, 9))
        J11_smooth = imfilter(J011, gaussian, "symmetric")
        J12_smooth = imfilter(J012, gaussian, "symmetric")
        J13_smooth = imfilter(J013, gaussian, "symmetric")
        J22_smooth = imfilter(J022, gaussian, "symmetric")
        J23_smooth = imfilter(J023, gaussian, "symmetric")
        J33_smooth = imfilter(J033, gaussian, "symmetric")
    end
    #structure tensor in matrix 3x3
    begin
        J = zeros(type, (N[1], N[2], N[3], 3, 3))
        J[:, :, :, 1, 1] = J11_smooth
        J[:, :, :, 1, 2] = J12_smooth
        J[:, :, :, 1, 3] = J13_smooth
        J[:, :, :, 2, 1] = J12_smooth
        J[:, :, :, 2, 2] = J22_smooth
        J[:, :, :, 2, 3] = J23_smooth
        J[:, :, :, 3, 1] = J13_smooth
        J[:, :, :, 3, 2] = J23_smooth
        J[:, :, :, 3, 3] = J33_smooth
    end
    # eigen decomposition for eigenvalues modification
    v, w = eigen_decomposition(J)
    # eig = mapslices(eigen, J; dims=(4,5))
    # eig = dropdims(eig; dims=(4,5))
    # v = [[e.vectors[i,j] for e in eig] for i in 1:3, j in 1:3]
    # w = [e.values[i] for e in eig[:,:,:], i in 1:3]
    l = scale_w_adaptive(w, mask_orig, type)
    wmod = @. 1 / (1 + l * w^4)

    begin
        structure_tensor = zeros(type, (size(wmod)[1:3]..., 6))
        @. structure_tensor[:, :, :, 1] = (((v[:, :, :, 1, 1] * v[:, :, :, 1, 1]) * wmod[:, :, :, 1]) + (v[:, :, :, 1, 2]^2) * wmod[:, :, :, 2] + (v[:, :, :, 1, 3]^2) * wmod[:, :, :, 3])
        @. structure_tensor[:, :, :, 2] = (((v[:, :, :, 1, 1] * v[:, :, :, 2, 1]) * wmod[:, :, :, 1]) + (v[:, :, :, 1, 2] * v[:, :, :, 2, 2]) * wmod[:, :, :, 2] + (v[:, :, :, 1, 3] * v[:, :, :, 2, 3]) * wmod[:, :, :, 3])
        @. structure_tensor[:, :, :, 3] = (((v[:, :, :, 1, 1] * v[:, :, :, 3, 1]) * wmod[:, :, :, 1]) + (v[:, :, :, 1, 2] * v[:, :, :, 3, 2]) * wmod[:, :, :, 2] + (v[:, :, :, 1, 3] * v[:, :, :, 3, 3]) * wmod[:, :, :, 3])
        @. structure_tensor[:, :, :, 4] = (((v[:, :, :, 2, 1]^2) * wmod[:, :, :, 1]) + (v[:, :, :, 2, 2]^2) * wmod[:, :, :, 2] + (v[:, :, :, 2, 3]^2) * wmod[:, :, :, 3])
        @. structure_tensor[:, :, :, 5] = (((v[:, :, :, 2, 1] * v[:, :, :, 3, 1]) * wmod[:, :, :, 1]) + (v[:, :, :, 2, 2] * v[:, :, :, 3, 2]) * wmod[:, :, :, 2] + (v[:, :, :, 2, 3] * v[:, :, :, 3, 3]) * wmod[:, :, :, 3])
        @. structure_tensor[:, :, :, 6] = (((v[:, :, :, 3, 1]^2) * wmod[:, :, :, 1]) + (v[:, :, :, 3, 2]^2) * wmod[:, :, :, 2] + (v[:, :, :, 3, 3]^2) * wmod[:, :, :, 3])
    end
    structure_tensor = permutedims(structure_tensor, (4,1,2,3))
    return structure_tensor
end

function scale_w_adaptive(w, mask, type)
    max_tries = 10e3
    grad = zeros(type, (size(mask)..., 3))
    wmod = similar(w)
    totpi = sum(mask)
    l = 1e-7
    for _ = 1:max_tries

        @. wmod = 1 / (1 + l * (w^4))
        @. grad[:, :, :, 1] = (wmod[:, :, :, 1] < 0.3) * mask
        @. grad[:, :, :, 2] = (wmod[:, :, :, 2] < 0.3) * mask
        @. grad[:, :, :, 3] = (wmod[:, :, :, 3] < 0.3) * mask

        wgrad = sum(grad)
        @show wgrad / totpi
        if wgrad >= 0.1 * totpi
            return l
        end
        l *= 1.1
    end
    return l
end

function eigen_decomposition(J)
    eigvec = zeros(size(J)...)
    eigval = zeros(size(J)[1:4]...)
    for i in axes(J, 1), j in axes(J, 2)
        Threads.@threads for k in axes(J, 3)
            e = eigen(view(J, i, j, k, :, :))
            eigvec[i, j, k, :, :] .= e.vectors
            eigval[i, j, k, :] .= e.values
        end
    end
    return (eigvec, eigval)
end

function identity_tensor(size_3d)
    tensor = zeros(Float32, (6, size_3d...))
    for (i, val) in enumerate([1, 0, 0, 1, 0, 1])
        tensor[i, :, :, :] .= val
    end
    return tensor
end
