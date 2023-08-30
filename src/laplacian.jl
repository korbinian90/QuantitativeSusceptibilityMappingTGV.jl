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

function get_laplace_phase_conv(phase, res)
    real = cos.(phase)
    imag = sin.(phase)

    k = laplacian_kernel(res)
    del_real = imfilter(real, k)
    del_imag = imfilter(imag, k)

    laplacian_phase = del_imag .* real - del_real .* imag

    return laplacian_phase
end

function laplacian_kernel(res)
    f = 1 ./ res .^ 2
    f_sum = -2 * sum(f)
    k = [0 0 0; 0 f[3] 0; 0 0 0;;;
        0 f[1] 0; f[2] f_sum f[2]; 0 f[1] 0;;;
        0 0 0; 0 f[3] 0; 0 0 0]
    return centered(k)
end

function get_laplace_phase_romeo(phase, res)
    unwrapped = unwrap(phase)
    laplacian = imfilter(unwrapped, laplacian_kernel(res))
    return laplacian
end