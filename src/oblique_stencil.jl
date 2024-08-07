using ImageFiltering, FFTW, LinearAlgebra

function dipole(x, y, z, r_treshold, direction=(0, 0, 1))
    r = sqrt(x * x + y * y + z * z)
    if r < r_treshold
        return NaN
    end
    xz = (direction[1] * x + direction[2] * y + direction[3] * z) / r
    kappa = 1 / 4pi * (3 * xz * xz - 1) / (r * r * r)
    return kappa
end

function stencil(; st=27, res=(1.0, 1.0, 1.0), singularity_cutout=4, direction=(0, 0, 1), gridsize=(64, 64, 64))
    if any(res .== 0) || all(direction .== 0) || any(gridsize .== 0)
        error("Please check the input parameters to the stencil computation. res $res, gridsize $gridsize and direction $direction must not be: any(res .== 0) || all(direction .== 0) || any(gridsize .== 0)")
    end

    middle = floor.(Int, gridsize ./ 2) .+ 1

    coord = [((1:gridsize[i]) .- middle[i]) for i in 1:3]

    d = [dipole(x, y, z, singularity_cutout, direction) for x in coord[1], y in coord[2], z in coord[3]]
    d_mask = isfinite.(d)

    # stencil mask
    if st == 19
        mask = centered(trues((3, 3, 3)))
        mask[[-1, 1], [-1, 1], [-1, 1]] .= false
    elseif st == 27
        mask = centered(trues((3, 3, 3)))
    else # 7-point stencil
        mask = centered(falses((3, 3, 3)))
        mask[0, 0, :] .= mask[0, :, 0] .= mask[:, 0, 0] .= true
    end
    mask[0, 0, 0] = false

    midInd = CartesianIndex(middle)

    # https://www.fftw.org/fftw3_doc/1d-Real_002dodd-DFTs-_0028DSTs_0029.html
    dst(x) = FFTW.r2r(x, FFTW.RODFT00)
    idst(x) = FFTW.r2r(x, FFTW.RODFT00) ./ prod(2 .* (size(x) .+ 1))

    coord2 = [2 * sinpi.(((0:N-1) .+ 1) / (2 * (N + 1))) for N in gridsize]
    coord2_grid = (coord2[1] .^ 2) .+ reshape(coord2[2] .^ 2, 1, :) .+ reshape(coord2[3] .^ 2, 1, 1, :)
    vdeltas = []
    for I in CartesianIndices(mask)
        if !mask[I]
            continue
        end
        delta = zeros(gridsize)
        delta[midInd+I] = 1
        delta[midInd-I] = 1
        delta[midInd] = -2
        Fdelta = dst(delta)
        Fdelta ./= -coord2_grid
        vdelta = idst(Fdelta)
        push!(vdeltas, vdelta)
    end

    A = vcat((transpose(v[d_mask]) for v in vdeltas)...)
    F = svd(A)

    singular_value_threshold = 1e-10
    s_mask = F.S .>= singular_value_threshold
    s_inv = zeros(size(F.S))
    s_inv[s_mask] = 1 ./ F.S[s_mask]

    y = F.Vt * d[d_mask]
    x = F.U * (y .* s_inv)

    stencil = zeros(Float32, 3, 3, 3)

    ind = 1
    for i in eachindex(stencil)
        if mask[i]
            stencil[i] = 2x[ind]
            ind += 1
        end
    end
    weights = [(i^2 / res[1]^2 + j^2 / res[2]^2 + k^2 / res[3]^2) / (i^2 + j^2 + k^2) for i in -1:1, j in -1:1, k in -1:1]
    stencil .*= weights

    stencil[2, 2, 2] = 0
    stencil[2, 2, 2] = -sum(stencil)

    return stencil
end
