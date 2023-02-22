function pad(arr; val=0)
    new_inds = Tuple(0:size(arr, i)+1 for i in 1:ndims(arr))
    return PaddedView(val, arr, new_inds)
end

@kernel function laplacian!(out, A, resinv)
    i, j, k = @index(Global, NTuple)
    nx, ny, nz = size(A)
    # compute -laplace(phi)
    A0 = A[i, j, k]
    A1m = (i > 1) ? A[i-1, j, k] : A0
    A1p = (i < nx) ? A[i+1, j, k] : A0
    A2m = (j > 1) ? A[i, j-1, k] : A0
    A2p = (j < ny) ? A[i, j+1, k] : A0
    A3m = (k > 1) ? A[i, j, k-1] : A0
    A3p = (k < nz) ? A[i, j, k+1] : A0

    out[i, j, k] = (2A0 - A1m - A1p) * resinv[1] +
                   (2A0 - A2m - A2p) * resinv[2] +
                   (2A0 - A3m - A3p) * resinv[3]
end

function laplacian(A, resinv, I)
    i, j, k = I
    nx, ny, nz = size(A)

    A0 = A[i, j, k]
    A1m = (i > 1) ? A[i-1, j, k] : A0
    A1p = (i < nx) ? A[i+1, j, k] : A0
    A2m = (j > 1) ? A[i, j-1, k] : A0
    A2p = (j < ny) ? A[i, j+1, k] : A0
    A3m = (k > 1) ? A[i, j, k-1] : A0
    A3p = (k < nz) ? A[i, j, k+1] : A0

    laplace = (2A0 - A1m - A1p) * resinv[1] +
              (2A0 - A2m - A2p) * resinv[2] +
              (2A0 - A3m - A3p) * resinv[3]
    return laplace
end

function wave(A, wresinv, I)
    i, j, k = I

    chi0 = A[i, j, k]
    chi1m = (i > 1) ? A[i-1, j, k] : chi0
    chi1p = (i < x) ? A[i+1, j, k] : chi0
    chi2m = (j > 1) ? A[i, j-1, k] : chi0
    chi2p = (j < y) ? A[i, j+1, k] : chi0
    chi3m = (k > 1) ? A[i, j, k-1] : chi0
    chi3p = (k < z) ? A[i, j, k+1] : chi0

    wave = (-2chi0 + chi1m + chi1p) * wresinv[1] +
           (-2chi0 + chi2m + chi2p) * wresinv[2] +
           (-2chi0 + chi3m + chi3p) * wresinv[3]
    return wave
end

@kernel function update_eta_kernel_functions!(eta, phi, chi, laplace_phi0, mask, sigma, resinv, wresinv)
    i, j, k = @index(Global, NTuple)

    laplace = laplacian(phi, resinv, (i, j, k))
    wave = wave(chi, wresinv, (i, j, k))

    eta[i, j, k] += sigma * mask[i, j, k] * (laplace + wave - laplace_phi0[i, j, k])
end

@kernel function update_eta_kernel!(eta, phi, chi, laplace_phi0, mask, sigma, resinv, wresinv)
    i, j, k = @index(Global, NTuple)
    x, y, z = size(phi)

    # compute -laplace(phi)
    A0 = phi[i, j, k]
    A1m = (i > 1) ? phi[i-1, j, k] : A0
    A1p = (i < x) ? phi[i+1, j, k] : A0
    A2m = (j > 1) ? phi[i, j-1, k] : A0
    A2p = (j < y) ? phi[i, j+1, k] : A0
    A3m = (k > 1) ? phi[i, j, k-1] : A0
    A3p = (k < z) ? phi[i, j, k+1] : A0

    laplace = (2A0 - A1m - A1p) * resinv[1] +
              (2A0 - A2m - A2p) * resinv[2] +
              (2A0 - A3m - A3p) * resinv[3]

    # compute wave(chi)
    chi0 = chi[i, j, k]
    chi1m = (i > 1) ? chi[i-1, j, k] : chi0
    chi1p = (i < x) ? chi[i+1, j, k] : chi0
    chi2m = (j > 1) ? chi[i, j-1, k] : chi0
    chi2p = (j < y) ? chi[i, j+1, k] : chi0
    chi3m = (k > 1) ? chi[i, j, k-1] : chi0
    chi3p = (k < z) ? chi[i, j, k+1] : chi0

    wave = (-2chi0 + chi1m + chi1p) * wresinv[1] +
           (-2chi0 + chi2m + chi2p) * wresinv[2] +
           (-2chi0 + chi3m + chi3p) * wresinv[3]

    eta[i, j, k] += sigma * mask[i, j, k] * (laplace + wave - laplace_phi0[i, j, k])
end

# Update eta <- eta + sigma*mask*(-laplace(phi) + wave(chi) - laplace_phi0). 
function tgv_update_eta!(eta, phi, chi, laplace_phi0, mask, sigma, res, omega; cu=cu, device=CUDADevice())
    resinv = res .^ -2
    wresinv = [1 / 3, 1 / 3, -2 / 3] .* resinv

    # update_eta_kernel_functions!(device, 64)(eta, phi, chi, laplace_phi0, mask, sigma, resinv, wresinv; ndrange=size(eta))
    update_eta_kernel!(device, 64)(eta, phi, chi, laplace_phi0, mask, sigma, cu(resinv), cu(wresinv); ndrange=size(eta))
    # update_eta_tullio!(eta, phi, chi, laplace_phi0, mask, sigma, cu(resinv), cu(wresinv))
end

function update_eta_tullio!(eta, phi, chi, laplace_phi0, mask, sigma, resinv, wresinv)
    @tullio eta[i, j, k] += begin
        laplace = phi[i, j, k] * resinv_0 - (phi[i-1, j, k] + phi[i+1, j, k]) * resinv[1] - (phi[i, j-1, k] + phi[i, j+1, k]) * resinv[2] - (phi[i, j, k-1] + phi[i, j, k+1]) * resinv[3]
        wave = chi[i, j, k] * wresinv_0 + (chi[i-1, j, k] + chi[i+1, j, k]) * wresinv[1] + (chi[i, j-1, k] + chi[i, j+1, k]) * wresinv[2] + (chi[i, j, k-1] + chi[i, j, k+1]) * wresinv[3]
        sigma * mask[i, j, k] * (laplace + wave - laplace_phi0[i, j, k])
    end
end

# Update p <- P_{||.||_\infty <= alpha}(p + sigma*(mask0*grad(phi_f) - mask*w). 
function tgv_update_p!(p, chi, w, mask, mask0, sigma, alpha, res; cu=cu, device=CUDADevice(), nblocks=64)
    alphainv = 1 / alpha
    resinv = cu(1 ./ res)

    update_p_kernel!(device, nblocks)(p, chi, w, mask, mask0, sigma, alphainv, resinv; ndrange=size(chi))

    # update_p_tullio!(p, chi, w, mask, mask0, sigma, alphainv, resinv)
end

@kernel function update_p_kernel!(p, chi, w, mask, mask0, sigma, alphainv, resinv)
    type = eltype(p)

    i, j, k = @index(Global, NTuple)
    x, y, z = size(p)

    chi0 = chi[i, j, k]

    dxp = (i < x) ? (chi[i+1, j, k] - chi0) * resinv[1] : zero(type)
    dyp = (j < y) ? (chi[i, j+1, k] - chi0) * resinv[2] : zero(type)
    dzp = (k < z) ? (chi[i, j, k+1] - chi0) * resinv[3] : zero(type)

    sigmaw0 = sigma * mask0[i, j, k]
    sigmaw = sigma * mask[i, j, k]

    px = p[i, j, k, 1] + sigmaw0 * dxp - sigmaw * w[i, j, k, 1]
    py = p[i, j, k, 2] + sigmaw0 * dyp - sigmaw * w[i, j, k, 2]
    pz = p[i, j, k, 3] + sigmaw0 * dzp - sigmaw * w[i, j, k, 3]
    pabs = sqrt(px * px + py * py * pz * pz) * alphainv
    pabs = (pabs > 1) ? 1 / pabs : one(type)

    p[i, j, k, 1] = px * pabs
    p[i, j, k, 2] = py * pabs
    p[i, j, k, 3] = pz * pabs
end

function update_p_tullio!(p, chi, w, mask, mask0, sigma, alphainv, resinv)
    @tullio p[i, j, k, 1] = begin
        chi0 = chi[i, j, k]

        dp = (chi[i+1, j, k] - chi0) * resinv[1]

        sigmaw0 = sigma * mask0[i, j, k]
        sigmaw = sigma * mask[i, j, k]

        p[i, j, k, 1] + sigmaw0 * dp - sigmaw * w[i, j, k, 1]
    end
    @tullio p[i, j, k, 2] = begin
        chi0 = chi[i, j, k]

        dp = (chi[i, j+1, k] - chi0) * resinv[2]

        sigmaw0 = sigma * mask0[i, j, k]
        sigmaw = sigma * mask[i, j, k]

        p[i, j, k, 2] + sigmaw0 * dp - sigmaw * w[i, j, k, 2]
    end
    @tullio p[i, j, k, 3] = begin
        chi0 = chi[i, j, k]

        dp = (chi[i, j, k+1] - chi0) * resinv[3]

        sigmaw0 = sigma * mask0[i, j, k]
        sigmaw = sigma * mask[i, j, k]

        p[i, j, k, 3] + sigmaw0 * dp - sigmaw * w[i, j, k, 3]
    end
end

# Update q <- P_{||.||_\infty <= alpha}(q + sigma*weight*symgrad(u)). 
function tgv_update_q!(q, u, weight, sigma, alpha, res; cu=cu, device=CUDADevice(), nblocks=64)
    alphainv = 1 / alpha
    resinv = cu(1 ./ res)
    resinv2 = cu(0.5 ./ res)

    update_q_kernel!(device, nblocks)(q, u, weight, sigma, alphainv, resinv, resinv2; ndrange=size(weight))
    # update_q_tullio!(q, u, weight, sigma, alphainv, resinv, resinv2)
end

function update_q_tullio!(q, u, weight, sigma, alphainv, resinv, resinv2)
    @tullio q[i, j, k, 1] += sigma * weight[i, j, k] * resinv[1] * (u[i+1, j, k, 1] - u[i, j, k, 1])
    @tullio q[i, j, k, 4] += sigma * weight[i, j, k] * resinv[2] * (u[i, j+1, k, 2] - u[i, j, k, 2])
    @tullio q[i, j, k, 6] += sigma * weight[i, j, k] * resinv[3] * (u[i, j, k+1, 3] - u[i, j, k, 3])
    @tullio q[i, j, k, 2] += sigma * weight[i, j, k] * (resinv2[1] * (u[i+1, j, k, 2] - u[i, j, k, 2]) + resinv2[2] * (u[i, j+1, k, 1] - u[i, j, k, 1]))
    @tullio q[i, j, k, 3] += sigma * weight[i, j, k] * (resinv2[1] * (u[i+1, j, k, 3] - u[i, j, k, 3]) + resinv2[3] * (u[i, j, k+1, 1] - u[i, j, k, 1]))
    @tullio q[i, j, k, 5] += sigma * weight[i, j, k] * (resinv2[2] * (u[i, j+1, k, 3] - u[i, j, k, 3]) + resinv2[3] * (u[i, j, k+1, 2] - u[i, j, k, 2]))

    # fq(qe::AbstractArray{T}) where {T} = sqrt(sum(qe[[1, 4, 6]] .* qe[[1, 4, 6]] .+ 2 .* qe[[2, 3, 5]] .* qe[[2, 3, 5]])) * alphainv |> x -> ifelse(x > 1, 1 / x, one(T))

    # # qabs = sqrt.(sum(q[:, :, :, [1, 4, 6]] .* q[:, :, :, [1, 4, 6]] .+ 2 .* q[:, :, :, [2, 3, 5]] .* q[:, :, :, [2, 3, 5]]; dims=4)) .* alphainv

    # # threshold(qab) =
    # #     if qab > 1
    # #         one(type) / qab
    # #     else
    # #         one(type)
    # #     end
    # @tullio qabs[i, j, k] := fq(q[i, j, k, :])
    # q .*= qabs
    # qabs .= [fq(qabs[i,j,k,:]) for ] .|> ifelse.(qabs .> 1, 1 ./ qabs, 1)
end

@kernel function update_q_kernel!(q, u, weight, sigma, alphainv, resinv, resinv2)
    i, j, k = @index(Global, NTuple)
    x, y, z = size(q)

    # compute symgrad(u)
    if (i < x)
        wxx = resinv[1] * (u[i+1, j, k, 1] - u[i, j, k, 1])
        wxy = resinv2[1] * (u[i+1, j, k, 2] - u[i, j, k, 2])
        wxz = resinv2[1] * (u[i+1, j, k, 3] - u[i, j, k, 3])
    else
        wxx = 0
        wxy = 0
        wxz = 0
    end

    if (j < y)
        wxy = wxy + resinv2[2] * (u[i, j+1, k, 1] - u[i, j, k, 1])
        wyy = resinv[2] * (u[i, j+1, k, 2] - u[i, j, k, 2])
        wyz = resinv2[2] * (u[i, j+1, k, 3] - u[i, j, k, 3])
    else
        wyy = 0
        wyz = 0
    end

    if (k < z)
        wxz = wxz + resinv2[3] * (u[i, j, k+1, 1] - u[i, j, k, 1])
        wyz = wyz + resinv2[3] * (u[i, j, k+1, 2] - u[i, j, k, 2])
        wzz = resinv[3] * (u[i, j, k+1, 3] - u[i, j, k, 3])
    else
        wzz = 0
    end

    sigmaw = sigma * weight[i, j, k]

    wxx = q[i, j, k, 1] + sigmaw * wxx
    wxy = q[i, j, k, 2] + sigmaw * wxy
    wxz = q[i, j, k, 3] + sigmaw * wxz
    wyy = q[i, j, k, 4] + sigmaw * wyy
    wyz = q[i, j, k, 5] + sigmaw * wyz
    wzz = q[i, j, k, 6] + sigmaw * wzz

    qabs = sqrt(wxx * wxx + wyy * wyy + wzz * wzz +
                2 * (wxy * wxy + wxz * wxz + wyz * wyz)) * alphainv
    qabs = (qabs > 1) ? 1 / qabs : 1

    q[i, j, k, 1] = wxx * qabs
    q[i, j, k, 2] = wxy * qabs
    q[i, j, k, 3] = wxz * qabs
    q[i, j, k, 4] = wyy * qabs
    q[i, j, k, 5] = wyz * qabs
    q[i, j, k, 6] = wzz * qabs
end


# Update phi_dest <- (phi + tau*laplace(mask0*eta))/(1+mask*tau). 
function tgv_update_phi!(phi_dest, phi, eta, mask, mask0, tau, res; cu=cu, device=CUDADevice(), nblocks=64)
    taup1inv = 1 / (tau + 1)
    resinv = cu(res .^ -2)

    update_phi_kernel!(device, nblocks)(phi_dest, phi, eta, mask, mask0, tau, taup1inv, resinv; ndrange=size(phi_dest))
    # update_phi_tullio!(phi_dest, phi, eta, mask, mask0, tau1inv, resinv)

end

function update_phi_tullio!(phi_dest, phi, eta, mask, mask0, taup1inv, resinv)
    resinv_0 = -2 * sum(resinv)
    @tullio phi_dest[i, j, k] = begin
        laplace = eta[i, j, k] * resinv_0 + (eta[i-1, j, k] * mask0[i-1, j, k] + eta[i+1, j, k] * mask0[i+1, j, k]) * resinv[1] + (eta[i, j-1, k] * mask0[i, j-1, k] + eta[i, j+1, k] * mask0[i, j+1, k]) * resinv[2] + (eta[i, j, k-1] * mask0[i, j, k-1] + eta[i, j, k+1] * mask0[i, j, k+1]) * resinv[3]
        fac = mask[i, j, k] * taup1inv + !mask[i, j, k]
        (phi[i, j, k] + tau * laplace) * fac
    end
end

@kernel function update_phi_kernel!(phi_dest, phi, eta, mask, mask0, tau, taup1inv, resinv)
    i, j, k = @index(Global, NTuple)
    x, y, z = size(phi_dest)

    # compute laplace(mask*eta)
    v0 = mask0[i, j, k] * eta[i, j, k]
    v1m = (i > 1) ? mask0[i-1, j, k] * eta[i-1, j, k] : v0
    v1p = (i < x) ? mask0[i+1, j, k] * eta[i+1, j, k] : v0
    v2m = (j > 1) ? mask0[i, j-1, k] * eta[i, j-1, k] : v0
    v2p = (j < y) ? mask0[i, j+1, k] * eta[i, j+1, k] : v0
    v3m = (k > 1) ? mask0[i, j, k-1] * eta[i, j, k-1] : v0
    v3p = (k < z) ? mask0[i, j, k+1] * eta[i, j, k+1] : v0

    laplace = (-2 * v0 + v1m + v1p) * resinv[1] +
              (-2 * v0 + v2m + v2p) * resinv[2] +
              (-2 * v0 + v3m + v3p) * resinv[3]

    fac = mask[i, j, k] ? taup1inv : 1
    phi_dest[i, j, k] = (phi[i, j, k] + tau * laplace) * fac
end

# Update chi_dest <- chi + tau*(div(p) - wave(mask*v)). 
function tgv_update_chi!(chi_dest, chi, v, p, mask0, tau, res, omega; cu=cu, device=CUDADevice(), nblocks=64)
    resinv = cu(1 ./ res)
    wresinv = cu([1 / 3, 1 / 3, -2 / 3] ./ (res .^ 2))

    update_chi_kernel!(device, nblocks)(chi_dest, chi, v, p, mask0, tau, resinv, wresinv; ndrange=size(chi_dest))
    # update_chi_tullio!(chi_dest, chi, v, p, mask0, tau, resinv, wresinv)
end

@kernel function update_chi_kernel!(chi_dest, chi, v, p, mask0, tau, resinv, wresinv)
    i, j, k = @index(Global, NTuple)
    x, y, z = size(chi_dest)

    m0 = mask0[i, j, k]

    # compute div(weight*v)
    div = (i < x) ? m0 * p[i, j, k, 1] * resinv[1] : 0
    div = (i > 1) ? div - mask0[i-1, j, k] * p[i-1, j, k, 1] * resinv[1] : div

    div = (j < y) ? div + m0 * p[i, j, k, 2] * resinv[2] : div
    div = (j > 1) ? div - mask0[i, j-1, k] * p[i, j-1, k, 2] * resinv[2] : div

    div = (k < z) ? div + m0 * p[i, j, k, 3] * resinv[3] : div
    div = (k > 1) ? div - mask0[i, j, k-1] * p[i, j, k-1, 3] * resinv[3] : div

    # compute wave(mask*v)
    v0 = m0 * v[i, j, k]
    v1m = (i > 1) ? mask0[i-1, j, k] * v[i-1, j, k] : v0
    v1p = (i < x) ? mask0[i+1, j, k] * v[i+1, j, k] : v0
    v2m = (j > 1) ? mask0[i, j-1, k] * v[i, j-1, k] : v0
    v2p = (j < y) ? mask0[i, j+1, k] * v[i, j+1, k] : v0
    v3m = (k > 1) ? mask0[i, j, k-1] * v[i, j, k-1] : v0
    v3p = (k < z) ? mask0[i, j, k+1] * v[i, j, k+1] : v0

    wave = (-2 * v0 + v1m + v1p) * wresinv[1] +
           (-2 * v0 + v2m + v2p) * wresinv[2] +
           (-2 * v0 + v3m + v3p) * wresinv[3]

    chi_dest[i, j, k] = chi[i, j, k] + tau * (div - wave)
end

function update_chi_tullio!(chi_dest, chi, v, p, mask0, tau, resinv, wresinv)
    wresinv_0 = -2 * sum(wresinv)
    @tullio chi_dest[i, j, k] = begin
        div = mask0[i, j, k] * (p[i, j, k, 1] * resinv[1] + p[i, j, k, 2] * resinv[2] + p[i, j, k, 3] * resinv[3]) - mask0[i-1, j, k] * p[i-1, j, k, 1] * resinv[1] - mask0[i, j-1, k] * p[i, j-1, k, 2] * resinv[2] - mask0[i, j, k-1] * p[i, j, k-1, 3] * resinv[3]
        wave = v[i, j, k] * wresinv_0 + (v[i-1, j, k] * mask0[i-1, j, k] + v[i+1, j, k] * mask0[i+1, j, k]) * wresinv[1] + (v[i, j-1, k] * mask0[i, j-1, k] + v[i, j+1, k] * mask0[i, j+1, k]) * wresinv[2] + (v[i, j, k-1] * mask0[i, j, k-1] + v[i, j, k+1] * mask0[i, j, k+1]) * wresinv[3]
        chi[i, j, k] + tau * (div - wave)
    end

    # @tullio div[i, j, k] := mask0[i, j, k] * p[i, j, k, n] - mask0[i-1, j, k] * p[i-1, j, k, 1] - mask0[i, j-1, k] * p[i, j-1, k, 2] - mask0[i, j, k-1] * p[i, j, k-1, 3]

    # @tullio wave[i, j, k] := v[i, j, k] * wresinv_0 + (v[i-1, j, k] + v[i+1, j, k]) * wresinv[1] + (v[i, j-1, k] + v[i, j+1, k]) * wresinv[2] + (v[i, j, k-1] + v[i, j, k+1]) * wresinv[3]

    # @tullio chi_dest[i, j, k] = chi[i, j, k] + tau * (div[i, j, k] - wave[i, j, k])
end


# Update w_dest <- w + tau*(mask*p + div(mask0*q)). 
function tgv_update_w!(w_dest, w, p, q, mask, mask0, tau, res, resinv_dim4, qx, qy, qz; cu=cu, device=CUDADevice(), nblocks=64)

    resinv = cu(1 ./ res)

    # update_w_tullio!(w_dest, w, p, q, mask, mask0, tau)
    # update_w_dot!(w_dest, w, p, q, mask, mask0, tau, resinv_dim4, qx, qy, qz)

    update_w_kernel!(device, nblocks)(w_dest, w, p, q, mask, mask0, tau, resinv; ndrange=size(mask0))
end

@kernel function update_w_kernel!(w_dest, w, p, q, mask, mask0, tau, resinv)
    type = eltype(w_dest)

    i, j, k = @index(Global, NTuple)
    x, y, z = size(w_dest)

    w0 = mask0[i, j, k]
    w1 = (i > 1) ? mask0[i-1, j, k] : zero(type)
    w2 = (j > 1) ? mask0[i, j-1, k] : zero(type)
    w3 = (k > 1) ? mask0[i, j, k-1] : zero(type)

    q0x = (i < x) ? w0 * q[i, j, k, 1] * resinv[1] : zero(type)
    q0x = (i > 1) ? q0x - w1 * q[i-1, j, k, 1] * resinv[1] : q0x
    q1y = (j < y) ? w0 * q[i, j, k, 2] * resinv[1] : zero(type)
    q1y = (j > 1) ? q1y - w2 * q[i, j-1, k, 2] * resinv[1] : q1y
    q2z = (k < z) ? w0 * q[i, j, k, 3] * resinv[1] : zero(type)
    q2z = (k > 1) ? q2z - w3 * q[i, j, k-1, 3] * resinv[1] : q2z

    q1x = (i < x) ? w0 * q[i, j, k, 2] * resinv[2] : zero(type)
    q1x = (i > 1) ? q1x - w1 * q[i-1, j, k, 2] * resinv[2] : q1x
    q3y = (j < y) ? w0 * q[i, j, k, 4] * resinv[2] : zero(type)
    q3y = (j > 1) ? q3y - w2 * q[i, j-1, k, 4] * resinv[2] : q3y
    q4z = (k < z) ? w0 * q[i, j, k, 5] * resinv[2] : zero(type)
    q4z = (k > 1) ? q4z - w3 * q[i, j, k-1, 5] * resinv[2] : q4z

    q2x = (i < x) ? w0 * q[i, j, k, 3] * resinv[3] : zero(type)
    q2x = (i > 1) ? q2x - w1 * q[i-1, j, k, 3] * resinv[3] : q2x
    q4y = (j < y) ? w0 * q[i, j, k, 5] * resinv[3] : zero(type)
    q4y = (j > 1) ? q4y - w2 * q[i, j-1, k, 5] * resinv[3] : q4y
    q5z = (k < z) ? w0 * q[i, j, k, 6] * resinv[3] : zero(type)
    q5z = (k > 1) ? q5z - w3 * q[i, j, k-1, 6] * resinv[3] : q5z

    m0 = mask[i, j, k]

    w_dest[i, j, k, 1] = w[i, j, k, 1] + tau * (m0 * p[i, j, k, 1] + q0x + q1y + q2z)
    w_dest[i, j, k, 2] = w[i, j, k, 2] + tau * (m0 * p[i, j, k, 2] + q1x + q3y + q4z)
    w_dest[i, j, k, 3] = w[i, j, k, 3] + tau * (m0 * p[i, j, k, 3] + q2x + q4y + q5z)
end

function update_w_dot!(w_dest, w, p, q, mask, mask0, tau, resinv_dim4, qx, qy, qz)
    # qx[2:end, :, :, :] .= diff(mask0 .* view(q, :, :, :, [1, 2, 3]); dims=1)
    # qy[:, 2:end, :, :] .= diff(mask0 .* view(q, :, :, :, [2, 4, 5]); dims=2)
    # qz[:, :, 2:end, :] .= diff(mask0 .* view(q, :, :, :, [3, 5, 6]); dims=3)

    @async @views qx[2:end, :, :, :] .= mask0[2:end, :, :] .* q[2:end, :, :, [1, 2, 3]] .- mask0[1:end-1, :, :] .* q[1:end-1, :, :, [1, 2, 3]]
    @async @views qy[:, 2:end, :, :] .= mask0[:, 2:end, :] .* q[:, 2:end, :, [2, 4, 5]] .- mask0[:, 1:end-1, :] .* q[:, 1:end-1, :, [2, 4, 5]]
    @async @views qz[:, :, 2:end, :] .= mask0[:, :, 2:end] .* q[:, :, 2:end, [3, 5, 6]] .- mask0[:, :, 1:end-1] .* q[:, :, 1:end-1, [3, 5, 6]]
    synchronize()
    w_dest .= w .+ tau .* (mask .* p .+ (qx .+ qy .+ qz) .* resinv_dim4)
end

function update_w_tullio!(w_dest, w, p, q, mask, mask0, tau)
    @tullio w_dest[i, j, k, 1] = @inbounds(begin
        q0x = mask0[i, j, k] * q[i, j, k, 1] - mask0[i-1, j, k] * q[i-1, j, k, 1]
        q1y = mask0[i, j, k] * q[i, j, k, 2] - mask0[i, j-1, k] * q[i, j-1, k, 2]
        q2z = mask0[i, j, k] * q[i, j, k, 3] - mask0[i, j, k-1] * q[i, j, k-1, 3]
        w[i, j, k, 1] + tau * (mask[i, j, k] * p[i, j, k, 1] + (q0x + q1y + q2z) * resinv[1])
    end)
    @tullio w_dest[i, j, k, 2] = begin
        q1x = mask0[i, j, k] * q[i, j, k, 1] - mask0[i-1, j, k] * q[i-1, j, k, 1]
        q3y = mask0[i, j, k] * q[i, j, k, 3] - mask0[i, j-1, k] * q[i, j-1, k, 3]
        q4z = mask0[i, j, k] * q[i, j, k, 4] - mask0[i, j, k-1] * q[i, j, k-1, 4]
        w[i, j, k, 2] + tau * (mask[i, j, k] * p[i, j, k, 2] + (q1x + q3y + q4z) * resinv[2])
    end
    @tullio w_dest[i, j, k, 3] = begin
        q2x = mask0[i, j, k] * q[i, j, k, 2] - mask0[i-1, j, k] * q[i-1, j, k, 2]
        q4y = mask0[i, j, k] * q[i, j, k, 4] - mask0[i, j-1, k] * q[i, j-1, k, 4]
        q5z = mask0[i, j, k] * q[i, j, k, 5] - mask0[i, j, k-1] * q[i, j, k-1, 5]
        w[i, j, k, 3] + tau * (mask[i, j, k] * p[i, j, k, 3] + (q2x + q4y + q5z) * resinv[3])
    end
end

function extragradient_update(u_, u)
    u_ .= 2 .* u .- u_
end
