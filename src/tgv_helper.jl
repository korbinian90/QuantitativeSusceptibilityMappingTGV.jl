function Δ(A, i, j, k, resinv2, mask=nothing)
    x, y, z = size(A)

    A0 = A[i, j, k]
    if isnothing(mask)
        A1m = (i > 1) ? A[i-1, j, k] : A0
        A1p = (i < x) ? A[i+1, j, k] : A0
        A2m = (j > 1) ? A[i, j-1, k] : A0
        A2p = (j < y) ? A[i, j+1, k] : A0
        A3m = (k > 1) ? A[i, j, k-1] : A0
        A3p = (k < z) ? A[i, j, k+1] : A0
    else
        A1m = (i > 1) ? mask[i-1, j, k] * A[i-1, j, k] : A0
        A1p = (i < x) ? mask[i+1, j, k] * A[i+1, j, k] : A0
        A2m = (j > 1) ? mask[i, j-1, k] * A[i, j-1, k] : A0
        A2p = (j < y) ? mask[i, j+1, k] * A[i, j+1, k] : A0
        A3m = (k > 1) ? mask[i, j, k-1] * A[i, j, k-1] : A0
        A3p = (k < z) ? mask[i, j, k+1] * A[i, j, k+1] : A0
    end

    res = (-2A0 + A1m + A1p) * resinv2[1] +
          (-2A0 + A2m + A2p) * resinv2[2] +
          (-2A0 + A3m + A3p) * resinv2[3]
    return res
end

function □(A, i, j, k, resinv2, omega2, mask=nothing)
    Δ(A, i, j, k, resinv2, mask) / 3 - D(A, i, j, k, resinv2, omega2, mask)
end

function D(A, i, j, k, resinv2, omega2, mask=nothing)
    x, y, z = size(A)

    A0 = A[i, j, k]
    if isnothing(mask)
        A1m = (i > 1) ? A[i-1, j, k] : A0
        A1p = (i < x) ? A[i+1, j, k] : A0
        A2m = (j > 1) ? A[i, j-1, k] : A0
        A2p = (j < y) ? A[i, j+1, k] : A0
        A3m = (k > 1) ? A[i, j, k-1] : A0
        A3p = (k < z) ? A[i, j, k+1] : A0
    else
        A1m = (i > 1) ? mask[i-1, j, k] * A[i-1, j, k] : A0
        A1p = (i < x) ? mask[i+1, j, k] * A[i+1, j, k] : A0
        A2m = (j > 1) ? mask[i, j-1, k] * A[i, j-1, k] : A0
        A2p = (j < y) ? mask[i, j+1, k] * A[i, j+1, k] : A0
        A3m = (k > 1) ? mask[i, j, k-1] * A[i, j, k-1] : A0
        A3p = (k < z) ? mask[i, j, k+1] * A[i, j, k+1] : A0
    end

    res = (-2A0 + A1m + A1p) * resinv2[1] * omega2[1] +
          (-2A0 + A2m + A2p) * resinv2[2] * omega2[2] +
          (-2A0 + A3m + A3p) * resinv2[3] * omega2[3]

    return res
end

# Update eta <- eta + sigma*mask*(-laplace(phi) + wave(chi) - laplace_phi0). 
function tgv_update_eta!(eta, phi, chi, laplace_phi0, mask, sigma, res, omega; cu=cu, device=CUDADevice())
    resinv2 = res .^ -2
    omega2 = omega .^ 2

    update_eta_kernel!(device, 64)(eta, phi, chi, laplace_phi0, mask, sigma, cu(resinv2), cu(omega2); ndrange=size(eta))
end

@kernel function update_eta_kernel!(eta, phi, chi, laplace_phi0, mask, sigma, resinv2, omega2)
    i, j, k = @index(Global, NTuple)

    if mask[i, j, k]
        laplace = Δ(phi, i, j, k, resinv2)
        wave = □(chi, i, j, k, resinv2, omega2)
        eta[i, j, k] += sigma * (-laplace + wave - laplace_phi0[i, j, k])
    end
end

# Update p <- P_{||.||_\infty <= alpha}(p + sigma*(mask0*grad(phi_f) - mask*w). 
function tgv_update_p!(p, chi, w, mask, mask0, sigma, alpha, res; cu=cu, device=CUDADevice(), nblocks=64)
    alphainv = 1 / alpha
    resinv = 1 ./ res

    update_p_kernel!(device, nblocks)(p, chi, w, mask, mask0, sigma, alphainv, cu(resinv); ndrange=size(chi))
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
    pabs = sqrt(px * px + py * py + pz * pz) * alphainv
    pabs = (pabs > 1) ? 1 / pabs : one(type)

    p[i, j, k, 1] = px * pabs
    p[i, j, k, 2] = py * pabs
    p[i, j, k, 3] = pz * pabs
end

# Update q <- P_{||.||_\infty <= alpha}(q + sigma*weight*symgrad(u)). 
function tgv_update_q!(q, u, weight, sigma, alpha, res; cu=cu, device=CUDADevice(), nblocks=64)
    alphainv = 1 / alpha
    resinv = 1 ./ res
    resinv_d2 = 0.5 ./ res

    update_q_kernel!(device, nblocks)(q, u, weight, sigma, alphainv, cu(resinv), cu(resinv_d2); ndrange=size(weight))
end

@kernel function update_q_kernel!(q, u, weight, sigma, alphainv, resinv, resinv_d2)
    i, j, k = @index(Global, NTuple)
    x, y, z = size(q)

    # compute symgrad(u)
    if (i < x)
        wxx = resinv[1] * (u[i+1, j, k, 1] - u[i, j, k, 1])
        wxy = resinv_d2[1] * (u[i+1, j, k, 2] - u[i, j, k, 2])
        wxz = resinv_d2[1] * (u[i+1, j, k, 3] - u[i, j, k, 3])
    else
        wxx = 0
        wxy = 0
        wxz = 0
    end

    if (j < y)
        wxy = wxy + resinv_d2[2] * (u[i, j+1, k, 1] - u[i, j, k, 1])
        wyy = resinv[2] * (u[i, j+1, k, 2] - u[i, j, k, 2])
        wyz = resinv_d2[2] * (u[i, j+1, k, 3] - u[i, j, k, 3])
    else
        wyy = 0
        wyz = 0
    end

    if (k < z)
        wxz = wxz + resinv_d2[3] * (u[i, j, k+1, 1] - u[i, j, k, 1])
        wyz = wyz + resinv_d2[3] * (u[i, j, k+1, 2] - u[i, j, k, 2])
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
    resinv2 = res .^ -2

    update_phi_kernel!(device, nblocks)(phi_dest, phi, eta, mask, mask0, tau, taup1inv, cu(resinv2); ndrange=size(phi_dest))
end

@kernel function update_phi_kernel!(phi_dest, phi, eta, mask, mask0, tau, taup1inv, resinv2)
    i, j, k = @index(Global, NTuple)

    laplace = Δ(eta, i, j, k, resinv2, mask0)

    fac = mask[i, j, k] ? taup1inv : 1
    phi_dest[i, j, k] = (phi[i, j, k] + tau * laplace) * fac
end

# Update chi_dest <- chi + tau*(div(p) - wave(mask*v)). 
function tgv_update_chi!(chi_dest, chi, v, p, mask0, tau, res, omega; cu=cu, device=CUDADevice(), nblocks=64)
    resinv = 1 ./ res
    resinv2 = res .^ -2
    omega2 = omega .^ 2

    update_chi_kernel!(device, nblocks)(chi_dest, chi, v, p, mask0, tau, cu(resinv), cu(resinv2), cu(omega2); ndrange=size(chi_dest))
end

@kernel function update_chi_kernel!(chi_dest, chi, v, p, mask0, tau, resinv, resinv2, omega2)
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

    wave = □(v, i, j, k, resinv2, omega2, mask0)

    chi_dest[i, j, k] = chi[i, j, k] + tau * (div - wave)
end

# Update w_dest <- w + tau*(mask*p + div(mask0*q)). 
function tgv_update_w!(w_dest, w, p, q, mask, mask0, tau, res, resinv_dim4, qx, qy, qz; cu=cu, device=CUDADevice(), nblocks=64)
    resinv = 1 ./ res

    update_w_kernel!(device, nblocks)(w_dest, w, p, q, mask, mask0, tau, cu(resinv); ndrange=size(mask0))
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

function extragradient_update!(u_, u)
    u_ .= 2 .* u .- u_
end
