# Update eta <- eta + sigma*mask*(-laplace(phi) + wave(chi) - laplace_phi0). 
function tgv_update_eta!(eta, phi, chi, laplace_phi0, mask, sigma, resinv2, wresinv2, device, nblocks)
    update_eta_kernel!(device, nblocks)(eta, phi, chi, laplace_phi0, mask, sigma, resinv2, wresinv2; ndrange=size(eta))
end

@kernel function update_eta_kernel!(eta, phi, chi, laplace_phi0, mask, sigma, resinv2, wresinv2)
    i, j, k = @index(Global, NTuple)
    x, y, z = @ndrange

    # compute -laplace(phi)
    A0 = phi[i, j, k]
    A1m = (i > 1) ? phi[i-1, j, k] : A0
    A1p = (i < x) ? phi[i+1, j, k] : A0
    A2m = (j > 1) ? phi[i, j-1, k] : A0
    A2p = (j < y) ? phi[i, j+1, k] : A0
    A3m = (k > 1) ? phi[i, j, k-1] : A0
    A3p = (k < z) ? phi[i, j, k+1] : A0

    laplace = (2A0 - A1m - A1p) * resinv2[1] +
              (2A0 - A2m - A2p) * resinv2[2] +
              (2A0 - A3m - A3p) * resinv2[3]

    # compute wave(chi)
    chi0 = chi[i, j, k]
    chi1m = (i > 1) ? chi[i-1, j, k] : chi0
    chi1p = (i < x) ? chi[i+1, j, k] : chi0
    chi2m = (j > 1) ? chi[i, j-1, k] : chi0
    chi2p = (j < y) ? chi[i, j+1, k] : chi0
    chi3m = (k > 1) ? chi[i, j, k-1] : chi0
    chi3p = (k < z) ? chi[i, j, k+1] : chi0

    wave = (-2chi0 + chi1m + chi1p) * wresinv2[1] +
           (-2chi0 + chi2m + chi2p) * wresinv2[2] +
           (-2chi0 + chi3m + chi3p) * wresinv2[3]

    eta[i, j, k] += sigma * mask[i, j, k] * (laplace + wave - laplace_phi0[i, j, k])
end

# Update p <- P_{||.||_\infty <= alpha}(p + sigma*(mask0*grad(phi_f) - mask*w). 
function tgv_update_p!(p, chi, w, mask, mask0, sigma, alpha, resinv, device, nblocks)
    alphainv = 1 / alpha
    update_p_kernel!(device, nblocks)(p, chi, w, mask, mask0, sigma, alphainv, resinv; ndrange=size(chi))
end

@kernel function update_p_kernel!(p::AbstractArray{T}, chi, w, mask, mask0, sigma, alphainv, resinv) where {T}
    i, j, k = @index(Global, NTuple)
    x, y, z = @ndrange

    chi0 = chi[i, j, k]

    dxp = (i < x) ? (chi[i+1, j, k] - chi0) * resinv[1] : zero(T)
    dyp = (j < y) ? (chi[i, j+1, k] - chi0) * resinv[2] : zero(T)
    dzp = (k < z) ? (chi[i, j, k+1] - chi0) * resinv[3] : zero(T)

    sigmaw0 = sigma * mask0[i, j, k]
    sigmaw = sigma * mask[i, j, k]

    px = p[1, i, j, k] + sigmaw0 * dxp - sigmaw * w[1, i, j, k]
    py = p[2, i, j, k] + sigmaw0 * dyp - sigmaw * w[2, i, j, k]
    pz = p[3, i, j, k] + sigmaw0 * dzp - sigmaw * w[3, i, j, k]
    pabs = sqrt(px * px + py * py + pz * pz) * alphainv
    pabs = (pabs > 1) ? 1 / pabs : one(T)

    p[1, i, j, k] = px * pabs
    p[2, i, j, k] = py * pabs
    p[3, i, j, k] = pz * pabs
end

# Update q <- P_{||.||_\infty <= alpha}(q + sigma*weight*symgrad(u)). 
function tgv_update_q!(q, u, weight, sigma, alpha, resinv, resinv_d2, device, nblocks)
    alphainv = 1 / alpha
    update_q_kernel!(device, nblocks)(q, u, weight, sigma, alphainv, resinv, resinv_d2; ndrange=size(weight))
end

@kernel function update_q_kernel!(q, u, weight, sigma, alphainv, resinv, resinv_d2)
    i, j, k = @index(Global, NTuple)
    x, y, z = @ndrange

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

    wxx = q[1, i, j, k] + sigmaw * wxx
    wxy = q[2, i, j, k] + sigmaw * wxy
    wxz = q[3, i, j, k] + sigmaw * wxz
    wyy = q[4, i, j, k] + sigmaw * wyy
    wyz = q[5, i, j, k] + sigmaw * wyz
    wzz = q[6, i, j, k] + sigmaw * wzz

    qabs = sqrt(wxx * wxx + wyy * wyy + wzz * wzz +
                2 * (wxy * wxy + wxz * wxz + wyz * wyz)) * alphainv
    qabs = (qabs > 1) ? 1 / qabs : 1

    q[1, i, j, k] = wxx * qabs
    q[2, i, j, k] = wxy * qabs
    q[3, i, j, k] = wxz * qabs
    q[4, i, j, k] = wyy * qabs
    q[5, i, j, k] = wyz * qabs
    q[6, i, j, k] = wzz * qabs
end

# Update phi_dest <- (phi + tau*laplace(mask0*eta))/(1+mask*tau). 
function tgv_update_phi!(phi_dest, phi, eta, mask, mask0, tau, resinv2, device, nblocks)
    taup1inv = 1 / (tau + 1)
    update_phi_kernel!(device, nblocks)(phi_dest, phi, eta, mask, mask0, tau, taup1inv, resinv2; ndrange=size(phi_dest))
end

@kernel function update_phi_kernel!(phi_dest, phi, eta, mask, mask0, tau, taup1inv, resinv2)
    i, j, k = @index(Global, NTuple)
    x, y, z = @ndrange

    # compute laplace(mask*eta)
    v0 = mask0[i, j, k] * eta[i, j, k]
    v1m = (i > 1) ? mask0[i-1, j, k] * eta[i-1, j, k] : v0
    v1p = (i < x) ? mask0[i+1, j, k] * eta[i+1, j, k] : v0
    v2m = (j > 1) ? mask0[i, j-1, k] * eta[i, j-1, k] : v0
    v2p = (j < y) ? mask0[i, j+1, k] * eta[i, j+1, k] : v0
    v3m = (k > 1) ? mask0[i, j, k-1] * eta[i, j, k-1] : v0
    v3p = (k < z) ? mask0[i, j, k+1] * eta[i, j, k+1] : v0

    laplace = (-2 * v0 + v1m + v1p) * resinv2[1] +
              (-2 * v0 + v2m + v2p) * resinv2[2] +
              (-2 * v0 + v3m + v3p) * resinv2[3]

    fac = mask[i, j, k] ? taup1inv : 1
    phi_dest[i, j, k] = (phi[i, j, k] + tau * laplace) * fac
end

# Update chi_dest <- chi + tau*(div(p) - wave(mask*v)). 
function tgv_update_chi!(chi_dest, chi, v, p, mask0, tau, resinv, wresinv2, device, nblocks)
    update_chi_kernel!(device, nblocks)(chi_dest, chi, v, p, mask0, tau, resinv, wresinv2; ndrange=size(chi_dest))
end

@kernel function update_chi_kernel!(chi_dest, chi, v, p, mask0, tau, resinv, wresinv2)
    i, j, k = @index(Global, NTuple)
    x, y, z = @ndrange

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

    wave = (-2 * v0 + v1m + v1p) * wresinv2[1] +
           (-2 * v0 + v2m + v2p) * wresinv2[2] +
           (-2 * v0 + v3m + v3p) * wresinv2[3]

    chi_dest[i, j, k] = chi[i, j, k] + tau * (div - wave)
end

# Update w_dest <- w + tau*(mask*p + div(mask0*q)). 
function tgv_update_w!(w_dest, w, p, q, mask, mask0, tau, resinv, device, nblocks)
    update_w_kernel!(device, nblocks)(w_dest, w, p, q, mask, mask0, tau, resinv; ndrange=size(mask0))
end

@kernel function update_w_kernel!(w_dest::AbstractArray{T}, w, p, q, mask, mask0, tau, resinv) where {T}
    i, j, k = @index(Global, NTuple)
    x, y, z = @ndrange

    w0 = mask0[i, j, k]
    w1 = (i > 1) ? mask0[i-1, j, k] : zero(T)
    w2 = (j > 1) ? mask0[i, j-1, k] : zero(T)
    w3 = (k > 1) ? mask0[i, j, k-1] : zero(T)

    q0x = (i < x) ? w0 * q[1, i, j, k] * resinv[1] : zero(T)
    q0x = (i > 1) ? q0x - w1 * q[1, i-1, j, k] * resinv[1] : q0x
    q1y = (j < y) ? w0 * q[2, i, j, k] * resinv[1] : zero(T)
    q1y = (j > 1) ? q1y - w2 * q[2, i, j-1, k] * resinv[1] : q1y
    q2z = (k < z) ? w0 * q[3, i, j, k] * resinv[1] : zero(T)
    q2z = (k > 1) ? q2z - w3 * q[3, i, j, k-1] * resinv[1] : q2z

    q1x = (i < x) ? w0 * q[2, i, j, k] * resinv[2] : zero(T)
    q1x = (i > 1) ? q1x - w1 * q[2, i-1, j, k] * resinv[2] : q1x
    q3y = (j < y) ? w0 * q[4, i, j, k] * resinv[2] : zero(T)
    q3y = (j > 1) ? q3y - w2 * q[4, i, j-1, k] * resinv[2] : q3y
    q4z = (k < z) ? w0 * q[5, i, j, k] * resinv[2] : zero(T)
    q4z = (k > 1) ? q4z - w3 * q[5, i, j, k-1] * resinv[2] : q4z

    q2x = (i < x) ? w0 * q[3, i, j, k] * resinv[3] : zero(T)
    q2x = (i > 1) ? q2x - w1 * q[3, i-1, j, k] * resinv[3] : q2x
    q4y = (j < y) ? w0 * q[5, i, j, k] * resinv[3] : zero(T)
    q4y = (j > 1) ? q4y - w2 * q[5, i, j-1, k] * resinv[3] : q4y
    q5z = (k < z) ? w0 * q[6, i, j, k] * resinv[3] : zero(T)
    q5z = (k > 1) ? q5z - w3 * q[6, i, j, k-1] * resinv[3] : q5z

    m0 = mask[i, j, k]

    w_dest[1, i, j, k] = w[1, i, j, k] + tau * (m0 * p[1, i, j, k] + q0x + q1y + q2z)
    w_dest[2, i, j, k] = w[2, i, j, k] + tau * (m0 * p[2, i, j, k] + q1x + q3y + q4z)
    w_dest[3, i, j, k] = w[3, i, j, k] + tau * (m0 * p[3, i, j, k] + q2x + q4y + q5z)
end

function extragradient_update!(u_, u)
    u_ .= 2 .* u .- u_
end
