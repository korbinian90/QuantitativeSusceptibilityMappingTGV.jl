@inline function stencil(((i, j, k), (x, y, z)), A, mask)
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
    return A0, A1m, A1p, A2m, A2p, A3m, A3p
end

@inline function laplace_local(I, A, resinv2, mask=nothing)
    A0, A1m, A1p, A2m, A2p, A3m, A3p = stencil(I, A, mask)

    (-2A0 + A1m + A1p) * resinv2[1] +
    (-2A0 + A2m + A2p) * resinv2[2] +
    (-2A0 + A3m + A3p) * resinv2[3]
end

@inline function wave_local(I, A, wresinv2, mask=nothing)
    A0, A1m, A1p, A2m, A2p, A3m, A3p = stencil(I, A, mask)

    (-2A0 + A1m + A1p) * wresinv2[1] +
    (-2A0 + A2m + A2p) * wresinv2[2] +
    (-2A0 + A3m + A3p) * wresinv2[3]
end

@inline function div_local(((i, j, k), (x, y, z)), A::AbstractArray{T}, (r1, r2, r3), mask, (i1, i2, i3)=(1, 2, 3)) where {T}
    m0 = mask[i, j, k]

    div = (i < x) ? m0 * A[i1, i, j, k] * r1 : zero(T)
    div = (i > 1) ? div - mask[i-1, j, k] * A[i1, i-1, j, k] * r1 : div

    div = (j < y) ? div + m0 * A[i2, i, j, k] * r2 : div
    div = (j > 1) ? div - mask[i, j-1, k] * A[i2, i, j-1, k] * r2 : div

    div = (k < z) ? div + m0 * A[i3, i, j, k] * r3 : div
    div = (k > 1) ? div - mask[i, j, k-1] * A[i3, i, j, k-1] * r3 : div
    return div
end

# Update eta <- eta + sigma*mask*(-laplace(phi) + wave(chi) - laplace_phi0). 
function tgv_update_eta!(eta, phi, chi, laplace_phi0, mask, sigma, resinv2, wresinv2, device, nblocks)
    update_eta_kernel!(device, nblocks)(eta, phi, chi, laplace_phi0, mask, sigma, resinv2, wresinv2; ndrange=size(eta))
end

@kernel function update_eta_kernel!(eta, phi, chi, laplace_phi0, mask, sigma, resinv2, wresinv2)
    i, j, k = I = @index(Global, NTuple)
    R = @ndrange
    laplace = laplace_local((I, R), phi, resinv2)
    wave = wave_local((I, R), chi, wresinv2)

    eta[i, j, k] += sigma * mask[i, j, k] * (-laplace + wave - laplace_phi0[i, j, k])
end

# Update p <- P_{||.||_\infty <= alpha}(p + sigma*(mask0*grad(phi_f) - mask*w). 
function tgv_update_p!(p, chi, w, mask, mask0, sigma, alphainv, resinv, device, nblocks)
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
function tgv_update_q!(q, u, weight, sigma, alphainv, resinv, resinv_d2, device, nblocks)
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
function tgv_update_phi!(phi_dest, phi, eta, mask, mask0, tau, taup1inv, resinv2, device, nblocks)
    update_phi_kernel!(device, nblocks)(phi_dest, phi, eta, mask, mask0, tau, taup1inv, resinv2; ndrange=size(phi_dest))
end

@kernel function update_phi_kernel!(phi_dest, phi, eta, mask, mask0, tau, taup1inv, resinv2)
    i, j, k = I = @index(Global, NTuple)
    R = @ndrange
    laplace = laplace_local((I, R), eta, resinv2, mask0)
    fac = mask[i, j, k] ? taup1inv : 1
    phi_dest[i, j, k] = (phi[i, j, k] + tau * laplace) * fac
end

# Update chi_dest <- chi + tau*(div(p) - wave(mask*v)). 
function tgv_update_chi!(chi_dest, chi, v, p, mask0, tau, resinv, wresinv2, device, nblocks)
    update_chi_kernel!(device, nblocks)(chi_dest, chi, v, p, mask0, tau, resinv, wresinv2; ndrange=size(chi_dest))
end

@kernel function update_chi_kernel!(chi_dest, chi, v, p, mask0, tau, resinv, wresinv2)
    i, j, k = I = @index(Global, NTuple)
    R = @ndrange
    div = div_local((I, R), p, resinv, mask0)
    wave = wave_local((I, R), v, wresinv2, mask0)

    chi_dest[i, j, k] = chi[i, j, k] + tau * (div - wave)
end

# Update w_dest <- w + tau*(mask*p + div(mask0*q)). 
function tgv_update_w!(w_dest, w, p, q, mask, mask0, tau, resinv, device, nblocks)
    update_w_kernel!(device, nblocks)(w_dest, w, p, q, mask, mask0, tau, resinv; ndrange=size(mask0))
end

@kernel function update_w_kernel!(w_dest::AbstractArray{T}, w, p, q, mask, mask0, tau, (r1, r2, r3)) where {T}
    I = @index(Global, NTuple)
    R = @ndrange
    w_dest[1, I...] = w[1, I...]
    w_dest[2, I...] = w[2, I...]
    w_dest[3, I...] = w[3, I...]
    if mask[I...]
        q123 = div_local((I, R), q, (r1, r1, r1), mask0, (1, 2, 3))
        q245 = div_local((I, R), q, (r2, r2, r2), mask0, (2, 4, 5))
        q356 = div_local((I, R), q, (r3, r3, r3), mask0, (3, 5, 6))
        w_dest[1, I...] += tau * (p[1, I...] + q123)
        w_dest[2, I...] += tau * (p[2, I...] + q245)
        w_dest[3, I...] += tau * (p[3, I...] + q356)
    end
end

function extragradient_update!(u_, u)
    u_ .= 2 .* u .- u_
end
