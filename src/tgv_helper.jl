@inline function stencil(((i, j, k), (x, y, z)), A, mask)
    @inbounds begin
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
    @inbounds begin
    m0 = mask[i, j, k]

    div = (i < x) ? m0 * A[i1, i, j, k] * r1 : zero(T)
    div = (i > 1) ? div - mask[i-1, j, k] * A[i1, i-1, j, k] * r1 : div

    div = (j < y) ? div + m0 * A[i2, i, j, k] * r2 : div
    div = (j > 1) ? div - mask[i, j-1, k] * A[i2, i, j-1, k] * r2 : div

    div = (k < z) ? div + m0 * A[i3, i, j, k] * r3 : div
    div = (k > 1) ? div - mask[i, j, k-1] * A[i3, i, j, k-1] * r3 : div
    end
    return div
end

function symgrad_local(((i, j, k), (x, y, z)), A::AbstractArray{T}, resinv, resinv_d2) where {T}
    @inbounds begin
    if (i < x)
        wxx = resinv[1] * (A[1, i+1, j, k] - A[1, i, j, k])
        wxy = resinv_d2[1] * (A[2, i+1, j, k] - A[2, i, j, k])
        wxz = resinv_d2[1] * (A[3, i+1, j, k] - A[3, i, j, k])
    else
        wxx = zero(T)
        wxy = zero(T)
        wxz = zero(T)
    end

    if (j < y)
        wxy += resinv_d2[2] * (A[1, i, j+1, k] - A[1, i, j, k])
        wyy = resinv[2] * (A[2, i, j+1, k] - A[2, i, j, k])
        wyz = resinv_d2[2] * (A[3, i, j+1, k] - A[3, i, j, k])
    else
        wyy = zero(T)
        wyz = zero(T)
    end

    if (k < z)
        wxz += resinv_d2[3] * (A[1, i, j, k+1] - A[1, i, j, k])
        wyz += resinv_d2[3] * (A[2, i, j, k+1] - A[2, i, j, k])
        wzz = resinv[3] * (A[3, i, j, k+1] - A[3, i, j, k])
    else
        wzz = zero(T)
    end
end
    return wxx, wxy, wxz, wyy, wyz, wzz
end

# Update eta <- eta + sigma*mask*(-laplace(phi) + wave(chi) - laplace_phi0). 
@kernel function update_eta_kernel!(eta, phi, chi, laplace_phi0, mask, sigma, resinv2, wresinv2)
    I = @index(Global, NTuple)
    R = @ndrange
    laplace = laplace_local((I, R), phi, resinv2)
    wave = wave_local((I, R), chi, wresinv2)

    @inbounds eta[I...] += sigma * mask[I...] * (-laplace + wave - laplace_phi0[I...])
end

# Update p <- P_{||.||_\infty <= alpha}(p + sigma*(mask0*grad(phi_f) - mask*w). 
@inline norm((x, y, z)) = sqrt(x * x + y * y + z * z)
@kernel function update_p_kernel!(p::AbstractArray{T}, chi, w, mask, mask0, sigma, alphainv, resinv) where {T}
    i, j, k = @index(Global, NTuple)
    x, y, z = @ndrange
@inbounds begin
    chi0 = chi[i, j, k]

    dxp = (i < x) ? (chi[i+1, j, k] - chi0) * resinv[1] : zero(T)
    dyp = (j < y) ? (chi[i, j+1, k] - chi0) * resinv[2] : zero(T)
    dzp = (k < z) ? (chi[i, j, k+1] - chi0) * resinv[3] : zero(T)

    sigmaw0 = sigma * mask0[i, j, k]
    sigmaw = sigma * mask[i, j, k]

    p[1, i, j, k] += sigmaw0 * dxp - sigmaw * w[1, i, j, k]
    p[2, i, j, k] += sigmaw0 * dyp - sigmaw * w[2, i, j, k]
    p[3, i, j, k] += sigmaw0 * dzp - sigmaw * w[3, i, j, k]

    pabs = norm((p[it, i, j, k] for it in 1:3)) * alphainv
    if pabs > 1
        p[1, i, j, k] /= pabs
        p[2, i, j, k] /= pabs
        p[3, i, j, k] /= pabs
    end
end
end

# Update q <- P_{||.||_\infty <= alpha}(q + sigma*weight*symgrad(u)).
@inline qnorm((xx, xy, xz, yy, yz, zz)) = sqrt(xx * xx + yy * yy + zz * zz + 2 * (xy * xy + xz * xz + yz * yz))
@kernel function update_q_kernel!(q, u, weight, sigma, alphainv, resinv, resinv_d2)
    I = @index(Global, NTuple)
    R = @ndrange
    @inbounds begin
    sigmaw = sigma * weight[I...]

    wxx, wxy, wxz, wyy, wyz, wzz = symgrad_local((I, R), u, resinv, resinv_d2)

    q[1, I...] += sigmaw * wxx
    q[2, I...] += sigmaw * wxy
    q[3, I...] += sigmaw * wxz
    q[4, I...] += sigmaw * wyy
    q[5, I...] += sigmaw * wyz
    q[6, I...] += sigmaw * wzz

    qabs = qnorm(q[i, I...] for i in 1:6) * alphainv
    if qabs > 1
        q[1, I...] /= qabs
        q[2, I...] /= qabs
        q[3, I...] /= qabs
        q[4, I...] /= qabs
        q[5, I...] /= qabs
        q[6, I...] /= qabs
    end
end
end

# Update phi_dest <- (phi + tau*laplace(mask0*eta))/(1+mask*tau). 
@kernel function update_phi_kernel!(phi_dest, phi, eta, mask, mask0, tau, taup1inv, resinv2)
    I = @index(Global, NTuple)
    R = @ndrange
    @inbounds begin
    laplace = laplace_local((I, R), eta, resinv2, mask0)
    fac = mask[I...] ? taup1inv : 1
    phi_dest[I...] = (phi[I...] + tau * laplace) * fac
    end
end

# Update chi_dest <- chi + tau*(div(p) - wave(mask*v)). 
@kernel function update_chi_kernel!(chi_dest, chi, v, p, mask0, tau, resinv, wresinv2)
    I = @index(Global, NTuple)
    R = @ndrange
    div = div_local((I, R), p, resinv, mask0)
    wave = wave_local((I, R), v, wresinv2, mask0)

    @inbounds chi_dest[I...] = chi[I...] + tau * (div - wave)
end

# Update w_dest <- w + tau*(mask*p + div(mask0*q)). 
@kernel function update_w_kernel!(w_dest::AbstractArray{T}, w, p, q, mask, mask0, tau, (r1, r2, r3)) where {T}
    I = @index(Global, NTuple)
    R = @ndrange
    @inbounds begin
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
end

function extragradient_update!(u_, u)
    u_ .= 2 .* u .- u_
end
