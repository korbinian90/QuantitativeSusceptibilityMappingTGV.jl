# Update eta <- eta + sigma*mask*(-laplace(phi) + wave(chi) - laplace_phi0). 
function tgv_update_eta!(eta, phi, chi, laplace_phi0, mask, sigma, res, omega; cu=cu, device=CUDADevice())
    resinv = res .^ -2
    wresinv = [1 / 3, 1 / 3, -2 / 3] .* resinv

    # update_eta_kernel!(device, 64)(eta, phi, chi, laplace_phi0, mask, sigma, cu(resinv), cu(wresinv); ndrange=size(eta))
    @parallel (2:size(phi,1)-1, 2:size(phi,2)-1,2:size(phi,3)-1) update_eta_parallel!(eta, phi, chi, laplace_phi0, mask, sigma, resinv[1], resinv[2], resinv[3], wresinv[1], wresinv[2], wresinv[3])
end

# macro laplace(A::Symbol)  esc(:(1.0/(1.0/$A[$ix  ,$iy  ,$iz  ] + 1.0/$A[$ix+1,$iy  ,$iz  ] +
#                                      1.0/$A[$ix+1,$iy+1,$iz  ] + 1.0/$A[$ix+1,$iy+1,$iz+1] +
#                                      1.0/$A[$ix  ,$iy+1,$iz+1] + 1.0/$A[$ix  ,$iy  ,$iz+1] +
#                                      1.0/$A[$ix+1,$iy  ,$iz+1] + 1.0/$A[$ix  ,$iy+1,$iz  ] )*8.0)) end

@parallel_indices (i,j,k) function update_eta_stencil!(eta, phi, chi, laplace_phi0, mask, sigma, resinv1, resinv2, resinv3, wresinv1, wresinv2, wresinv3)
    # x, y, z = size(phi)
    # if (i > 1) && (i < x)&&(j > 1)&&(j < y)&&(k > 1)&&(k < z)
    # compute -laplace(phi)
    A0 = phi[i, j, k]
    A1m = phi[i-1, j, k]
    A1p = phi[i+1, j, k]
    A2m = phi[i, j-1, k]
    A2p = phi[i, j+1, k]
    A3m = phi[i, j, k-1]
    A3p = phi[i, j, k+1]

    laplace = (2A0 - A1m - A1p) * resinv1 +
              (2A0 - A2m - A2p) * resinv2 +
              (2A0 - A3m - A3p) * resinv3

    # compute wave(chi)
    chi0 = chi[i, j, k]
    chi1m = chi[i-1, j, k]
    chi1p = chi[i+1, j, k]
    chi2m = chi[i, j-1, k]
    chi2p = chi[i, j+1, k]
    chi3m = chi[i, j, k-1]
    chi3p = chi[i, j, k+1]

    wave = (-2chi0 + chi1m + chi1p) * wresinv1 +
           (-2chi0 + chi2m + chi2p) * wresinv2 +
           (-2chi0 + chi3m + chi3p) * wresinv3

    eta[i, j, k] += sigma * mask[i, j, k] * (laplace + wave - laplace_phi0[i, j, k])
    # end
    return
end

@parallel function update_eta_parallel!(eta, phi, chi, laplace_phi0, mask, sigma, resinv1, resinv2, resinv3, wresinv1, wresinv2, wresinv3)
    # laplace = @d2_xi(phi) * resinv[1] + @d2_yi(phi) * resinv[2] + @d2_zi(phi) * resinv[3]

    # wave = -@d2_xi(chi) * wresinv[1] - @d2_yi(chi) * wresinv[2] - @d2_zi(chi) * wresinv[3]

    @inn(eta) = @inn(eta) + sigma * @inn(mask) * (
        @d2_xi(phi) * resinv1 + @d2_yi(phi) * resinv2 + @d2_zi(phi) * resinv3 - # laplacian
        @d2_xi(chi) * wresinv1 - @d2_yi(chi) * wresinv2 - @d2_zi(chi) * wresinv3 - # wave
        @inn(laplace_phi0))
    return
end


# Update p <- P_{||.||_\infty <= alpha}(p + sigma*(mask0*grad(phi_f) - mask*w). 


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

# Update p <- P_{||.||_\infty <= alpha}(p + sigma*(mask0*grad(phi_f) - mask*w). 
function tgv_update_p!(p, chi, w, tensor, mask, mask0, sigma, alpha, res; cu=cu, device=CUDADevice(), nblocks=64)
    alphainv = 1 / alpha
    resinv = 1 ./ res

    # update_p_kernel!(device, nblocks)(p, chi, w, mask, mask0, sigma, alphainv, resinv; ndrange=size(chi))
    @parallel (2:size(p,1)-1, 2:size(p,2)-1,2:size(p,3)-1) update_p_kernel!(p, chi, w, tensor, mask, mask0, sigma, alphainv, resinv[1], resinv[2], resinv[3])
end

@parallel_indices (i,j,k) function update_p_kernel!(p, chi, w, tensor, mask, mask0, sigma, alphainv, resinv1, resinv2, resinv3)
    type = eltype(p)

    # i, j, k = @index(Global, NTuple)
    x, y, z = size(p)

    chi0 = chi[i, j, k]

    dxp = (i < x) ? (chi[i+1, j, k] - chi0) * resinv1 : zero(type)
    dyp = (j < y) ? (chi[i, j+1, k] - chi0) * resinv2 : zero(type)
    dzp = (k < z) ? (chi[i, j, k+1] - chi0) * resinv3 : zero(type)

    sigmaw0 = sigma * mask0[i, j, k]
    sigmaw = sigma * mask[i, j, k]

    grad = (dxp, dyp, dzp)
    grad_minus_w = sigmaw0 .* grad .- sigmaw .* (w[i, j, k, 1], w[i, j, k, 2], w[i, j, k, 3])
    
    t123 = (tensor[i,j,k,1], tensor[i,j,k,2], tensor[i,j,k,3])
    t245 = (tensor[i,j,k,2], tensor[i,j,k,4], tensor[i,j,k,5])
    t356 = (tensor[i,j,k,3], tensor[i,j,k,5], tensor[i,j,k,6])
    px = p[i, j, k, 1] + sum(t123 .* grad_minus_w)
    py = p[i, j, k, 2] + sum(t245 .* grad_minus_w)
    pz = p[i, j, k, 3] + sum(t356 .* grad_minus_w)

    pabs = sqrt(px * px + py * py * pz * pz) * alphainv
    pabs = (pabs > 1) ? 1 / pabs : one(type)

    p[i, j, k, 1] = px * pabs
    p[i, j, k, 2] = py * pabs
    p[i, j, k, 3] = pz * pabs
    return
end

# Update q <- P_{||.||_\infty <= alpha}(q + sigma*weight*symgrad(u)). 
function tgv_update_q!(q, u, weight, sigma, alpha, res; cu=cu, device=CUDADevice(), nblocks=64)
    alphainv = 1 / alpha
    resinv = cu(1 ./ res)
    resinv2 = cu(0.5 ./ res)

    update_q_kernel!(device, nblocks)(q, u, weight, sigma, alphainv, resinv, resinv2; ndrange=size(weight))
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

# Update w_dest <- w + tau*(mask*p + div(mask0*q)). 
function tgv_update_w!(w_dest, w, p, q, mask, mask0, tau, res, resinv_dim4, qx, qy, qz; cu=cu, device=CUDADevice(), nblocks=64)
    resinv = cu(1 ./ res)

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

function extragradient_update(u_, u)
    u_ .= 2 .* u .- u_
end
