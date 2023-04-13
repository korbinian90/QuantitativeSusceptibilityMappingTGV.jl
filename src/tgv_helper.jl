# Update eta <- eta + sigma*mask*(-laplace(phi) + wave(chi) - laplace_phi0). 
function tgv_update_eta!(eta, phi, chi, laplace_phi0, mask, sigma, res, omega)
    resinv = res .^ -2
    wresinv = [1 / 3, 1 / 3, -2 / 3] .* resinv

    @parallel (2:size(phi,1)-1, 2:size(phi,2)-1,2:size(phi,3)-1) update_eta_stencil!(eta, phi, chi, laplace_phi0, mask, sigma, resinv[1], resinv[2], resinv[3], wresinv[1], wresinv[2], wresinv[3])
end

# macro laplace(A::Symbol)  esc(:(1.0/(1.0/$A[$ix  ,$iy  ,$iz  ] + 1.0/$A[$ix+1,$iy  ,$iz  ] +
#                                      1.0/$A[$ix+1,$iy+1,$iz  ] + 1.0/$A[$ix+1,$iy+1,$iz+1] +
#                                      1.0/$A[$ix  ,$iy+1,$iz+1] + 1.0/$A[$ix  ,$iy  ,$iz+1] +
#                                      1.0/$A[$ix+1,$iy  ,$iz+1] + 1.0/$A[$ix  ,$iy+1,$iz  ] )*8.0)) end

@parallel_indices (i,j,k) function update_eta_stencil!(eta, phi, chi, laplace_phi0, mask, sigma, resinv1, resinv2, resinv3, wresinv1, wresinv2, wresinv3)
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
    A1m = phi[i-1, j, k] 
    A1p = phi[i+1, j, k] 
    A2m = phi[i, j-1, k] 
    A2p = phi[i, j+1, k] 
    A3m = phi[i, j, k-1] 
    A3p = phi[i, j, k+1] 

    laplace = (2A0 - A1m - A1p) * resinv[1] +
              (2A0 - A2m - A2p) * resinv[2] +
              (2A0 - A3m - A3p) * resinv[3]

    # compute wave(chi)
    chi0 = chi[i, j, k]
    chi1m = chi[i-1, j, k] 
    chi1p = chi[i+1, j, k] 
    chi2m = chi[i, j-1, k] 
    chi2p = chi[i, j+1, k] 
    chi3m = chi[i, j, k-1] 
    chi3p = chi[i, j, k+1] 

    wave = (-2chi0 + chi1m + chi1p) * wresinv[1] +
           (-2chi0 + chi2m + chi2p) * wresinv[2] +
           (-2chi0 + chi3m + chi3p) * wresinv[3]

    eta[i, j, k] += sigma * mask[i, j, k] * (laplace + wave - laplace_phi0[i, j, k])
end

# Update p <- P_{||.||_\infty <= alpha}(p + sigma*(mask0*grad(phi_f) - mask*w). 
function tgv_update_p!(p, chi, w, tensor, mask, mask0, sigma, alpha, res)
    alphainv = 1 / alpha
    resinv = 1 ./ res

    @parallel (2:size(p,1)-1, 2:size(p,2)-1,2:size(p,3)-1) update_p_kernel!(p, chi, w, tensor, mask, mask0, sigma, alphainv, resinv[1], resinv[2], resinv[3])
end

@parallel_indices (i,j,k) function update_p_kernel!(p, chi, w, tensor, mask, mask0, sigma, alphainv, resinv1, resinv2, resinv3)
    chi0 = chi[i, j, k]

    dxp = (chi[i+1, j, k] - chi0) * resinv1
    dyp = (chi[i, j+1, k] - chi0) * resinv2
    dzp = (chi[i, j, k+1] - chi0) * resinv3

    sigmaw0 = sigma * mask0[i, j, k]
    sigmaw = sigma * mask[i, j, k]

    grad = (dxp, dyp, dzp)
    grad_minus_w = sigmaw0 .* grad .- sigmaw .* (w[1,i,j,k], w[2,i,j,k], w[3,i,j,k])

    t123 = (tensor[1,i,j,k], tensor[2,i,j,k], tensor[3,i,j,k])
    t245 = (tensor[2,i,j,k], tensor[4,i,j,k], tensor[5,i,j,k])
    t356 = (tensor[3,i,j,k], tensor[5,i,j,k], tensor[6,i,j,k])
    px = p[1,i,j,k] + sum(t123 .* grad_minus_w)
    py = p[2,i,j,k] + sum(t245 .* grad_minus_w)
    pz = p[3,i,j,k] + sum(t356 .* grad_minus_w)

    pabs = sqrt(px * px + py * py * pz * pz) * alphainv
    pabs = (pabs > 1) ? 1 / pabs : one(eltype(p))

    p[1,i,j,k] = px * pabs
    p[2,i,j,k] = py * pabs
    p[3,i,j,k] = pz * pabs
    return
end

# Update q <- P_{||.||_\infty <= alpha}(q + sigma*weight*symgrad(u)). 
function tgv_update_q!(q, u, weight, sigma, alpha, res)
    alphainv = 1 / alpha
    resinv = 1 ./ res

    @parallel (2:size(q,1)-1, 2:size(q,2)-1,2:size(q,3)-1) update_q_kernel!(q, u, weight, sigma, alphainv, resinv...)
end

@parallel_indices (i,j,k) function update_q_kernel!(q, u, weight, sigma, alphainv, resinv1, resinv2, resinv3)
    # compute symgrad(u)
    wxx = resinv1 * (u[1,i+1, j, k] - u[1,i, j, k])
    wxy = 0.5 * resinv1 * (u[2,i+1, j, k] - u[2,i, j, k])
    wxz = 0.5 * resinv1 * (u[3,i+1, j, k] - u[3,i, j, k])

    wxy = wxy + 0.5 * resinv2 * (u[1,i, j+1, k] - u[1,i, j, k])
    wyy = resinv2 * (u[2,i, j+1, k] - u[2,i, j, k])
    wyz = 0.5 * resinv2 * (u[3,i, j+1, k] - u[3,i, j, k])

    wxz = wxz + 0.5 * resinv3 * (u[1,i, j, k+1] - u[1,i, j, k])
    wyz = wyz + 0.5 * resinv3 * (u[2,i, j, k+1] - u[2,i, j, k])
    wzz = resinv3 * (u[3,i, j, k+1] - u[3,i, j, k])

    sigmaw = sigma * weight[i, j, k]

    wxx = q[1,i,j,k] + sigmaw * wxx
    wxy = q[2,i,j,k] + sigmaw * wxy
    wxz = q[3,i,j,k] + sigmaw * wxz
    wyy = q[4,i,j,k] + sigmaw * wyy
    wyz = q[5,i,j,k] + sigmaw * wyz
    wzz = q[6,i,j,k] + sigmaw * wzz

    qabs = sqrt(wxx * wxx + wyy * wyy + wzz * wzz +
                2 * (wxy * wxy + wxz * wxz + wyz * wyz)) * alphainv
    qabs = (qabs > 1) ? 1 / qabs : 1

    q[1,i,j,k] = wxx * qabs
    q[2,i,j,k] = wxy * qabs
    q[3,i,j,k] = wxz * qabs
    q[4,i,j,k] = wyy * qabs
    q[5,i,j,k] = wyz * qabs
    q[6,i,j,k] = wzz * qabs
    return
end

# Update phi_dest <- (phi + tau*laplace(mask0*eta))/(1+mask*tau). 
function tgv_update_phi!(phi_dest, phi, eta, mask, mask0, tau, res)
    taup1inv = 1 / (tau + 1)
    resinv = res .^ -2
    indices = (2:size(phi_dest,1)-1, 2:size(phi_dest,2)-1, 2:size(phi_dest,3)-1)

    @parallel indices update_phi_kernel!(phi_dest, phi, eta, mask, mask0, tau, taup1inv, resinv...)
end

@parallel_indices (i,j,k)  function update_phi_kernel!(phi_dest, phi, eta, mask, mask0, tau, taup1inv, resinv1, resinv2, resinv3)
    # compute laplace(mask*eta)
    v0 = mask0[i, j, k] * eta[i, j, k]
    v1m = mask0[i-1, j, k] * eta[i-1, j, k]
    v1p = mask0[i+1, j, k] * eta[i+1, j, k]
    v2m = mask0[i, j-1, k] * eta[i, j-1, k]
    v2p = mask0[i, j+1, k] * eta[i, j+1, k]
    v3m = mask0[i, j, k-1] * eta[i, j, k-1]
    v3p = mask0[i, j, k+1] * eta[i, j, k+1]

    laplace = (-2 * v0 + v1m + v1p) * resinv1 +
              (-2 * v0 + v2m + v2p) * resinv2 +
              (-2 * v0 + v3m + v3p) * resinv3
    fac = taup1inv ^ mask[i,j,k]
    phi_dest[i, j, k] = (phi[i, j, k] + tau * laplace) * fac
    return
end

# Update chi_dest <- chi + tau*(div(p) - wave(mask*v)). 
function tgv_update_chi!(chi_dest, chi, v, p, tensor, mask0, tau, res, omega)
    resinv = 1 ./ res
    wresinv = [1 / 3, 1 / 3, -2 / 3] ./ (res .^ 2)
    indices = (2:size(chi_dest,1)-1, 2:size(chi_dest,2)-1, 2:size(chi_dest,3)-1)

    @parallel indices update_chi_kernel!(chi_dest, chi, v, p, tensor, mask0, tau, resinv..., wresinv...)
end

@parallel_indices (i,j,k) function update_chi_kernel!(chi_dest, chi, v, p, tensor, mask0, tau, resinv1, resinv2, resinv3, wresinv1, wresinv2, wresinv3)
    m0 = mask0[i, j, k]

    t123 = (tensor[1,i,j,k], tensor[2,i,j,k], tensor[3,i,j,k])
    t245 = (tensor[2,i,j,k], tensor[4,i,j,k], tensor[5,i,j,k])
    t356 = (tensor[3,i,j,k], tensor[5,i,j,k], tensor[6,i,j,k])
    p_ = (p[1,i,j,k], p[2,i,j,k], p[3,i,j,k])
    p1 = sum(t123 .* p_)
    p2 = sum(t245 .* p_)
    p3 = sum(t356 .* p_)

    # compute div(weight*v)
    div = m0 * p1 * resinv1
    
        px = sum(t123 .* (p[1,i-1, j, k], p[2,i-1, j, k], p[3,i-1, j, k]))
        div = div - mask0[i-1, j, k] * px * resinv1
    
    div =  div + m0 * p2 * resinv2
    
        py = sum(t245 .* (p[1,i, j-1, k], p[2,i, j-1, k], p[3,i, j-1, k]))
        div = div - mask0[i, j-1, k] * py * resinv2
    
    div =  div + m0 * p3 * resinv3
    
        pz = sum(t356 .* (p[1,i, j, k-1], p[2,i, j, k-1], p[3,i, j, k-1]))
        div = div - mask0[i, j, k-1] * pz * resinv3
    

    # compute wave(mask*v)
    v0 = m0 * v[i, j, k]
    v1m = mask0[i-1, j, k] * v[i-1, j, k]
    v1p = mask0[i+1, j, k] * v[i+1, j, k]
    v2m = mask0[i, j-1, k] * v[i, j-1, k]
    v2p = mask0[i, j+1, k] * v[i, j+1, k]
    v3m = mask0[i, j, k-1] * v[i, j, k-1]
    v3p = mask0[i, j, k+1] * v[i, j, k+1]

    wave = (-2 * v0 + v1m + v1p) * wresinv1 +
           (-2 * v0 + v2m + v2p) * wresinv2 +
           (-2 * v0 + v3m + v3p) * wresinv3

    chi_dest[i, j, k] = chi[i, j, k] + tau * (div - wave)
    return
end

# Update w_dest <- w + tau*(mask*p + div(mask0*q)). 
function tgv_update_w!(w_dest, w, p, q, tensor, mask, mask0, tau, res)
    resinv = 1 ./ res
    indices = (2:size(w_dest,1)-1, 2:size(w_dest,2)-1,2:size(w_dest,3)-1)

    @parallel indices update_w_kernel!(w_dest, w, p, q, tensor, mask, mask0, tau, resinv...)
end

@parallel_indices (i,j,k) function update_w_kernel!(w_dest, w, p, q, tensor, mask, mask0, tau, resinv1, resinv2, resinv3)
    # if mask[i,j,k]
        w0 = mask0[i, j, k]
        w1 = mask0[i-1, j, k] 
        w2 = mask0[i, j-1, k] 
        w3 = mask0[i, j, k-1] 

        q0x = w0 * q[1,i,j,k] * resinv1
        q0x = q0x - w1 * q[1, i-1, j, k] * resinv1
        q1y = w0 * q[2,i,j,k] * resinv1
        q1y = q1y - w2 * q[2,i, j-1, k] * resinv1
        q2z = w0 * q[3,i,j,k] * resinv1
        q2z = q2z - w3 * q[3,i, j, k-1] * resinv1

        q1x = w0 * q[2,i,j,k] * resinv2
        q1x = q1x - w1 * q[2,i-1, j, k] * resinv2
        q3y = w0 * q[4,i,j,k] * resinv2
        q3y = q3y - w2 * q[4,i, j-1, k] * resinv2
        q4z = w0 * q[5,i,j,k] * resinv2
        q4z = q4z - w3 * q[5,i, j, k-1] * resinv2

        q2x = w0 * q[3,i,j,k] * resinv3
        q2x = q2x - w1 * q[3, i-1, j, k] * resinv3
        q4y = w0 * q[5,i,j,k] * resinv3
        q4y = q4y - w2 * q[5,i, j-1, k] * resinv3
        q5z = w0 * q[6,i,j,k] * resinv3
        q5z = q5z - w3 * q[6,i, j, k-1] * resinv3

        m0 = mask[i, j, k]

        t123 = (tensor[1,i,j,k], tensor[2,i,j,k], tensor[3,i,j,k])
        t245 = (tensor[2,i,j,k], tensor[4,i,j,k], tensor[5,i,j,k])
        t356 = (tensor[3,i,j,k], tensor[5,i,j,k], tensor[6,i,j,k])
        p_ = (p[1,i,j,k], p[2,i,j,k], p[3,i,j,k])
        p1 = sum(t123 .* p_)
        p2 = sum(t245 .* p_)
        p3 = sum(t356 .* p_)

        w_dest[1,i,j,k] = w[1,i,j,k] + tau * (m0 * p1 + q0x + q1y + q2z)
        w_dest[2,i,j,k] = w[2,i,j,k] + tau * (m0 * p2 + q1x + q3y + q4z)
        w_dest[3,i,j,k] = w[3,i,j,k] + tau * (m0 * p3 + q2x + q4y + q5z)
    # end
    return
end

function extragradient_update(u_, u)
    u_ .= 2 .* u .- u_
end
