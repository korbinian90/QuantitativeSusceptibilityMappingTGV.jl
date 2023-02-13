# Update eta <- eta + sigma*mask*(-laplace(phi) + wave(chi) - laplace_phi0). 
function tgv_update_eta!(eta, phi, chi, laplace_phi0, mask, sigma, res, omega)

    nx = size(eta, 1)
    ny = size(eta, 2)
    nz = size(eta, 3)

    res_inv = res .^ -2
    wres_inv = [1 / 3, 1 / 3, -2 / 3] .* res_inv

    for i in axes(eta, 1)
        for j in axes(eta, 2)
            for k in axes(eta, 3)
                # compute -laplace(phi)
                phi0 = phi[i, j, k]
                phi1m = if (i > 1)
                    phi[i-1, j, k]
                else
                    phi0
                end
                phi1p = if (i < nx)
                    phi[i+1, j, k]
                else
                    phi0
                end
                phi2m = if (j > 1)
                    phi[i, j-1, k]
                else
                    phi0
                end
                phi2p = if (j < ny)
                    phi[i, j+1, k]
                else
                    phi0
                end
                phi3m = if (k > 1)
                    phi[i, j, k-1]
                else
                    phi0
                end
                phi3p = if (k < nz)
                    phi[i, j, k+1]
                else
                    phi0
                end

                laplace = (2phi0 - phi1m - phi1p) * res_inv[1] +
                          (2phi0 - phi2m - phi2p) * res_inv[2] +
                          (2phi0 - phi3m - phi3p) * res_inv[3]

                # compute wave(chi)
                chi0 = chi[i, j, k]
                chi1m = if (i > 1)
                    chi[i-1, j, k]
                else
                    chi0
                end
                chi1p = if (i < nx)
                    chi[i+1, j, k]
                else
                    chi0
                end
                chi2m = if (j > 1)
                    chi[i, j-1, k]
                else
                    chi0
                end
                chi2p = if (j < ny)
                    chi[i, j+1, k]
                else
                    chi0
                end
                chi3m = if (k > 1)
                    chi[i, j, k-1]
                else
                    chi0
                end
                chi3p = if (k < nz)
                    chi[i, j, k+1]
                else
                    chi0
                end

                wave = (-2chi0 + chi1m + chi1p) * wres_inv[1] +
                       (-2chi0 + chi2m + chi2p) * wres_inv[2] +
                       (-2chi0 + chi3m + chi3p) * wres_inv[3]

                eta[i, j, k] += sigma * mask[i, j, k] * (laplace + wave - laplace_phi0[i, j, k])
            end
        end
    end
end

# Update p <- P_{||.||_\infty <= alpha}(p + sigma*(mask0*grad(phi_f) - mask*w). 
function tgv_update_p!(p, chi, w, mask, mask0, sigma, alpha, res)

    nx = size(p, 1)
    ny = size(p, 2)
    nz = size(p, 3)

    alphainv = 1 / alpha
    res_inv = 1 ./ res

    for i in axes(p, 1)
        for j in axes(p, 2)
            for k in axes(p, 3)
                chi0 = chi[i, j, k]

                dxp = if (i < nx)
                    (chi[i+1, j, k] - chi0) * res_inv[1]
                else
                    0
                end
                dyp = if (j < ny)
                    (chi[i, j+1, k] - chi0) * res_inv[2]
                else
                    0
                end
                dzp = if (k < nz)
                    (chi[i, j, k+1] - chi0) * res_inv[3]
                else
                    0
                end

                sigmaw0 = sigma * mask0[i, j, k]
                sigmaw = sigma * mask[i, j, k]

                px = p[i, j, k, 1] + sigmaw0 * dxp - sigmaw * w[i, j, k, 1]
                py = p[i, j, k, 2] + sigmaw0 * dyp - sigmaw * w[i, j, k, 2]
                pz = p[i, j, k, 3] + sigmaw0 * dzp - sigmaw * w[i, j, k, 3]

                pabs = sqrt(px * px + py * py * pz * pz) * alphainv # TODO looks weird
                pabs = if (pabs > 1)
                    1 / pabs
                else
                    1
                end

                p[i, j, k, 1] = px * pabs
                p[i, j, k, 2] = py * pabs
                p[i, j, k, 3] = pz * pabs
            end
        end
    end
end
# Update q <- P_{||.||_\infty <= alpha}(q + sigma*weight*symgrad(u)). 
function tgv_update_q!(q, u, weight, sigma, alpha, res)

    nx = size(q, 1)
    ny = size(q, 2)
    nz = size(q, 3)

    alphainv = 1 / alpha
    res_inv = 1 ./ res
    res_inv2 = 0.5 ./ res

    for i in axes(q, 1)
        for j in axes(q, 2)
            for k in axes(q, 3)
                # compute symgrad(u)
                if (i < nx)
                    wxx = res_inv[1] * (u[i+1, j, k, 1] - u[i, j, k, 1])
                    wxy = res_inv2[1] * (u[i+1, j, k, 2] - u[i, j, k, 2])
                    wxz = res_inv2[1] * (u[i+1, j, k, 3] - u[i, j, k, 3])
                else
                    wxx = 0
                    wxy = 0
                    wxz = 0
                end

                if (j < ny)
                    wxy = wxy + res_inv2[2] * (u[i, j+1, k, 1] - u[i, j, k, 1])
                    wyy = res_inv[2] * (u[i, j+1, k, 2] - u[i, j, k, 2])
                    wyz = res_inv2[2] * (u[i, j+1, k, 3] - u[i, j, k, 3])
                else
                    wyy = 0
                    wyz = 0
                end

                if (k < nz)
                    wxz = wxz + res_inv2[3] * (u[i, j, k+1, 1] - u[i, j, k, 1])
                    wyz = wyz + res_inv2[3] * (u[i, j, k+1, 2] - u[i, j, k, 2])
                    wzz = res_inv[3] * (u[i, j, k+1, 3] - u[i, j, k, 3])
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

                qabs = sqrt(wxx * wxx + wyy * wyy + wzz * wzz + 2 * (wxy * wxy + wxz * wxz + wyz * wyz)) * alphainv
                qabs = if (qabs > 1)
                    1 / qabs
                else
                    1
                end

                q[i, j, k, 1] = wxx * qabs
                q[i, j, k, 2] = wxy * qabs
                q[i, j, k, 3] = wxz * qabs
                q[i, j, k, 4] = wyy * qabs
                q[i, j, k, 5] = wyz * qabs
                q[i, j, k, 6] = wzz * qabs
            end
        end
    end
end


# Update phi_dest <- (phi + tau*laplace(mask0*eta))/(1+mask*tau). 
function tgv_update_phi!(phi_dest, phi, eta, mask, mask0, tau, res)

    nx = size(phi, 1)
    ny = size(phi, 2)
    nz = size(phi, 3)

    taup1inv = 1 / (tau + 1)

    res_inv = res .^ -2

    for i in axes(phi, 1)
        for j in axes(phi, 2)
            for k in axes(phi, 3)
                # compute laplace(mask*eta)
                v0 = mask0[i, j, k] * eta[i, j, k]
                v1m = if (i > 1)
                    mask0[i-1, j, k] * eta[i-1, j, k]
                else
                    v0
                end
                v1p = if (i < nx)
                    mask0[i+1, j, k] * eta[i+1, j, k]
                else
                    v0
                end
                v2m = if (j > 1)
                    mask0[i, j-1, k] * eta[i, j-1, k]
                else
                    v0
                end
                v2p = if (j < ny)
                    mask0[i, j+1, k] * eta[i, j+1, k]
                else
                    v0
                end
                v3m = if (k > 1)
                    mask0[i, j, k-1] * eta[i, j, k-1]
                else
                    v0
                end
                v3p = if (k < nz)
                    mask0[i, j, k+1] * eta[i, j, k+1]
                else
                    v0
                end

                laplace = (-2 * v0 + v1m + v1p) * res_inv[1] +
                          (-2 * v0 + v2m + v2p) * res_inv[2] +
                          (-2 * v0 + v3m + v3p) * res_inv[3]

                fac = if mask[i, j, k]
                    taup1inv
                else
                    1
                end
                phi_dest[i, j, k] = (phi[i, j, k] + tau * laplace) * fac
            end
        end
    end
end


# Update chi_dest <- chi + tau*(div(p) - wave(mask*v)). 
function tgv_update_chi!(chi_dest, chi, v, p, mask0, tau, res, omega)

    nx = size(chi, 1)
    ny = size(chi, 2)
    nz = size(chi, 3)

    res_inv = 1 ./ res

    wres_inv = [1 / 3, 1 / 3, -2 / 3] ./ (res .^ 2)

    # cdef float wres0inv = <float>(1.0/3.0)/(res0**2) - (omega0**2)
    # cdef float wres1inv = <float>(1.0/3.0)/(res1**2) - (omega1**2)
    # cdef float wres2inv = <float>(1.0/3.0)/(res2**2) - (omega2**2)

    for i in axes(chi, 1)
        for j in axes(chi, 2)
            for k in axes(chi, 3)
                m0 = mask0[i, j, k]

                # compute div(weight*v)
                div = if i < nx
                    m0 * p[i, j, k, 1] * res_inv[1]
                else
                    0
                end
                div = if i > 1
                    div - mask0[i-1, j, k] * p[i-1, j, k, 1] * res_inv[1]
                else
                    div
                end

                div = if j < ny
                    div + m0 * p[i, j, k, 2] * res_inv[2]
                else
                    div
                end
                div = if j > 1
                    div - mask0[i, j-1, k] * p[i, j-1, k, 2] * res_inv[2]
                else
                    div
                end

                div = if k < nz
                    div + m0 * p[i, j, k, 3] * res_inv[3]
                else
                    div
                end
                div = if k > 1
                    div - mask0[i, j, k-1] * p[i, j, k-1, 3] * res_inv[3]
                else
                    div
                end

                # compute wave(mask*v)
                v0 = m0 * v[i, j, k]
                v1m = if (i > 1)
                    mask0[i-1, j, k] * v[i-1, j, k]
                else
                    v0
                end
                v1p = if (i < nx)
                    mask0[i+1, j, k] * v[i+1, j, k]
                else
                    v0
                end
                v2m = if (j > 1)
                    mask0[i, j-1, k] * v[i, j-1, k]
                else
                    v0
                end
                v2p = if (j < ny)
                    mask0[i, j+1, k] * v[i, j+1, k]
                else
                    v0
                end
                v3m = if (k > 1)
                    mask0[i, j, k-1] * v[i, j, k-1]
                else
                    v0
                end
                v3p = if (k < nz)
                    mask0[i, j, k+1] * v[i, j, k+1]
                else
                    v0
                end

                wave = (-2 * v0 + v1m + v1p) * wres_inv[1] +
                       (-2 * v0 + v2m + v2p) * wres_inv[2] +
                       (-2 * v0 + v3m + v3p) * wres_inv[3]

                chi_dest[i, j, k] = chi[i, j, k] + tau * (div - wave)
            end
        end
    end
end

# Update w_dest <- w + tau*(mask*p + div(mask0*q)). 
function tgv_update_w!(w_dest, w, p, q, mask, mask0, tau, res)

    nx = size(w, 1)
    ny = size(w, 2)
    nz = size(w, 3)

    res_inv = 1 ./ res

    for i in axes(w, 1)
        for j in axes(w, 2)
            for k in axes(w, 3)
                w0 = mask0[i, j, k]
                w1 = if i > 1
                    mask0[i-1, j, k]
                else
                    0
                end
                w2 = if j > 1
                    mask0[i, j-1, k]
                else
                    0
                end
                w3 = if k > 1
                    mask0[i, j, k-1]
                else
                    0
                end

                q0x = if i < nx
                    w0 * q[i, j, k, 1] * res_inv[1]
                else
                    0
                end
                q0x = if i > 1
                    q0x - w1 * q[i-1, j, k, 1] * res_inv[1]
                else
                    q0x
                end
                q1y = if j < ny
                    w0 * q[i, j, k, 2] * res_inv[1]
                else
                    0
                end
                q1y = if j > 1
                    q1y - w2 * q[i, j-1, k, 2] * res_inv[1]
                else
                    q1y
                end
                q2z = if k < nz
                    w0 * q[i, j, k, 3] * res_inv[1]
                else
                    0
                end
                q2z = if k > 1
                    q2z - w3 * q[i, j, k-1, 3] * res_inv[1]
                else
                    q2z
                end

                q1x = if i < nx
                    w0 * q[i, j, k, 2] * res_inv[2]
                else
                    0
                end
                q1x = if i > 1
                    q1x - w1 * q[i-1, j, k, 2] * res_inv[2]
                else
                    q1x
                end
                q3y = if j < ny
                    w0 * q[i, j, k, 4] * res_inv[2]
                else
                    0
                end
                q3y = if j > 1
                    q3y - w2 * q[i, j-1, k, 4] * res_inv[2]
                else
                    q3y
                end
                q4z = if k < nz
                    w0 * q[i, j, k, 5] * res_inv[2]
                else
                    0
                end
                q4z = if k > 1
                    q4z - w3 * q[i, j, k-1, 5] * res_inv[2]
                else
                    q4z
                end

                q2x = if i < nx
                    w0 * q[i, j, k, 3] * res_inv[3]
                else
                    0
                end
                q2x = if i > 1
                    q2x - w1 * q[i-1, j, k, 3] * res_inv[3]
                else
                    q2x
                end
                q4y = if j < ny
                    w0 * q[i, j, k, 5] * res_inv[3]
                else
                    0
                end
                q4y = if j > 1
                    q4y - w2 * q[i, j-1, k, 5] * res_inv[3]
                else
                    q4y
                end
                q5z = if k < nz
                    w0 * q[i, j, k, 6] * res_inv[3]
                else
                    0
                end
                q5z = if k > 1
                    q5z - w3 * q[i, j, k-1, 6] * res_inv[3]
                else
                    q5z
                end

                m0 = mask[i, j, k]

                w_dest[i, j, k, 1] = w[i, j, k, 1] + tau * (m0 * p[i, j, k, 1] + q0x + q1y + q2z)
                w_dest[i, j, k, 2] = w[i, j, k, 2] + tau * (m0 * p[i, j, k, 2] + q1x + q3y + q4z)
                w_dest[i, j, k, 3] = w[i, j, k, 3] + tau * (m0 * p[i, j, k, 3] + q2x + q4y + q5z)
            end
        end
    end
end

function extragradient_update(u_, u)
    for i in eachindex(u_)
        u_[i] = 2 * u[i] - u_[i]
    end
end
