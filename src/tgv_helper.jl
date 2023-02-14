# Update eta <- eta + sigma*mask*(-laplace(phi) + wave(chi) - laplace_phi0). 
function tgv_update_eta!(eta, phi, chi, laplace_phi0, mask, sigma, res, omega)
    type = eltype(eta)

    nx = size(eta, 1)
    ny = size(eta, 2)
    nz = size(eta, 3)

    res_inv = res .^ -2
    res_inv_0 = 2 * sum(res_inv)
    wres_inv = [1 / 3, 1 / 3, -2 / 3] .* res_inv
    wres_inv_0 = -2 * sum(wres_inv)

    @tullio eta[i, j, k] += begin
        laplace = phi[i, j, k] * res_inv_0 - (phi[i-1, j, k] + phi[i+1, j, k]) * res_inv[1] - (phi[i, j-1, k] + phi[i, j+1, k]) * res_inv[2] - (phi[i, j, k-1] + phi[i, j, k+1]) * res_inv[3]
        wave = chi[i, j, k] * wres_inv_0 + (chi[i-1, j, k] + chi[i+1, j, k]) * wres_inv[1] + (chi[i, j-1, k] + chi[i, j+1, k]) * wres_inv[2] + (chi[i, j, k-1] + chi[i, j, k+1]) * wres_inv[3]
        sigma * mask[i, j, k] * (laplace + wave - laplace_phi0[i, j, k])
    end

    # for i in axes(eta, 1)
    #     for j in axes(eta, 2)
    #         for k in axes(eta, 3)
    #             # compute -laplace(phi)
    #             phi0 = phi[i, j, k]
    #             phi1m = if (i > 1)
    #                 phi[i-1, j, k]
    #             else
    #                 phi0
    #             end
    #             phi1p = if (i < nx)
    #                 phi[i+1, j, k]
    #             else
    #                 phi0
    #             end
    #             phi2m = if (j > 1)
    #                 phi[i, j-1, k]
    #             else
    #                 phi0
    #             end
    #             phi2p = if (j < ny)
    #                 phi[i, j+1, k]
    #             else
    #                 phi0
    #             end
    #             phi3m = if (k > 1)
    #                 phi[i, j, k-1]
    #             else
    #                 phi0
    #             end
    #             phi3p = if (k < nz)
    #                 phi[i, j, k+1]
    #             else
    #                 phi0
    #             end

    #             laplace = (2phi0 - phi1m - phi1p) * res_inv[1] +
    #                       (2phi0 - phi2m - phi2p) * res_inv[2] +
    #                       (2phi0 - phi3m - phi3p) * res_inv[3]

    #             # compute wave(chi)
    #             chi0 = chi[i, j, k]
    #             chi1m = if (i > 1)
    #                 chi[i-1, j, k]
    #             else
    #                 chi0
    #             end
    #             chi1p = if (i < nx)
    #                 chi[i+1, j, k]
    #             else
    #                 chi0
    #             end
    #             chi2m = if (j > 1)
    #                 chi[i, j-1, k]
    #             else
    #                 chi0
    #             end
    #             chi2p = if (j < ny)
    #                 chi[i, j+1, k]
    #             else
    #                 chi0
    #             end
    #             chi3m = if (k > 1)
    #                 chi[i, j, k-1]
    #             else
    #                 chi0
    #             end
    #             chi3p = if (k < nz)
    #                 chi[i, j, k+1]
    #             else
    #                 chi0
    #             end

    #             wave = (-2chi0 + chi1m + chi1p) * wres_inv[1] +
    #                    (-2chi0 + chi2m + chi2p) * wres_inv[2] +
    #                    (-2chi0 + chi3m + chi3p) * wres_inv[3]

    #             eta[i, j, k] += sigma * mask[i, j, k] * (laplace + wave - laplace_phi0[i, j, k])
    #         end
    #     end
    # end
end

# Update p <- P_{||.||_\infty <= alpha}(p + sigma*(mask0*grad(phi_f) - mask*w). 
function tgv_update_p!(p, chi, w, mask, mask0, sigma, alpha, res)
    type = eltype(p)

    nx = size(p, 1)
    ny = size(p, 2)
    nz = size(p, 3)

    alphainv = 1 / alpha
    res_inv = 1 ./ res

    @tullio p[i, j, k, 1] = begin
        chi0 = chi[i, j, k]

        dp = (chi[i+1, j, k] - chi0) * res_inv[1]

        sigmaw0 = sigma * mask0[i, j, k]
        sigmaw = sigma * mask[i, j, k]

        p[i, j, k, 1] + sigmaw0 * dp - sigmaw * w[i, j, k, 1]
    end
    @tullio p[i, j, k, 2] = begin
        chi0 = chi[i, j, k]

        dp = (chi[i, j+1, k] - chi0) * res_inv[2]

        sigmaw0 = sigma * mask0[i, j, k]
        sigmaw = sigma * mask[i, j, k]

        p[i, j, k, 2] + sigmaw0 * dp - sigmaw * w[i, j, k, 2]
    end
    @tullio p[i, j, k, 3] = begin
        chi0 = chi[i, j, k]

        dp = (chi[i, j, k+1] - chi0) * res_inv[3]

        sigmaw0 = sigma * mask0[i, j, k]
        sigmaw = sigma * mask[i, j, k]

        p[i, j, k, 3] + sigmaw0 * dp - sigmaw * w[i, j, k, 3]
    end

    # for i in axes(p, 1)
    #     for j in axes(p, 2)
    #         for k in axes(p, 3)
    #             chi0 = chi[i, j, k]

    #             dxp = if (i < nx)
    #                 (chi[i+1, j, k] - chi0) * res_inv[1]
    #             else
    #                 zero(type)
    #             end
    #             dyp = if (j < ny)
    #                 (chi[i, j+1, k] - chi0) * res_inv[2]
    #             else
    #                 zero(type)
    #             end
    #             dzp = if (k < nz)
    #                 (chi[i, j, k+1] - chi0) * res_inv[3]
    #             else
    #                 zero(type)
    #             end

    #             sigmaw0 = sigma * mask0[i, j, k]
    #             sigmaw = sigma * mask[i, j, k]

    #             px = p[i, j, k, 1] + sigmaw0 * dxp - sigmaw * w[i, j, k, 1]
    #             py = p[i, j, k, 2] + sigmaw0 * dyp - sigmaw * w[i, j, k, 2]
    #             pz = p[i, j, k, 3] + sigmaw0 * dzp - sigmaw * w[i, j, k, 3]

    #             pabs = sqrt(px * px + py * py * pz * pz) * alphainv # TODO looks weird
    #             pabs = if (pabs > 1)
    #                 1 / pabs
    #             else
    #                 one(type)
    #             end

    #             p[i, j, k, 1] = px * pabs
    #             p[i, j, k, 2] = py * pabs
    #             p[i, j, k, 3] = pz * pabs
    #         end
    #     end
    # end
end
# Update q <- P_{||.||_\infty <= alpha}(q + sigma*weight*symgrad(u)). 
function tgv_update_q!(q, u, weight, sigma, alpha, res)
    type = eltype(q)

    nx = size(q, 1)
    ny = size(q, 2)
    nz = size(q, 3)

    alphainv = 1 / alpha
    res_inv = 1 ./ res
    res_inv2 = 0.5 ./ res

    @tullio q[i, j, k, 1] += sigma * weight[i, j, k] * res_inv[1] * (u[i+1, j, k, 1] - u[i, j, k, 1])
    @tullio q[i, j, k, 4] += sigma * weight[i, j, k] * res_inv[2] * (u[i, j+1, k, 2] - u[i, j, k, 2])
    @tullio q[i, j, k, 6] += sigma * weight[i, j, k] * res_inv[3] * (u[i, j, k+1, 3] - u[i, j, k, 3])
    @tullio q[i, j, k, 2] += sigma * weight[i, j, k] * (res_inv2[1] * (u[i+1, j, k, 2] - u[i, j, k, 2]) + res_inv2[2] * (u[i, j+1, k, 1] - u[i, j, k, 1]))
    @tullio q[i, j, k, 3] += sigma * weight[i, j, k] * (res_inv2[1] * (u[i+1, j, k, 3] - u[i, j, k, 3]) + res_inv2[3] * (u[i, j, k+1, 1] - u[i, j, k, 1]))
    @tullio q[i, j, k, 5] += sigma * weight[i, j, k] * (res_inv2[2] * (u[i, j+1, k, 3] - u[i, j, k, 3]) + res_inv2[3] * (u[i, j, k+1, 2] - u[i, j, k, 2]))

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
    # q .*= pmap()


    # for i in axes(q, 1)
    #     for j in axes(q, 2)
    #         for k in axes(q, 3)
    #             # compute symgrad(u)
    #             if (i < nx)
    #                 wxx = res_inv[1] * (u[i+1, j, k, 1] - u[i, j, k, 1])
    #                 wxy = res_inv2[1] * (u[i+1, j, k, 2] - u[i, j, k, 2])
    #                 wxz = res_inv2[1] * (u[i+1, j, k, 3] - u[i, j, k, 3])
    #             else
    #                 wxx = zero(type)
    #                 wxy = zero(type)
    #                 wxz = zero(type)
    #             end

    #             if (j < ny)
    #                 wxy = wxy + res_inv2[2] * (u[i, j+1, k, 1] - u[i, j, k, 1])
    #                 wyy = res_inv[2] * (u[i, j+1, k, 2] - u[i, j, k, 2])
    #                 wyz = res_inv2[2] * (u[i, j+1, k, 3] - u[i, j, k, 3])
    #             else
    #                 wyy = zero(type)
    #                 wyz = zero(type)
    #             end

    #             if (k < nz)
    #                 wxz = wxz + res_inv2[3] * (u[i, j, k+1, 1] - u[i, j, k, 1])
    #                 wyz = wyz + res_inv2[3] * (u[i, j, k+1, 2] - u[i, j, k, 2])
    #                 wzz = res_inv[3] * (u[i, j, k+1, 3] - u[i, j, k, 3])
    #             else
    #                 wzz = zero(type)
    #             end

    #             sigmaw = sigma * weight[i, j, k]

    #             wxx = q[i, j, k, 1] + sigmaw * wxx
    #             wxy = q[i, j, k, 2] + sigmaw * wxy
    #             wxz = q[i, j, k, 3] + sigmaw * wxz
    #             wyy = q[i, j, k, 4] + sigmaw * wyy
    #             wyz = q[i, j, k, 5] + sigmaw * wyz
    #             wzz = q[i, j, k, 6] + sigmaw * wzz

    #             qabs = sqrt(wxx * wxx + wyy * wyy + wzz * wzz + 2 * (wxy * wxy + wxz * wxz + wyz * wyz)) * alphainv
    #             qabs = if (qabs > 1)
    #                 1 / qabs
    #             else
    #                 one(type)
    #             end

    #             q[i, j, k, 1] = wxx * qabs
    #             q[i, j, k, 2] = wxy * qabs
    #             q[i, j, k, 3] = wxz * qabs
    #             q[i, j, k, 4] = wyy * qabs
    #             q[i, j, k, 5] = wyz * qabs
    #             q[i, j, k, 6] = wzz * qabs
    #         end
    #     end
    # end
end


# Update phi_dest <- (phi + tau*laplace(mask0*eta))/(1+mask*tau). 
function tgv_update_phi!(phi_dest, phi, eta, mask, mask0, tau, res)
    type = eltype(phi)

    nx = size(phi, 1)
    ny = size(phi, 2)
    nz = size(phi, 3)

    taup1inv = 1 / (tau + 1)

    res_inv = res .^ -2
    res_inv_0 = -2 * sum(res)

    @tullio phi_dest[i, j, k] = begin
        laplace = eta[i, j, k] * res_inv_0 + (eta[i-1, j, k] * mask0[i-1, j, k] + eta[i+1, j, k] * mask0[i+1, j, k]) * res_inv[1] + (eta[i, j-1, k] * mask0[i, j-1, k] + eta[i, j+1, k] * mask0[i, j+1, k]) * res_inv[2] + (eta[i, j, k-1] * mask0[i, j, k-1] + eta[i, j, k+1] * mask0[i, j, k+1]) * res_inv[3]
        fac = mask[i, j, k] * taup1inv + !mask[i, j, k]
        (phi[i, j, k] + tau * laplace) * fac
    end

    # for i in axes(phi, 1)
    #     for j in axes(phi, 2)
    #         for k in axes(phi, 3)
    #             # compute laplace(mask*eta)
    #             v0 = mask0[i, j, k] * eta[i, j, k]
    #             v1m = if (i > 1)
    #                 mask0[i-1, j, k] * eta[i-1, j, k]
    #             else
    #                 v0
    #             end
    #             v1p = if (i < nx)
    #                 mask0[i+1, j, k] * eta[i+1, j, k]
    #             else
    #                 v0
    #             end
    #             v2m = if (j > 1)
    #                 mask0[i, j-1, k] * eta[i, j-1, k]
    #             else
    #                 v0
    #             end
    #             v2p = if (j < ny)
    #                 mask0[i, j+1, k] * eta[i, j+1, k]
    #             else
    #                 v0
    #             end
    #             v3m = if (k > 1)
    #                 mask0[i, j, k-1] * eta[i, j, k-1]
    #             else
    #                 v0
    #             end
    #             v3p = if (k < nz)
    #                 mask0[i, j, k+1] * eta[i, j, k+1]
    #             else
    #                 v0
    #             end

    #             laplace = (-2 * v0 + v1m + v1p) * res_inv[1] +
    #                       (-2 * v0 + v2m + v2p) * res_inv[2] +
    #                       (-2 * v0 + v3m + v3p) * res_inv[3]

    #             fac = if mask[i, j, k]
    #                 taup1inv
    #             else
    #                 one(type)
    #             end
    #             phi_dest[i, j, k] = (phi[i, j, k] + tau * laplace) * fac
    #         end
    #     end
    # end
end


# Update chi_dest <- chi + tau*(div(p) - wave(mask*v)). 
function tgv_update_chi!(chi_dest, chi, v, p, mask0, tau, res, omega)
    type = eltype(chi)

    nx = size(chi, 1)
    ny = size(chi, 2)
    nz = size(chi, 3)

    res_inv = 1 ./ res

    wres_inv = [1 / 3, 1 / 3, -2 / 3] ./ (res .^ 2)
    wres_inv_0 = -2 * sum(wres_inv)

    # cdef float wres0inv = <float>(1.0/3.0)/(res0**2) - (omega0**2)
    # cdef float wres1inv = <float>(1.0/3.0)/(res1**2) - (omega1**2)
    # cdef float wres2inv = <float>(1.0/3.0)/(res2**2) - (omega2**2)

    @tullio chi_dest[i, j, k] = begin
        div = mask0[i, j, k] * (p[i, j, k, 1] * res_inv[1] + p[i, j, k, 2] * res_inv[2] + p[i, j, k, 3] * res_inv[3]) - mask0[i-1, j, k] * p[i-1, j, k, 1] * res_inv[1] - mask0[i, j-1, k] * p[i, j-1, k, 2] * res_inv[2] - mask0[i, j, k-1] * p[i, j, k-1, 3] * res_inv[3]
        wave = v[i, j, k] * wres_inv_0 + (v[i-1, j, k] * mask0[i-1, j, k] + v[i+1, j, k] * mask0[i+1, j, k]) * wres_inv[1] + (v[i, j-1, k] * mask0[i, j-1, k] + v[i, j+1, k] * mask0[i, j+1, k]) * wres_inv[2] + (v[i, j, k-1] * mask0[i, j, k-1] + v[i, j, k+1] * mask0[i, j, k+1]) * wres_inv[3]
        chi[i, j, k] + tau * (div - wave)
    end

    # @tullio div[i, j, k] := mask0[i, j, k] * p[i, j, k, n] - mask0[i-1, j, k] * p[i-1, j, k, 1] - mask0[i, j-1, k] * p[i, j-1, k, 2] - mask0[i, j, k-1] * p[i, j, k-1, 3]

    # @tullio wave[i, j, k] := v[i, j, k] * wres_inv_0 + (v[i-1, j, k] + v[i+1, j, k]) * wres_inv[1] + (v[i, j-1, k] + v[i, j+1, k]) * wres_inv[2] + (v[i, j, k-1] + v[i, j, k+1]) * wres_inv[3]

    # @tullio chi_dest[i, j, k] = chi[i, j, k] + tau * (div[i, j, k] - wave[i, j, k])

    # for i in axes(chi, 1)
    #     for j in axes(chi, 2)
    #         for k in axes(chi, 3)
    #             m0 = mask0[i, j, k]

    #             # compute div(weight*v)
    #             div = if i < nx
    #                 m0 * p[i, j, k, 1] * res_inv[1]
    #             else
    #                 zero(type)
    #             end
    #             div = if i > 1
    #                 div - mask0[i-1, j, k] * p[i-1, j, k, 1] * res_inv[1]
    #             else
    #                 div
    #             end

    #             div = if j < ny
    #                 div + m0 * p[i, j, k, 2] * res_inv[2]
    #             else
    #                 div
    #             end
    #             div = if j > 1
    #                 div - mask0[i, j-1, k] * p[i, j-1, k, 2] * res_inv[2]
    #             else
    #                 div
    #             end

    #             div = if k < nz
    #                 div + m0 * p[i, j, k, 3] * res_inv[3]
    #             else
    #                 div
    #             end
    #             div = if k > 1
    #                 div - mask0[i, j, k-1] * p[i, j, k-1, 3] * res_inv[3]
    #             else
    #                 div
    #             end

    #             # compute wave(mask*v)
    #             v0 = m0 * v[i, j, k]
    #             v1m = if (i > 1)
    #                 mask0[i-1, j, k] * v[i-1, j, k]
    #             else
    #                 v0
    #             end
    #             v1p = if (i < nx)
    #                 mask0[i+1, j, k] * v[i+1, j, k]
    #             else
    #                 v0
    #             end
    #             v2m = if (j > 1)
    #                 mask0[i, j-1, k] * v[i, j-1, k]
    #             else
    #                 v0
    #             end
    #             v2p = if (j < ny)
    #                 mask0[i, j+1, k] * v[i, j+1, k]
    #             else
    #                 v0
    #             end
    #             v3m = if (k > 1)
    #                 mask0[i, j, k-1] * v[i, j, k-1]
    #             else
    #                 v0
    #             end
    #             v3p = if (k < nz)
    #                 mask0[i, j, k+1] * v[i, j, k+1]
    #             else
    #                 v0
    #             end

    #             wave = (-2 * v0 + v1m + v1p) * wres_inv[1] +
    #                    (-2 * v0 + v2m + v2p) * wres_inv[2] +
    #                    (-2 * v0 + v3m + v3p) * wres_inv[3]

    #             chi_dest[i, j, k] = chi[i, j, k] + tau * (div - wave)
    #         end
    #     end
    # end
end

# Update w_dest <- w + tau*(mask*p + div(mask0*q)). 
function tgv_update_w!(w_dest, w, p, q, mask, mask0, tau, res)

    res_inv = 1 ./ res

    @tullio w_dest[i, j, k, 1] = @inbounds(begin
        q0x = mask0[i, j, k] * q[i, j, k, 1] - mask0[i-1, j, k] * q[i-1, j, k, 1]
        q1y = mask0[i, j, k] * q[i, j, k, 2] - mask0[i, j-1, k] * q[i, j-1, k, 2]
        q2z = mask0[i, j, k] * q[i, j, k, 3] - mask0[i, j, k-1] * q[i, j, k-1, 3]
        w[i, j, k, 1] + tau * (mask[i, j, k] * p[i, j, k, 1] + (q0x + q1y + q2z) * res_inv[1])
    end)
    @tullio w_dest[i, j, k, 2] = begin
        q1x = mask0[i, j, k] * q[i, j, k, 1] - mask0[i-1, j, k] * q[i-1, j, k, 1]
        q3y = mask0[i, j, k] * q[i, j, k, 3] - mask0[i, j-1, k] * q[i, j-1, k, 3]
        q4z = mask0[i, j, k] * q[i, j, k, 4] - mask0[i, j, k-1] * q[i, j, k-1, 4]
        w[i, j, k, 2] + tau * (mask[i, j, k] * p[i, j, k, 2] + (q1x + q3y + q4z) * res_inv[2])
    end
    @tullio w_dest[i, j, k, 3] = begin
        q2x = mask0[i, j, k] * q[i, j, k, 2] - mask0[i-1, j, k] * q[i-1, j, k, 2]
        q4y = mask0[i, j, k] * q[i, j, k, 4] - mask0[i, j-1, k] * q[i, j-1, k, 4]
        q5z = mask0[i, j, k] * q[i, j, k, 5] - mask0[i, j, k-1] * q[i, j, k-1, 5]
        w[i, j, k, 3] + tau * (mask[i, j, k] * p[i, j, k, 3] + (q2x + q4y + q5z) * res_inv[3])
    end
    # res_inv_dim4 = reshape(res .^ -1, 1, 1, 1, 3)

    # Threads.@spawn qx[2:end, :, :, :] .= diff(mask0 .* view(q, :, :, :, [1, 2, 3]); dims=1)
    # Threads.@spawn qy[:, 2:end, :, :] .= diff(mask0 .* view(q, :, :, :, [2, 4, 5]); dims=2)
    # Threads.@spawn qz[:, :, 2:end, :] .= diff(mask0 .* view(q, :, :, :, [3, 5, 6]); dims=3)

    # w_dest .= w .+ tau .* (mask .* p .+ (fetch(qx) .+ fetch(qy) .+ fetch(qz)) .* res_inv_dim4)
end

function extragradient_update(u_, u)
    @tullio u_[i] = 2 * u[i] - u_[i]
end
