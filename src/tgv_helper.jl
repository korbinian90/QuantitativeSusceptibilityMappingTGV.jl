# Update eta <- eta + sigma*mask*(-laplace(phi) + wave(chi) - laplace_phi0). 
@kernel function update_eta_kernel!(eta, phi, chi, laplace_phi0, mask, sigma, laplace_kernel, dipole_kernel)
    I = @index(Global, Cartesian)
    R = @ndrange

    laplace = filter_local((I, R), phi, laplace_kernel)
    wave = filter_local((I, R), chi, dipole_kernel)

    @inbounds eta[I] += sigma * mask[I] * (-laplace + wave - laplace_phi0[I])
end

# Update p <- P_{||.||_\infty <= alpha}(p + sigma*(mask0*grad(phi_f) - mask*w). 
@kernel function update_p_kernel!(p, chi, w, tensor, mask, mask0, sigma, alphainv, resinv)
    I = @index(Global, Cartesian)
    R = @ndrange

    dxp, dyp, dzp = grad_local((I, R), chi, resinv)

    @inbounds begin
        sigmaw0 = sigma * mask0[I]
        sigmaw = sigma * mask[I]

        grad_minus_w = sigmaw0 .* (dxp, dyp, dzp) .- sigmaw .* (w[1, I], w[2, I], w[3, I])
        grad_minus_w = apply_tensor(grad_minus_w, tensor, I)

        p[1, I] += grad_minus_w[1]
        p[2, I] += grad_minus_w[2]
        p[3, I] += grad_minus_w[3]

        pabs = norm((p[it, I] for it in 1:3)) * alphainv
        if pabs > 1
            p[1, I] /= pabs
            p[2, I] /= pabs
            p[3, I] /= pabs
        end
    end
end
@inline norm((x, y, z)) = sqrt(x * x + y * y + z * z)

# Update q <- P_{||.||_\infty <= alpha}(q + sigma*weight*symgrad(u)).
@kernel function update_q_kernel!(q, u, weight, sigma, alphainv, resinv)
    I = @index(Global, Cartesian)
    R = @ndrange

    wxx, wxy, wxz, wyy, wyz, wzz = symgrad_local((I, R), u, resinv)

    @inbounds begin
        sigmaw = sigma * weight[I]

        q[1, I] += sigmaw * wxx
        q[2, I] += sigmaw * wxy
        q[3, I] += sigmaw * wxz
        q[4, I] += sigmaw * wyy
        q[5, I] += sigmaw * wyz
        q[6, I] += sigmaw * wzz

        qabs = qnorm(q[i, I] for i in 1:6) * alphainv
        if qabs > 1
            q[1, I] /= qabs
            q[2, I] /= qabs
            q[3, I] /= qabs
            q[4, I] /= qabs
            q[5, I] /= qabs
            q[6, I] /= qabs
        end
    end
end
@inline qnorm((xx, xy, xz, yy, yz, zz)) = sqrt(xx * xx + yy * yy + zz * zz + 2 * (xy * xy + xz * xz + yz * yz))

# Update phi_dest <- (phi + tau*laplace(mask0*eta))/(1+mask*tau). 
@kernel function update_phi_kernel!(phi_dest::AbstractArray{T}, phi, eta, mask, mask0, tau, laplace_kernel) where {T}
    I = @index(Global, Cartesian)
    R = @ndrange

    laplace = filter_local((I, R), eta, laplace_kernel, mask0)
    @inbounds phi_dest[I] = (phi[I] + tau * laplace) / (one(T) + mask[I] * tau)
end

# Update chi_dest <- chi + tau*(div(p) - wave(mask*v)). 
@kernel function update_chi_kernel!(chi_dest, chi, v, p, tensor, mask0, tau, resinv, dipole_kernel)
    I = @index(Global, Cartesian)
    R = @ndrange

    div = div_local_tensor((I, R), p, resinv, mask0, tensor)
    wave = filter_local((I, R), v, dipole_kernel, mask0)

    @inbounds chi_dest[I] = chi[I] + tau * (div - wave)
end

# Update w_dest <- w + tau*(mask*p + div(mask0*q)). 
@kernel function update_w_kernel!(w_dest, w, p, q, tensor, mask, mask0, tau, (r1, r2, r3))
    I = @index(Global, Cartesian)
    R = @ndrange

    @inbounds begin
        w_dest[1, I] = w[1, I]
        w_dest[2, I] = w[2, I]
        w_dest[3, I] = w[3, I]
        if mask[I]
            q123 = div_local((I, R), q, (r1, r1, r1), mask0, (1, 2, 3))
            q245 = div_local((I, R), q, (r2, r2, r2), mask0, (2, 4, 5))
            q356 = div_local((I, R), q, (r3, r3, r3), mask0, (3, 5, 6))
            (px, py, pz) = apply_tensor((p[1, I], p[2, I], p[3, I]), tensor, I)
            w_dest[1, I] += tau * (px + q123)
            w_dest[2, I] += tau * (py + q245)
            w_dest[3, I] += tau * (pz + q356)
        end
    end
end

function extragradient_update!(u_, u)
    u_ .= 2 .* u .- u_
end

@inline function stencil((I, (x, y, z)), A, mask)
    i, j, k = Tuple(I)
    @inbounds begin
        A0 = A[i, j, k]
        if isnothing(mask)
            A1m = (i > 1) ? A[i-1, j, k] : A0
            A1p = (i < x) ? A[i+1, j, k] : A0
            A2m = (j > 1) ? A[i, j-1, k] : A0
            A2p = (j < y) ? A[i, j+1, k] : A0
            A3m = (k > 1) ? A[i, j, k-1] : A0
            A3p = (k < z) ? A[i, j, k+1] : A0
        else # TODO test to switch mask to && (also in div_local)
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

@inline function filter_local(I, A, w, mask=nothing)
    A0, A1m, A1p, A2m, A2p, A3m, A3p = stencil(I, A, mask)

    (-2A0 + A1m + A1p) * w[1] +
    (-2A0 + A2m + A2p) * w[2] +
    (-2A0 + A3m + A3p) * w[3]
end

@inline function div_local((I, (x, y, z)), A::AbstractArray{T}, (r1, r2, r3), mask, (i1, i2, i3)=(1, 2, 3)) where {T}
    i, j, k = Tuple(I)
    @inbounds begin
        div = mask[i, j, k] * A[i1, i, j, k] * r1
        div -= (i > 1) ? mask[i-1, j, k] * A[i1, i-1, j, k] * r1 : zero(T)

        div += mask[i, j, k] * A[i2, i, j, k] * r2
        div -= (j > 1) ? mask[i, j-1, k] * A[i2, i, j-1, k] * r2 : zero(T)

        div += mask[i, j, k] * A[i3, i, j, k] * r3
        div -= (k > 1) ? mask[i, j, k-1] * A[i3, i, j, k-1] * r3 : zero(T)
    end
    return div
end

@inline function div_local_tensor((I, (x, y, z)), A::AbstractArray{T}, (r1, r2, r3), mask, tensor) where {T}
    @inbounds begin
        div = mask[I] * tensor_sum(A, tensor, (1, 2, 3), I) * r1
        ind = I - CartesianIndex(1, 0, 0)
        if I[1] > 1 && mask[ind]
            div -= tensor_sum(A, tensor, (1, 2, 3), ind)
        end

        div += mask[I] * tensor_sum(A, tensor, (2, 4, 5), I) * r2
        ind = I - CartesianIndex(0, 1, 0)
        if I[2] > 1 && mask[ind]
            div -= tensor_sum(A, tensor, (2, 4, 5), ind)
        end

        div += mask[I] * tensor_sum(A, tensor, (3, 5, 6), I) * r3
        ind = I - CartesianIndex(0, 0, 1)
        if I[3] > 1 && mask[ind]
            div -= tensor_sum(A, tensor, (3, 5, 6), ind)
        end
    end
    return div
end

@inline function tensor_sum(A, t, (i, j, k), I)
    sum((t[i, I], t[j, I], t[k, I]) .* (A[1, I], A[2, I], A[3, I]))
end

function symgrad_local((I, (x, y, z)), A::AbstractArray{T}, (r1, r2, r3)) where {T}
    i, j, k = Tuple(I)
    A1, A2, A3 = A[1, I], A[2, I], A[3, I]
    @inbounds begin
        if (i < x)
            wxx = (A[1, i+1, j, k] - A1) * r1
            wxy = (A[2, i+1, j, k] - A2) * r1 / 2
            wxz = (A[3, i+1, j, k] - A3) * r1 / 2
        else
            wxx = zero(T)
            wxy = zero(T)
            wxz = zero(T)
        end

        if (j < y)
            wxy += (A[1, i, j+1, k] - A1) * r2 / 2
            wyy = (A[2, i, j+1, k] - A2) * r2
            wyz = (A[3, i, j+1, k] - A3) * r2 / 2
        else
            wyy = zero(T)
            wyz = zero(T)
        end

        if (k < z)
            wxz += (A[1, i, j, k+1] - A1) * r3 / 2
            wyz += (A[2, i, j, k+1] - A2) * r3 / 2
            wzz = (A[3, i, j, k+1] - A3) * r3
        else
            wzz = zero(T)
        end
    end
    return wxx, wxy, wxz, wyy, wyz, wzz
end

@inline function grad_local((I, (x, y, z)), A::AbstractArray{T}, resinv) where {T}
    i, j, k = Tuple(I)
    A0 = A[i, j, k]
    dx = (i < x) ? (A[i+1, j, k] - A0) * resinv[1] : zero(T)
    dy = (j < y) ? (A[i, j+1, k] - A0) * resinv[2] : zero(T)
    dz = (k < z) ? (A[i, j, k+1] - A0) * resinv[3] : zero(T)
    return dx, dy, dz
end

@inline function apply_tensor(x, t, I)
    t123 = (t[1, I], t[2, I], t[3, I])
    t245 = (t[2, I], t[4, I], t[5, I])
    t356 = (t[3, I], t[5, I], t[6, I])
    return sum(t123 .* x), sum(t245 .* x), sum(t356 .* x)
end
