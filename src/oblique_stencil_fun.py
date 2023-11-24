from numpy import *
from scipy.fft import dst, idst
from scipy.linalg import svd

# Idea:
# We know that the dipole kernel \kappa satisfies
# \laplace^{-1} \wave \delta = \kappa,
# so we find a stencil for \wave, i.e., \wave \delta
# by solving:
# \min_w ||\laplace^{-1} w - \kappa||^2
# over a suitable domain

def dipole(x, y, z, direction=(0, 0, 1)):
    r = sqrt(x*x + y*y + z*z)
    mask = (r > 0)
    xz = (direction[0]*x + direction[1]*y + direction[2]*z)/r
    kappa = 1/(4*pi)*(3*xz*xz - 1)/(r*r*r)
    return kappa, mask

def stencil(st=27, res=(1.0, 1.0, 1.0), direction=(0,0,1), gridsize=(128, 128, 128)):

    coord = [(array(range(N)) - N/2)*h for (N, h) in zip(gridsize, res)]

    print(f"Direction: ({direction[0]}, {direction[1]}, {direction[2]})")
    print(f"Resolution: ({res[0]}, {res[1]}, {res[2]})")
    X, Y, Z = meshgrid(*coord)
    d, d_mask = dipole(X, Y, Z, direction)

    # stencil mask
    if st == 19: # 19-point stencil
        mask = ones((3, 3, 3))
        mask[::2,::2,::2] = 0
    elif st == 27:
        mask = ones((3, 3, 3))
    else: # 7-point stencil
        mask = zeros((3, 3, 3))
        mask[1,1,:] = 1
        mask[1,:,1] = 1
        mask[:,1,1] = 1

    I, J, K = nonzero(mask)
    I += gridsize[0]//2 - 1 
    J += gridsize[1]//2 - 1 
    K += gridsize[2]//2 - 1 

    vdeltas = []
    coord2 = [2*sin(pi*(array(range(N)) + 1)/(2*(N+1))) for N in gridsize]
    coord2_grid = ((coord2[0]*coord2[0])[:,newaxis,newaxis]/(res[0]*res[0]) + (coord2[1]*coord2[1])[newaxis,:,newaxis]/(res[1]*res[1]) + (coord2[2]*coord2[2])[newaxis,newaxis,:]/(res[2]*res[2]))
    for i, j, k in zip(I, J, K):
        delta = zeros(gridsize)
        delta[i,j,k] = 1
        Fdelta = dst(dst(dst(delta, type=1, axis=0), type=1, axis=1), type=1, axis=2)
        Fdelta /= -coord2_grid
        vdelta = idst(idst(idst(Fdelta, type=1, axis=0), type=1, axis=1), type=1, axis=2)
        vdeltas.append(vdelta)

    A = array([v[d_mask].flatten() for v in vdeltas])
    U, s, V = svd(A, full_matrices=False)

    singular_value_threshold = 1e-10
    s_mask = s >= singular_value_threshold
    sinv = zeros_like(s)
    sinv[s_mask] = 1/s[s_mask]

    y = V @ d[d_mask].flatten()
    x = U @ (y*sinv)
    x_corr = x*res[0]*res[1]*res[2] # correction factor according to voxel size

    I, J, K = nonzero(mask)
    I = 1 - I
    J = 1 - J 
    K = 1 - K
    
    stencil = zeros((3, 3, 3))
    ind = 0
    for k in range(3):
        for j in range(3):
            for i in range(3):
                if mask[i,j,k]:
                    stencil[i,j,k] = x_corr[ind]
                    ind += 1
                else:
                    stencil[i,j,k] = 0


    Ax = zeros_like(d)
    for (vdelta, c) in zip(vdeltas, x):
        Ax += c*vdelta

    err = Ax - d
    rel_err = sum(err[d_mask]*err[d_mask])/sum(d_mask)
    print(f"Normalized error: {rel_err}")
    
    return stencil

def norm_sqr(dipole_kernel, res):
    grad_norm_sqr = 4.0*(sum(1.0/(res**2)))
    grad_norm = sqrt(grad_norm_sqr)
    wave_norm = sum(abs(dipole_kernel))
    norm_matrix = [[0, grad_norm, 1], [0, 0, grad_norm], [grad_norm_sqr, wave_norm, 0]]
    U, s, Vt = svd(norm_matrix)
    norm_sqr = s[0]*s[0]
    return norm_sqr

