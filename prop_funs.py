#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from zernike_p import zernike_p, get_zernike_index
import os
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift

k = 2*np.pi
osa_indexs = get_zernike_index()
folder = "training_set"
lamb = 525e-6   # mm, wavelength
L =  2/lamb     # lambdas, half window width
f = 100/lamb    # lambdas, focal length of the optical system
z =  0/lamb

def crea_camp(npix, r, m, k_noise=2, cos_order=0, misalign=0):
    E = np.zeros((npix, npix), dtype=np.complex_)
    rp = r*npix
    k = .5/rp/rp
    y, x = np.mgrid[-npix//2:npix//2, -npix//2:npix//2]

    E[:] = np.exp(-k*(x*x+y*y))
    if m > 0:
        if misalign:
            dx = np.random.normal(scale=misalign)*misalign*npix
            dy = np.random.normal(scale=misalign)*misalign*npix
            y = y+dy
            x = x+dx
        phi = np.arctan2(y, x)
        W = m*phi+np.random.normal(scale=k_noise, size=(npix, npix))

        E[:] *= np.exp(1j*W)
    if cos_order:
        if m > 0:
            pass
        elif misalign:
            dx = np.random.normal(scale=misalign)*misalign*npix
            dy = np.random.normal(scale=misalign)*misalign*npix
            y = y+dy
            x = x+dx
        phi = np.arctan2(y, x)
        #E[:] = afegeix_bruticia(E, 2, niter=bruticia)
        E[:] = np.cos(cos_order*phi)*E
    return E/abs(E).max()

def convoluciona(E, P):
    A = fft2(E)
    return fftshift(ifft2(E*P))

def propaga_h(E, H):
    A = fft2(E)
    E_o = ifft2(A*H)
    return E_o

def propagadors(npix):
    y, x = np.mgrid[-npix//2:npix//2, -npix//2:npix//2]
    umax = npix/4/L
    v = y/npix*umax*2
    v[:] = fftshift(v)
    u = x/npix*umax*2
    u[:] = fftshift(u)
    u2 = u*u+v*v
    xx = x/npix*L*2
    yy = y/npix*L*2
    H1 = np.exp(-1j*np.pi*f*u2)         # Prop lens plane
    #H1[:] = fftshift(H1)
    H2 = np.exp(-1j*np.pi*(f-z)*u2)     # Prop intermediate plane
    #H2[:] = fftshift(H2)
    H3 = np.exp(-1j*np.pi*z*u2)         # Prop detector plane
    #H3[:] = fftshift(H3)
    const = np.pi/f                     # Transmission factor lens
    t_lens = np.exp(-1j*const*(xx*xx+yy*yy))
    return H1, H2, H3, t_lens, v, u

def propaga_os(E, P, t_lens, *tf):
    """Propagate the field E through an optical system using the
    varyadic list of tf of transfer functions."""
    if len(tf) == 1:
        # Propagate to intermediate plane
        # TODO: Include backpropagation to add noise
        H = tf[0]
        E_in = E*P
        return fftshift(fft2(E_in)) 
    if len(tf) == 2:
        H1, H2 = tf
        E_p = propaga_h(E, H1)
        E_p[:] = E_p*P*t_lens
        E_out = propaga_h(E_p, H2)
    elif len(tf) == 3:
        H1, H2, H3 = tf
        E_p = propaga_h(E, H1)
        E_p[:] = E_p*P*t_lens
        E_half = propaga_h(E_p, H2)
        #E_half[:] = afegeix_bruticia(E_half, 1, niter=10)
        E_out = propaga_h(E_half, H3)
    return E_out

def calcula_imatges(npix, r, rp, max_photons, sigma, phase_error,
        m, misalign, cos_order, aberracions, nimatges):
    """Function to calculate the pairs of Intensity-Phase values for random
    values of the amplitude and phase of an electric field."""
    # Try to create folder
    try:
        os.mkdir(folder)
    except:
        pass

    P, rho, phi = crea_pupila(npix, rp, work=True)
    # Get only the aberrations with non zero max coefficient
    if aberracions:
        coeff_num = aberracions.keys()
    W = np.zeros((npix, npix), dtype=np.float_)
    H1, H2, H3, t_lens, v, u = propagadors(npix)
    mask = rho < rp*rho.shape[0]
    rho[:] = rho*mask
    rho[:] = rho/rho.max()
    for i in range(nimatges):
        r_i = (np.random.rand()+1e-3)*r
        #r_i = r
        order = np.random.randint(cos_order+1)
        m_vortex = np.random.randint(m+1)
        #order = cos_order
        E_in = crea_camp(npix, r_i, m_vortex, misalign=misalign, k_noise=phase_error,
                cos_order=order)
        # Compute random wavefront aberration
        if aberracions:
            W[:] = 0
            for n_coeff in coeff_num:
                # Maximum amplitude of the coefficient multiplied by the polynomial
                A = aberracions[n_coeff]
                n, m = osa_indexs[n_coeff]
                W[:] += A*np.random.rand()*zernike_p(rho, phi, n, m)
            P_ab = P*np.exp(1j*k*W)
        else:
            P_ab = P

        # Compute the final field
        #if z > 0:
        #    E_out = propaga_os(E_in, P, t_lens, H1, H2, H3)
        #else:
        #    E_out = propaga_os(E_in, P, t_lens, H1, H2)
        E_out = propaga_os(E_in, P_ab, t_lens, H3)
        E_out /= abs(E_out).max()
        #phi_out = np.angle(E_out)
        I_out = captura_intensitat(E_out, max_photons, sigma)

        # Compute the z component
        Ek = fft2(E_out)
        Ez = ifft2(Ek*u)
        I_z = np.real(np.conj(Ez)*Ez)

        # Save data
        path = os.path.join(folder, f"{i:08d}.npz")
        np.savez(path, I_x=np.floor(I_out), I_z=I_z)

def afegeix_bruticia(E, r_brut, niter=10):
    ny, nx = E.shape
    W = np.zeros((ny, nx), dtype=np.complex_)
    y, x = np.mgrid[-r_brut:r_brut+1, -r_brut:r_brut+1]
    # Phase variation
    rho2 = x*x+y*y
    phi = -k/2/r_brut*rho2 * (rho2<r_brut*r_brut)

    # Select random centroid and plot the complex phase
    i = 0
    while i<niter:
        cy, cx = np.random.randint(ny, size=2)
        if (cx<r_brut) or (cy<r_brut) or (cy>=ny-r_brut) or (cx>=nx-r_brut):
            continue
        W[cy-r_brut:cy+r_brut+1, cx-r_brut:cx+r_brut+1] += phi*\
                1
                #(2*np.random.randint(2)-1)
        i += 1
    t = np.exp(1j*W)
    return E*t

def crea_pupila(npix, r, aberrations=None, work=False):
    P = np.zeros((npix, npix), dtype=np.complex_)
    rp = r*npix
    y, x = np.mgrid[-npix//2:npix//2, -npix//2:npix//2]
    rho2 = x*x+y*y
    mask = rho2 < rp**2
    P[:] = np.complex_(mask)
    if aberrations:
        phi = np.arctan2(y, x)
        rho = np.sqrt(rho2)
        rhomax = rho[mask].max()
        rho[:] = mask*rho/rhomax    # Radi normalitzat!
        W = np.zeros((npix, npix), dtype=np.float_)
        for coeff_num in aberrations:
            indexs = osa_indexs[coeff_num]
            coeff = aberrations[coeff_num]
            W[:] += coeff*zernike_p(rho, phi, *indexs)
        P[:] *= np.exp(1j*k*W)

    #P[:] = np.fft.fftshift(P)
    if work:
        rho = np.sqrt(rho2)
        phi = np.arctan2(y, x)
        #rho[:] = np.fft.fftshift(rho)
        #phi[:] = np.fft.fftshift(phi)
        return P, rho, phi
    else:
        return P

def captura_intensitat(E, mean_photon, std_dark):
    I = np.real(np.conj(E)*E)
    Imax = I.max()
    if np.isnan(Imax):
        import matplotlib.pyplot as plt
        plt.imshow(I)
        plt.show()
    if Imax > 0:
        I[:] = I/I.max()*mean_photon
    dark_noise = np.random.normal(scale=std_dark, size=E.shape)
    I_poiss = np.random.poisson(lam=I, size=E.shape)

    return I+I_poiss+np.abs(dark_noise)

def set_constants(data_dict):
    # FIXME: DIRTY HACK
    global lamb, L, f, z
    lamb = data_dict["lamb"]
    L = data_dict["length"]/lamb
    f = data_dict["focal"]/lamb
    z = data_dict["z"]/lamb
