#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from zernike_p import zernike_p, get_zernike_index
import os

k = 2*np.pi
osa_indexs = get_zernike_index()
folder = "training_set"

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
    return E

def convoluciona(E, P):
    A = np.fft.fft2(E)
    return np.fft.fftshift(np.fft.ifft2(E*P))

def calcula_imatges(npix, r, rp, max_photons, sigma, phase_error,
        m, misalign, cos_order, aberracions, nimatges):
    """Function to calculate the pairs of Intensity-Phase values for random
    values of the amplitude and phase of an electric field."""
    # Try to create folder
    try:
        os.mkdir(folder)
    except:
        pass

    P, rho, phi = crea_pupila(npix, r, work=True)
    # Get only the aberrations with non zero max coefficient
    zernikes = []
    for i, coeff in enumerate(aberracions):
        if coeff != 0.0:
            zernikes.append((osa_indexs[i+1], coeff))

    W = np.zeros((npix, npix), dtype=np.float_)
    for i in range(nimatges):
        r_i = np.random.rand()*r
        order = np.random.randint(cos_order+1)
        E_in = crea_camp(npix, r_i, m, misalign=misalign, k_noise=phase_error,
                cos_order=order)
        # Compute random wavefront aberration
        if zernikes:
            W[:] = 0
            for k in range(len(zernikes)):
                # Maximum amplitude of the coefficient multiplied by the polynomial
                (n, m), A = zernikes[k]
                W[:] += A*np.random.rand()*\
                        zernike_p(rho, phi, n, m)
            P_ab = P*np.exp(1j*k*W)
        else:
            P_ab = P

        # Compute the final field
        E_out = convoluciona(E_in, P_ab)
        phi_out = np.angle(E_out)
        I_out = captura_intensitat(E_out, max_photons, sigma)

        # Save data
        path = os.path.join(folder, f"{i}.npz")
        np.savez(path, intensity=I_out, phase=phi_out)

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
    if Imax > 0:
        I[:] = I/I.max()*mean_photon
    dark_noise = np.random.normal(scale=std_dark, size=E.shape)
    I_poiss = np.random.poisson(lam=I, size=E.shape)

    return I+I_poiss+np.abs(dark_noise)

