#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author = Marcos Pérez
import numpy as np
import matplotlib.pyplot as plt
import imageio as io
import tkinter as tk
import tkinter.ttk as ttk
import pyfftw
import multiprocessing as mp
import os
from PIL import Image, ImageTk
from zernike_p import zernike_p, get_zernike_index
import imageio


osa_indexs = get_zernike_index()
k = 2*np.pi
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
        for i, coeff in enumerate(aberrations):
            if coeff != 0.0:    # Calcula si es necessari
                indexs = osa_indexs[i+1]
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

class GeneradorParelles:
    """Interface for generating multiple pairs of intensity-phase values
    for neural network training or other purposes."""
    def __init__(self, master):
        self.master = master
        self.master.title("Generador Parelles")
        # Create interface
        self.dades = PanellDades(self.master, command=self.calcula_series)
        self.visualitzador = Visualitzador(self.master, width=512, height=512)

        self.dades.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.visualitzador.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Generate test field and plot it
        self.act_mostra()

        self.master.bind("<Return>", self.act_mostra)

    def act_mostra(self, event=None):
        dades = self.dades.get_beam()
        z_coeffs = self.dades.get_zernikes()
        #npix = int(dades[0])
        npix = 512      # Fix, nomes es camp de mostra!
        r = dades[1]
        r_pupil = dades[2]
        phase_error = dades[6]
        m = int(dades[7])    # Vortex order
        misalign = dades[8]
        cos_order = int(dades[9])
        
        # Create input and output fields
        self.E_in = crea_camp(npix, r, m, k_noise=phase_error, misalign=misalign,
                cos_order=cos_order)
        self.P = crea_pupila(npix, r_pupil, z_coeffs)
        #self.P[:] = afegeix_bruticia(self.P, 16, niter=bruticia)
        self.E_out = convoluciona(self.E_in, self.P)

        # Calculate intensities and plot
        I = captura_intensitat(self.E_out, dades[3], dades[4])
        I_in = np.real(np.conj(self.E_in)*self.E_in)
        self.visualitzador.imshow(0, I_in)
        self.visualitzador.imshow(1, I)
        self.visualitzador.imshow(2, np.real(self.P))

    def calcula_series(self, event=None):
        # Obtain important data
        dades = self.dades.get_beam()
        z_coeffs = self.dades.get_zernikes()
        npix = int(dades[0])
        r = dades[1]
        r_pupil = dades[2]
        max_photons = dades[3]
        sigma = dades[4]
        nimatges = int(dades[5])
        phase_error = dades[6]
        m = int(dades[7])
        misalign = dades[8]
        cos_order = int(dades[9])

        # Call process
        self.process = mp.Process(target=calcula_imatges, args=(npix, r, r_pupil, 
            max_photons, sigma, phase_error, m, misalign, cos_order, z_coeffs, nimatges))
        # Start progressbar and process
        self.process.start()
        self.dades.start_progress()
        # Check status
        self.comprova_fi()

    def comprova_fi(self, event=None):
        if self.process.exitcode is None:
            self.master.after(100, self.comprova_fi)
        else:
            self.dades.stop_progress()
            self.process.join()

class PanellDades(ttk.Notebook):
    def __init__(self, master, command=None):
        self.master = master
        ttk.Notebook.__init__(self, master)

        # Beam data
        self.beam_data = EntradesFeix(self.master, command=command)
        self.add(self.beam_data, text="Beam data")

        # Zernike data
        self.zernike_data = EntradesZernike(self.master)
        self.add(self.zernike_data, text="Zernike coeffs")

    def get_beam(self):
        return self.beam_data.dump()

    def get_zernikes(self):
        return self.zernike_data.dump()

    def start_progress(self):
        self.beam_data.start_progress()

    def stop_progress(self):
        self.beam_data.stop_progress()

class EntradesZernike(tk.Frame):
    def __init__(self, master):
        self.master = master
        tk.Frame.__init__(self, master)

        # Generate the Zernike polynomial entries based on OSA convention
        nrow = 7
        ncol = 2
        self.entries = []
        for i in range(nrow):
            for j in range(ncol):
                e = myEntry(self, text=f"{i*ncol+j+1}", default="0.0", width=8)
                e.grid(row=i, column=j)
                self.entries.append(e)

    def dump(self):
        return [float(coeff.get()) for coeff in self.entries]

class EntradesFeix(tk.Frame):
    def __init__(self, master, command=None):
        self.master = master
        tk.Frame.__init__(self, master)

        # Generate the entries for the parameters of the beam
        self.entries = []
        entry_pix = myEntry(self, text="Lateral resolution", default="256",
                width=8)
        entry_pix.pack(anchor=tk.W)
        self.entries.append(entry_pix)

        entry_r = myEntry(self, text="Beam Radius", default="0.05", width=8)
        entry_r.pack(anchor=tk.W)
        self.entries.append(entry_r)

        entry_pup = myEntry(self, text="Pupil radius", default="0.05", width=8)
        entry_pup.pack(anchor=tk.W)
        self.entries.append(entry_pup)

        entry_np = myEntry(self, text="Max photons per pixel", default="500", 
                width=8)
        entry_np.pack(anchor=tk.W)
        self.entries.append(entry_np)
        
        entry_dn = myEntry(self, text="Std dark noise", default="20",
                width=8)
        entry_dn.pack(anchor=tk.W)
        self.entries.append(entry_dn)

        entry_nim = myEntry(self, text="Number of images", default="100",
                width=8)
        entry_nim.pack(anchor=tk.W)
        self.entries.append(entry_nim)

        # Buttons and checks for the final image calculations
        #self.check_xy = tk.Checkbutton(self, text="Alternate radial images")
        #self.check_xy.pack(anchor=tk.W)
        #self.check_rab = tk.Checkbutton(self, text="Random Zernike coeff")
        #self.check_rab.pack(anchor=tk.W)

        entry_an = myEntry(self, text="Relative phase error", default="0.0",
                width=8)
        entry_an.pack(anchor=tk.W)
        self.entries.append(entry_an)

        entry_ord = myEntry(self, text="Vortex order", default="0",
                width=8)
        entry_ord.pack(anchor=tk.W)
        self.entries.append(entry_ord)

        entry_mis = myEntry(self, text="Relative Vortex misalign", default="0",
                width=8)
        entry_mis.pack(anchor=tk.W)
        self.entries.append(entry_mis)

        entry_sp = myEntry(self, text="Cosine order", default="0",
                width=8)
        entry_sp.pack(anchor=tk.W)
        self.entries.append(entry_sp)

        self.commence = ttk.Button(self, text="Start computation", command=command)
        self.commence.pack(anchor=tk.W)

        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(anchor=tk.W)

    def dump(self):
        return [float(x.get()) for x in self.entries]
    
    def start_progress(self):
        self.progress.start(16)

    def stop_progress(self):
        self.progress.stop()

class myEntry(tk.Frame):
    def __init__(self, master, text="", default="0.5", width=None):
        self.master = master
        tk.Frame.__init__(self, master)
        
        # Create the label and the entry
        self.label = ttk.Label(self, text=text)
        self.entry = ttk.Entry(self, width=width)
        self.entry.delete(0, tk.END)
        self.entry.insert(0, default)

        self.label.pack(side=tk.TOP, anchor=tk.W)
        self.entry.pack(side=tk.BOTTOM, anchor=tk.W)

    def get(self):
        return self.entry.get()

class Plotter(tk.Canvas):
    def __init__(self, master, width=256, height=256):
        self.master = master
        self.w = width
        self.h = height
        tk.Canvas.__init__(self, master, width=width, height=height)

    def imshow(self, im):
        """Transform an image to the desired range and plot it."""
        self.impil = Image.fromarray(np.uint8(im/im.max()*255))    # Save a reference
        self.imtk = ImageTk.PhotoImage(self.impil)
        # Clear canvas
        self.delete("all")
        # Paint image
        self.create_image((0, 0), anchor=tk.NW, image=self.imtk)

class Visualitzador(ttk.Notebook):
    def __init__(self, master, width=None, height=None):
        self.master = master
        ttk.Notebook.__init__(self, master)

        self.plots = []
        plot_in = Plotter(self, width=width, height=height)
        self.add(plot_in, text="Input beam")
        self.plots.append(plot_in)

        plot_I = Plotter(self, width=width, height=height)
        self.add(plot_I, text="Recorded Intensity")
        self.plots.append(plot_I)

        plot_P = Plotter(self, width=width, height=height)
        self.add(plot_P, text="Pupil")
        self.plots.append(plot_P)

    def imshow(self, n, im):
        self.plots[n].imshow(im)

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("vista")
    except:
        style.theme_use("clam")
    app = GeneradorParelles(root)
    root.mainloop()
