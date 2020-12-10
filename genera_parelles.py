#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author = Marcos Pérez
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.ttk as ttk
import multiprocessing as mp
from PIL import Image, ImageTk
from zernike_p import zernike_p, get_zernike_index
import imageio

# Import custom widgets and functions
from widgets import PanellDades, Visualitzador
import prop_funs as pr
fftshift = np.fft.fftshift

k = 2*np.pi

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
        self.E_in = pr.crea_camp(npix, r, m, k_noise=phase_error, misalign=misalign,
                cos_order=cos_order)
        self.E_in[:] = pr.afegeix_bruticia(self.E_in, 1, niter=10)
        self.P = pr.crea_pupila(npix, r_pupil, z_coeffs)
        #self.P[:] = afegeix_bruticia(self.P, 16, niter=bruticia)
        H1, H2, H3, t_lens = pr.propagadors(npix)
        self.E_out = pr.propaga_os(self.E_in, self.P, t_lens, H1, H2)

        # Calculate intensities and plot
        I = pr.captura_intensitat(self.E_out, dades[3], dades[4])
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
        self.process = mp.Process(target=pr.calcula_imatges, args=(npix, r, r_pupil, 
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


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("vista")
    except:
        style.theme_use("clam")
    app = GeneradorParelles(root)
    root.mainloop()
