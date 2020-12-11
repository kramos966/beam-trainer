#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author = Marcos Pérez
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.ttk as ttk
import multiprocessing as mp
import json
import imageio
import os
import time
from PIL import Image, ImageTk

# Import custom widgets and functions
from zernike_p import zernike_p, get_zernike_index
from widgets import PanellDades, Visualitzador
import prop_funs as pr


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

        # Generate the menu
        bmenu = tk.Menu(self.master)
        filemenu = tk.Menu(bmenu)
        filemenu.add_command(label="Load configuration", command=self.loadconfig)
        filemenu.add_command(label="Save configuration", command=self.saveconfig)
        bmenu.add_cascade(label="File", menu=filemenu)

        self.master.config(menu=bmenu)

        # Generate test field and plot it
        self.act_mostra()

        self.master.bind("<Return>", self.act_mostra)

    def act_mostra(self, event=None):
        dades = self.dades.get_beam()
        z_coeffs = self.dades.get_zernikes()
        #npix = int(dades[0])
        npix = 512      # Fix, nomes es camp de mostra!
        r = dades["r"]
        r_pupil = dades["r_pupil"]
        phase_error = dades["std_phase"]
        m = int(dades["m"])    # Vortex order
        misalign = dades["miss"]
        cos_order = int(dades["l"])
        
        # Create input and output fields
        self.E_in = pr.crea_camp(npix, r, m, k_noise=phase_error, misalign=misalign,
                cos_order=cos_order)
        self.P = pr.crea_pupila(npix, r_pupil, z_coeffs)
        #self.P[:] = afegeix_bruticia(self.P, 16, niter=bruticia)
        self.E_out = pr.convoluciona(self.E_in, self.P)

        # Calculate intensities and plot
        I = pr.captura_intensitat(self.E_out, dades["lam"], dades["sigma"])
        I_in = np.real(np.conj(self.E_in)*self.E_in)
        self.visualitzador.imshow(0, I_in)
        self.visualitzador.imshow(1, I)
        self.visualitzador.imshow(2, np.real(self.P))

    def calcula_series(self, event=None):
        # Select save folder
        foldname = filedialog.askdirectory(title="Select save directory")
        pr.folder = foldname
        try:
            os.mkdir(foldname)
        except:
            pass
        # Dump configuration to the folder
        t = time.localtime()
        name = f"conf_{t[0]}{t[1]}{t[2]}{t[3]}{t[4]}.json"
        path = os.path.join(foldname, name)
        self.saveconfig(name=path)
        # Obtain important data
        dades = self.dades.get_beam()
        z_coeffs = self.dades.get_zernikes()
        npix = int(dades["npix"])
        r = dades["r"]
        r_pupil = dades["r_pupil"]
        max_photons = dades["lam"]
        sigma = dades["sigma"]
        nimatges = int(dades["nim"])
        phase_error = dades["std_phase"]
        m = int(dades["m"])
        misalign = dades["miss"]
        cos_order = int(dades["l"])

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

    def loadconfig(self, event=None, ask=False, name=None):
        # Ask file
        if ask:
            name = filedialog.askopenfilename(title="Load configuration file",
                    filetypes=(("JSON", "*.json"), ("All", "*.*")))
        with open(name, "r") as openfile:
            joint_dict = json.load(openfile, parse_int=True)

        # Reflect the loaded values in each entry
        self.dades.update_values(joint_dict)

    def saveconfig(self, event=None, ask=False, name=None):
        """Save current values of the configuration into a JSON file."""
        joint_dict = dict()
        zernikes = self.dades.get_zernikes()
        dades = self.dades.get_beam()
        joint_dict["beam_data"] = dades
        joint_dict["zernikes"] = zernikes
        # Save finally the joint dictionary
        if ask:
            name = filedialog.asksaveasfilename(title="Save configuration file",
                    filetypes=(("JSON", "*.json"), ("All", "*.*")))
        if not name.endswith(".json"):
            name = f"{name}.json"
        with open(name, "w") as savefile:
            json.dump(joint_dict, savefile, sort_keys=True, indent=4)

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("vista")
    except:
        style.theme_use("clam")
    app = GeneradorParelles(root)
    root.mainloop()
