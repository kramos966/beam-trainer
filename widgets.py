#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
import imageio
from zernike_p import zernike_p, get_zernike_index
from PIL import Image, ImageTk

osa_indexs = get_zernike_index()
k = 2*np.pi
folder = "training_set"

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

    def update_values(self, update_dict):
        """Update the values inside each entry with the ones inside update_dict"""
        zernikes = update_dict["zernikes"]
        beam_data = update_dict["beam_data"]
        self.beam_data.update_entries(beam_data)
        self.zernike_data.update_entries(zernikes)

class EntradesZernike(tk.Frame):
    def __init__(self, master):
        self.master = master
        tk.Frame.__init__(self, master)

        # Generate the Zernike polynomial entries based on OSA convention
        nrow = 7
        ncol = 2
        self.entries = dict()
        for i in range(nrow):
            for j in range(ncol):
                num = i*ncol+j+1
                e = myEntry(self, text=str(num), default="0.0", width=8)
                e.grid(row=i, column=j)
                self.entries[num] = e

    def dump(self):
        """Dump only the zernike coefficients with non zero value"""
        keys = self.entries.keys()
        val_dict = dict()
        for key in keys:
            val = float(self.entries[key].get())
            if val > 0:
                val_dict[key] = val
        return val_dict

    def update_entries(self, entry_dict):
        keys = entry_dict.keys()
        for key in keys:
            entry = self.entries[int(key)]
            entry.delete(0, tk.END)
            entry.insert(0, entry_dict[key])

class EntradesFeix(tk.Frame):
    def __init__(self, master, command=None):
        self.master = master
        tk.Frame.__init__(self, master)

        # Generate the entries for the parameters of the beam
        self.entries = dict()   # Empty dictionary to save each config
        entry_pix = myEntry(self, text="Lateral resolution", default="256",
                width=8)
        entry_pix.pack(anchor=tk.W)
        self.entries["npix"] = entry_pix

        entry_r = myEntry(self, text="Beam Radius", default="0.05", width=8)
        entry_r.pack(anchor=tk.W)
        self.entries["r"] = entry_r

        entry_pup = myEntry(self, text="Pupil radius", default="0.5", width=8)
        entry_pup.pack(anchor=tk.W)
        self.entries["r_pupil"] = entry_pup

        entry_np = myEntry(self, text="Max photons per pixel", default="200", 
                width=8)
        entry_np.pack(anchor=tk.W)
        self.entries["lam"] = entry_np
        
        entry_dn = myEntry(self, text="Std dark noise", default="10",
                width=8)
        entry_dn.pack(anchor=tk.W)
        self.entries["sigma"] = entry_dn

        entry_nim = myEntry(self, text="Number of images", default="100",
                width=8)
        entry_nim.pack(anchor=tk.W)
        self.entries["nim"] = entry_nim

        entry_an = myEntry(self, text="Relative phase error", default="0.0",
                width=8)
        entry_an.pack(anchor=tk.W)
        self.entries["std_phase"] = entry_an

        entry_ord = myEntry(self, text="Vortex order", default="0",
                width=8)
        entry_ord.pack(anchor=tk.W)
        self.entries["m"] = entry_ord

        entry_mis = myEntry(self, text="Relative Vortex misalign", default="0",
                width=8)
        entry_mis.pack(anchor=tk.W)
        self.entries["miss"] = entry_mis

        entry_sp = myEntry(self, text="Cosine order", default="0",
                width=8)
        entry_sp.pack(anchor=tk.W)
        self.entries["l"] = entry_sp

        self.commence = ttk.Button(self, text="Start computation", command=command)
        self.commence.pack(anchor=tk.W)

        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(anchor=tk.W)

    def dump(self):
        """Generate a dictionary containing the values of the entries."""
        val_dict = dict()
        for key in self.entries.keys():
            val_dict[key] = float(self.entries[key].get())
        return val_dict
    
    def start_progress(self):
        self.progress.start(16)

    def stop_progress(self):
        self.progress.stop()

    def update_entries(self, entry_dict):
        keys = entry_dict.keys()
        for key in keys:
            entry = self.entries[key]
            entry.delete(0, tk.END)
            entry.insert(0, entry_dict[key])

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

    def delete(self, begin, end):
        self.entry.delete(begin, end)

    def insert(self, begin, string):
        self.entry.insert(begin, string)

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
