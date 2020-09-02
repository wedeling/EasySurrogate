"""
EasySurrogate campaign class
"""
import pickle
import copy
import numpy as np
from tkinter import filedialog
import tkinter as tk
import h5py

class Campaign:
    """
    The main EasySurrogate Campaign object
    """

    def __init__(self, name = None, load_data=False, 
                 load_state=False, **kwargs):
        """
        Create a Campaign object

        Parameters
        ----------
        load_data (boolean): flag for loading training data

        load_state (boolean): flag for loading the campaign state
        """
        if load_data:
            self.load_data(**kwargs)

        if load_state:
            self.load_state(**kwargs)

        if name is None:
            self.campaign_name = 'EasySurrogate_Campaign'
        else:
            self.campaign_name = name

    def load_hdf5_data(self, **kwargs):
        """
        Load training data from HDF5 file

        Parameters
        ----------
        **kwargs : if this contains file_path=<path_to_file>, load this file
        directly, otherwise open up a filedialog window

        Returns
        -------
        None.

        """
        if 'file_path' in kwargs:
            file_path = kwargs['file_path']
        else:
            root = tk.Tk()
            root.withdraw()
            file_path = tk.filedialog.askopenfilename(title="Load training data",
                                                      filetypes=(('HDF5 files', '*.hdf5'),
                                                                 ('All files', '*.*')))
        #load HDF5 data
        h5f = h5py.File(file_path, 'r')
        data_frame = {}

        #convert HDF5 data to a dictionary
        for key in h5f.keys():
            print('Loaded %s' % key)
            data_frame[key] = h5f[key][()]

        return data_frame
    
    def store_data_to_hdf5(self, data, **kwargs):
        """
        Store data to HDF5 format

        Parameters
        ----------
        + data : data stored in a dictionary or h5py file object.

        + file_path : (optional) a path to the file. If unspecified a file dialog
        is opened

        Returns
        -------
        None.

        """

        if type(data) != dict and type(data) != h5py.File:
            print('Data must be stored in a dict or h5py File object.')
            return

        if 'file_path' in kwargs:
            file_path = kwargs['file_path']
            file = open(file_path, 'wb')
        else:
            root = tk.Tk()
            root.withdraw()
            file = filedialog.asksaveasfile(title="Store data in HDF5 file",
                                            mode='wb', defaultextension=".hdf5")
        print('Saving data in', file.name) 
        #create HDF5 file
        h5f = h5py.File(file, 'w')

        #store numpy sample arrays as individual datasets in the hdf5 file
        for name in data.keys():
            h5f.create_dataset(name, data = data[name])

        h5f.close()
        file.close()
        print('done')

    def add_app(self, name=None, surrogate=None):
        """
        Add an application 

        Parameters
        ----------
        name : name of the EasySurrogate campaign,
        surrogate : a surrogate object

        Returns
        -------
        None.

        """
        if name is None:
            name = self.campaign_name

        self.campaign_name = name
        self.surrogate = surrogate

    def load_state(self, **kwargs):
        """
        Loads the state of the current campaign from a pickle file

        Parameters
        ----------
        **kwargs : if this contains file_path=<path_to_file>, load this file
                   directly, otherwise open up a filedialog window
        Returns
        -------
        None.

        """

        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = ''

        if 'file_path' in kwargs:
            file_path = kwargs['file_path']
        else:
            root = tk.Tk()
            root.withdraw()

            file_path = filedialog.askopenfilename(title="Load state %s" %name,
                                                   filetypes=(('pickle files', '*.pickle'),
                                                              ('All files', '*.*')))

        print('Loading state from', file_path)

        file = open(file_path, 'rb')
        state = pickle.load(file)
        file.close()

        self.__dict__ = state

    def save_state(self, state=None, **kwargs):
        """
        Save the state of the current campaign to a pickle file

        Parameters
        ----------
        state, default is None : If None, store the state of the Campaign, not
        including the training data, which is stored separately.        

        **kwargs : can contain file_path=<path_to_file>,  to save to this file
                   directly, otherwise open up a filedialog window.
        Returns
        -------
        None.

        """

        if state is None:
            state = self.__dict__

        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = self.campaign_name

        if 'file_path' in kwargs:
            file = open(kwargs['file_path'], 'wb')
        else:
            root = tk.Tk()
            root.withdraw()

            file = filedialog.asksaveasfile(title="Save state %s" %name,
                                            mode='wb', 
                                            defaultextension=".pickle")

        pickle.dump(state, file)
        file.close()
