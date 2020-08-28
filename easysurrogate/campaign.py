"""
EasySurrogate campain class
"""
import pickle
from tkinter import filedialog
import tkinter as tk
import h5py

class Campaign:
    """
    The main EasySurrogate Campaign object
    """

    def __init__(self, load_data=False, load_state=False, **kwargs):
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

    def load_data(self, **kwargs):
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
            file_path = tk.filedialog.askopenfilename(title="Post processing: Open data file",
                                                      filetypes=(('HDF5 files', '*.hdf5'),
                                                                 ('All files', '*.*')))

        self.data_frame = h5py.File(file_path, 'r')
        self.file_path = file_path

        data_frame = h5py.File(file_path, 'r')
        print('Loaded', data_frame.keys())

        return data_frame
        
    def get_data_frame(self):
        """
        Returns the HDF5 data frame
        """
        return self.data_frame

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

        if 'file_path' in kwargs:
            file_path = kwargs['file_path']
        else:
            root = tk.Tk()
            root.withdraw()

            file_path = filedialog.askopenfilename(title="Open campaign state",
                                                   filetypes=(('pickle files', '*.pickle'),
                                                              ('All files', '*.*')))

        print('Loading state from', file_path)

        file = open(file_path, 'rb')
        state = pickle.load(file)
        file.close()

        for key in state:
            print('Loading:', key)
            vars(self)[key] = state[key]

    def save_state(self, state_vars, **kwargs):
        """
        Save the state of the current campaign to a pickle file

        Parameters
        ----------
        state_vars : (dict) a dictionary of the state variables

        **kwargs : if this contains file_path=<path_to_file>, save to this file
                   directly, otherwise open up a filedialog window
        Returns
        -------
        None.

        """

        if 'file_path' in kwargs:
            file = open(kwargs['file_path'], 'wb')
        else:
            root = tk.Tk()
            root.withdraw()

            file = filedialog.asksaveasfile(title="Save campaign state",
                                            mode='wb', defaultextension=".pickle")

        pickle.dump(state_vars, file)
        file.close()
