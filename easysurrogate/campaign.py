"""
EasySurrogate campaign class
"""

from tkinter import filedialog
import tkinter as tk
import pickle
import h5py


class Campaign:
    """
    The main EasySurrogate Campaign object
    """

    def __init__(self, name=None, load_state=False, **kwargs):
        """
        Create a Campaign object.

        Parameters
        ----------
        name : string, optional
            The name of this object. The default is None.
        load_state : boolean, optional
            Load a campaign state from file. The default is False.

        Returns
        -------
        None.

        """

        if load_state:
            self.load_state(**kwargs)

        if name is None:
            self.campaign_name = 'EasySurrogate_Campaign'
        else:
            self.campaign_name = name

        self.accum_data = {}

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

        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = 'Load training data'

        if 'file_path' in kwargs:
            file_path = kwargs['file_path']
        else:
            root = tk.Tk()
            root.withdraw()
            file_path = tk.filedialog.askopenfilename(title=name,
                                                      filetypes=(('HDF5 files', '*.hdf5'),
                                                                 ('All files', '*.*')))
        # load HDF5 data
        h5f = h5py.File(file_path, 'r')
        data_frame = {}

        # convert HDF5 data to a dictionary
        for key in h5f.keys():
            print('Loaded %s' % key)
            data_frame[key] = h5f[key][()]

        return data_frame

    def store_data_to_hdf5(self, data, **kwargs):
        """
        Store data to HDF5 format

        Parameters
        ----------
        data : dict
            data stored in a dictionary or h5py file object.

        file_path : sting (optional)
            A path to the file. If unspecified a file dialog is opened.

        Returns
        -------
        None.

        """

        if not isinstance(data, dict) and not isinstance(data, h5py.File):
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
        # create HDF5 file
        h5f = h5py.File(file, 'w')

        # store numpy sample arrays as individual datasets in the hdf5 file
        for name in data.keys():
            h5f.create_dataset(name, data=data[name])

        h5f.close()
        file.close()
        print('done')

    def accumulate_data(self, data, **kwargs):
        """
        Accumulate data over a number of (time) steps. Data is stored in
        a list in the self.accum_data dictionary, under the same key as
        given in the 'data' argument.

        Parameters
        ----------
        data : dict
            The key-worded data that is accumulated over a number of (time) steps.
        kwargs : keyword arguments
            If kwargs contains 'names', which is a list of keyword names, then
            the data is stored under the specified names rather than the names
            given by data.keys().

        Returns
        -------
        None.

        """

        if not isinstance(data, dict):
            print('Data must be stored in a dict.')
            return

        if 'names' in kwargs:
            names = kwargs['names']
        else:
            names = list(data.keys())
        idx = 0
        for key in data.keys():
            if not names[idx] in self.accum_data:
                self.accum_data[names[idx]] = []
            self.accum_data[names[idx]].append(data[key])
            idx += 1

    def store_accumulated_data(self, **kwargs):
        """
        Stores the data accumulated by the 'accumulate_data' subroutine to
        a HDF5 file.

        Returns
        -------
        None.

        """
        self.store_data_to_hdf5(self.accum_data, **kwargs)

    def add_app(self, name=None, surrogate=None):
        """
        Add an application.

        Parameters
        ----------
        name : string (optional)
            Name of the EasySurrogate campaign. If unspecified the self.campaign_name is used.
        surrogate : object
            A surrogate object.

        Returns
        -------
        None.

        """
        if name is None:
            name = self.campaign_name

        self.campaign_name = name
        self.surrogate = surrogate

    def add_lags(self, name=None, lags=None):
        """
        Add an application.

        Parameters
        ----------
        name : string (optional)
            Name of the EasySurrogate campaign. If unspecified the self.campaign_name is used.
        lags : integer(s)
            Lags used in the training process.

        Returns
        -------
        None.

        """
        if name is None:
            name = self.campaign_name

        self.campaign_name = name
        self.lags = lags

    def add_scalers(self, name=None, scaler_features=None, scaler_target=None):
        """
        Add features and target scalers.

        Parameters
        ----------
        name : string (optional)
            Name of the EasySurrogate campaign. If unspecified the self.campaign_name is used.
        scaler_features : object
            A scaler object used in the training process for the features.
        scaler_target : object
            A scaler object used in the training process for the target.

        Returns
        -------
        None.

        """
        if name is None:
            name = self.campaign_name

        self.campaign_name = name
        self.scaler_features = scaler_features
        self.scaler_target = scaler_target

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

            file_path = filedialog.askopenfilename(title="Load state %s" % name,
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
        state: dict or None
            If None, store the state of the Campaign, not including the training data. This
            subroutine is inherited by the Surrogate classes to store their state.

        **kwargs :
            Can contain file_path=<path_to_file>, to save to this file  directly,
            otherwise open up a filedialog window.

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

            file = filedialog.asksaveasfile(title="Save state %s" % name,
                                            mode='wb',
                                            defaultextension=".pickle")

        print('Saving state to %s.' % file.name)
        pickle.dump(state, file)
        file.close()
