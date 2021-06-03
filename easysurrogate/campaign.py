"""
EasySurrogate campaign class
"""

from tkinter import filedialog
import tkinter as tk
import pickle
import h5py
import numpy as np


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

    def load_easyvvuq_data(self, easyvvuq_campaign, qoi_cols):
        """
        Extract the input samples (features) and code samples (target data)
        from an EasyVVUQ dataframe.

        Parameters
        ----------
        easyvvuq_campaign : EasyVVUQ Campaign
            The EasyVVUQ campaign that was used to sample the input-output relation
            of a code.
        qoi_cols : A single string or a list of strings
            A list containing the names of the Quantities of Interest, matching
            the columns names in the EasyVVUQ data frame.

        Returns
        -------
        features : array, size (n_samples, n_inputs)
            An array of all input samples.
        samples : dict
            A dictionary with qoi_cols as keys. Each key contains an array of code
            output samples.

        """
        # if just a single string is provided, add to a list anyway
        if isinstance(qoi_cols, str):
            qoi_cols = [qoi_cols]
        # easyvuq vary object
        vary = easyvvuq_campaign._active_sampler.vary
        # number of inputs
        n_inputs = len(vary.get_keys())
        # number of code samples
        n_samples = easyvvuq_campaign._active_sampler.n_samples()
        # store the parameters in theta, will be used as features
        features = np.zeros([n_samples, n_inputs])
        # loop over all runs
        print("Extracting features %s" % (list(vary.get_keys())))
        for i, run in enumerate(easyvvuq_campaign.list_runs()):
            # get the paramater values
            values = run[1]['params']
            for j, param in enumerate(vary.get_keys()):
                # store input values as features
                features[i, j] = values[param]
        # Extract the QoIs from the EasyVVUQ data frame
        data_frame = easyvvuq_campaign.get_collation_result()
        samples = {k: [] for k in qoi_cols}
        # Loop over all code samples and qois
        print("Extracting output data %s " % (qoi_cols))
        for run_id in data_frame[('run_id', 0)].unique():
            for k in qoi_cols:
                values = data_frame.loc[data_frame[('run_id', 0)] == run_id][k].values
                samples[k].append(values.flatten())
        samples[k] = np.array(samples[k])
        # return features and code samples
        return features, samples

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
