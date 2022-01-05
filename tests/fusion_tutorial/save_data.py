import easyvvuq as uq
import numpy as np
import chaospy as cp
import pandas as pd
import time
import os


def write_template(params):
    str = ""
    first = True
    for k in params.keys():
        if first:
            str += '{"%s": "$%s"' % (k, k)
            first = False
        else:
            str += ', "%s": "$%s"' % (k, k)
    str += '}'
    print(str, file=open('fusion.template', 'w'))


def define_params():
    return {
        "Qe_tot": {"type": "float", "min": 1.0e6, "max": 50.0e6, "default": 2e6},
        "H0": {"type": "float", "min": 0.00, "max": 1.0, "default": 0},
        "Hw": {"type": "float", "min": 0.01, "max": 100.0, "default": 0.1},
        "Te_bc": {"type": "float", "min": 10.0, "max": 1000.0, "default": 100},
        "chi": {"type": "float", "min": 0.01, "max": 100.0, "default": 1},
        "a0": {"type": "float", "min": 0.2, "max": 10.0, "default": 1},
        "R0": {"type": "float", "min": 0.5, "max": 20.0, "default": 3},
        "E0": {"type": "float", "min": 1.0, "max": 10.0, "default": 1.5},
        "b_pos": {"type": "float", "min": 0.95, "max": 0.99, "default": 0.98},
        "b_height": {"type": "float", "min": 3e19, "max": 10e19, "default": 6e19},
        "b_sol": {"type": "float", "min": 2e18, "max": 3e19, "default": 2e19},
        "b_width": {"type": "float", "min": 0.005, "max": 0.025, "default": 0.01},
        "b_slope": {"type": "float", "min": 0.0, "max": 0.05, "default": 0.01},
        "nr": {"type": "integer", "min": 10, "max": 1000, "default": 100},
        "dt": {"type": "float", "min": 1e-3, "max": 1e3, "default": 100},
        "out_file": {"type": "string", "default": "output.csv"}
    }


def define_vary():
    vary_all = {
        "Qe_tot": cp.Uniform(1.8e6, 2.2e6),
        "H0": cp.Uniform(0.0, 0.2),
        "Hw": cp.Uniform(0.1, 0.5),
        "chi": cp.Uniform(0.8, 1.2),
        "Te_bc": cp.Uniform(80.0, 120.0),
        "a0": cp.Uniform(0.9, 1.1),
        "R0": cp.Uniform(2.7, 3.3),
        "E0": cp.Uniform(1.4, 1.6),
        "b_pos": cp.Uniform(0.95, 0.99),
        "b_height": cp.Uniform(5e19, 7e19),
        "b_sol": cp.Uniform(1e19, 3e19),
        "b_width": cp.Uniform(0.015, 0.025),
        "b_slope": cp.Uniform(0.005, 0.020)
    }
    vary_2 = {
        "Qe_tot": cp.Uniform(1.8e6, 2.2e6),
        "Te_bc": cp.Uniform(80.0, 120.0)
    }
    vary_5 = {
        "Qe_tot": cp.Uniform(1.8e6, 2.2e6),
        "H0": cp.Uniform(0.0, 0.2),
        "Hw": cp.Uniform(0.1, 0.5),
        "chi": cp.Uniform(0.8, 1.2),
        "Te_bc": cp.Uniform(80.0, 120.0)
    }
    vary_10 = {
        "Qe_tot": cp.Uniform(1.8e6, 2.2e6),
        "H0": cp.Uniform(0.0, 0.2),
        "Hw": cp.Uniform(0.1, 0.5),
        "chi": cp.Uniform(0.8, 1.2),
        "Te_bc": cp.Uniform(80.0, 120.0),
        "b_pos": cp.Uniform(0.95, 0.99),
        "b_height": cp.Uniform(5e19, 7e19),
        "b_sol": cp.Uniform(1e19, 3e19),
        "b_width": cp.Uniform(0.015, 0.025),
        "b_slope": cp.Uniform(0.005, 0.020)
    }
    return vary_10


def run_MC_case(n_mc, local=True, dask=True, batch_size=os.cpu_count()):
    """
    Inputs:
        n_mc: the number of MC samples
        local: if using Dask, whether to use the local option (True)
        dask: whether to use dask (True)
        batch_size: for the non Dask option, number of cases to run in parallel (16)
    Outputs:
        results_df: Pandas dataFrame containing inputs to and output from the model
        my_campaign: EasyVVUQ MC campaign object
        my_sampler: EasyVVUQ RandomSampler object
    """

    times = np.zeros(9)

    time_start = time.time()
    time_start_whole = time_start
    # Set up a fresh campaign called "fusion_pce."
    if dask:
        my_campaign = uq.CampaignDask(name='fusion_mc.')
    else:
        my_campaign = uq.Campaign(name='fusion_mc.')

        # Define parameter space
    params = define_params()

    # Create an encoder and decoder for PCE test app
    encoder = uq.encoders.GenericEncoder(template_fname='fusion.template',
                                         delimiter='$',
                                         target_filename='fusion_in.json')

    decoder = uq.decoders.SimpleCSV(target_filename="output.csv",
                                    output_columns=["te", "ne", "rho", "rho_norm"])

    # Add the app (automatically set as current app)
    my_campaign.add_app(name="fusion", params=params, encoder=encoder, decoder=decoder)

    time_end = time.time()
    times[1] = time_end - time_start
    print('Time for phase 1 = %.3f' % (times[1]))

    time_start = time.time()
    # Create the sampler
    vary = define_vary()

    # Associate a sampler with the campaign
    my_sampler = uq.sampling.RandomSampler(vary=vary, max_num=n_mc)
    my_campaign.set_sampler(my_sampler)

    # Will draw all (of the finite set of samples)
    my_campaign.draw_samples()
    print('Number of samples = %s' % my_campaign.get_active_sampler().count)

    time_end = time.time()
    times[2] = time_end - time_start
    print('Time for phase 2 = %.3f' % (times[2]))

    time_start = time.time()
    # Create and populate the run directories
    my_campaign.populate_runs_dir()

    time_end = time.time()
    times[3] = time_end - time_start
    print('Time for phase 3 = %.3f' % (times[3]))

    time_start = time.time()
    # Run the cases
    cwd = os.getcwd().replace(' ', '\\ ')  # deal with ' ' in the path
    cmd = f"{cwd}/fusion_model.py fusion_in.json"

    if dask:
        if local:
            print('Running locally')
            import multiprocessing.popen_spawn_posix
            #        from distributed import Client
            from dask.distributed import Client, LocalCluster
            cluster = LocalCluster(threads_per_worker=1)
            #        Client()
            client = Client(cluster)  # processes=True, threads_per_worker=1)
        else:
            print('Running using SLURM')
            from dask.distributed import Client
            from dask_jobqueue import SLURMCluster
            cluster = SLURMCluster(
                job_extra=[
                    '--qos=p.tok.openmp.2h',
                    '--mail-type=end',
                    '--mail-user=dpc@rzg.mpg.de',
                    '-t 2:00:00'],
                queue='p.tok.openmp',
                cores=8,
                memory='8 GB',
                processes=8)
            cluster.scale(32)
            print(cluster)
            print(cluster.job_script())
            client = Client(cluster)
        print(client)
        my_campaign.apply_for_each_run_dir(
            uq.actions.ExecuteLocal(
                cmd, interpret='python3'), client)

        client.close()
        if not local:
            client.shutdown()
        else:
            client.shutdown()
    else:  # in case there is a problem with dask
        execution = my_campaign.apply_for_each_run_dir(
            uq.actions.ExecuteLocalV2(
                cmd, interpret='python3'), batch_size=batch_size)
        execution.start()
        while my_campaign.get_active_sampler().count != execution.progress()['finished']:
            print(execution.progress())
            time.sleep(1)
        print(execution.progress())

    time_end = time.time()
    times[4] = time_end - time_start
    print('Time for phase 4 = %.3f' % (times[4]))

    time_start = time.time()
    # Collate the results
    my_campaign.collate()
    results_df = my_campaign.get_collation_result()

    return results_df, my_campaign, my_sampler


def save_features(campaign, sampler, n_mc):
    # number of inputs
    D = len(sampler.vary.get_keys())
    # store the parameters in theta, will be used as features
    theta = np.zeros([n_mc, D])
    # loop over all runs
    for i, run in enumerate(campaign.list_runs()):
        # get the parameter values
        values = run[1]['params']
        for j, param in enumerate(values):
            # last entry in values will be the output filename, do not store
            if j < D:
                theta[i, j] = values[param]
    theta = theta.tolist()
    theta = [np.array(x).reshape(-1, 1) for x in theta]
    return theta


def save_target(data_frame):
    # Extract the QoI from the EasyVVUQ data frame, store as an array
    # Note: this should be made a subroutine in EasySurrogate, such that you can just feed it
    # an EasyVVUQ data frame, and get an QoI array in return
    qoi = 'te'
    samples = []
    for run_id in data_frame[('run_id', 0)].unique():
        values = data_frame.loc[data_frame[('run_id', 0)] == run_id][qoi].values
        samples.append(values.flatten())
    samples = np.array(samples)
    return samples


def main():

    local = True
    dask = True

    n_mc = 500

    data_frame, campaign, sampler = run_MC_case(n_mc=500, local=local, dask=dask, batch_size=7)

    features = save_features(campaign, sampler, n_mc)
    pd.DataFrame(features).to_pickle('inputs.pickle')

    samples = save_target(data_frame)
    pd.DataFrame(samples).to_pickle('outputs.pickle')


if __name__ == "__main__":
    main()
