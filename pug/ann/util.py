"""Utilities for maniuplating, analyzing and plotting `pybrain` `Network` and `DataSet` objects

TODO:
    Incorporate into pybrain fork so pug doesn't have to depend on pybrain

"""
from __future__ import print_function
import os

import pandas as pd
from scipy import ndarray, reshape  # array, amin, amax, 
np = pd.np
from matplotlib import pyplot as plt
import pybrain.datasets
import pybrain.structure
import pybrain.supervised
import pybrain.tools
pb = pybrain
# from pybrain.supervised.trainers import Trainer
from pybrain.tools.customxml import NetworkReader
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.structure.connections.connection import Connection




#import pug.nlp.util as nlp

# print(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')


def build_ann(N_input=None, N_hidden=2, N_output=1):
    """Build a neural net with the indicated input, hidden, and outout dimensions

    Arguments:
        params (dict or PyBrainParams namedtuple): 
            default: {'N_hidden': 6}
            (this is the only parameter that affects the NN build)

    Returns:
        FeedForwardNetwork with N_input + N_hidden + N_output nodes in 3 layers
    """
    N_input = N_input or 1
    N_output = N_output or 1

    nn = pb.structure.FeedForwardNetwork()

    # layers
    nn.addInputModule(pb.structure.BiasUnit(name='bias'))
    nn.addInputModule(pb.structure.LinearLayer(N_input, name='input'))
    if N_hidden:
        nn.addModule(pb.structure.LinearLayer(N_hidden, name='hidden'))
    nn.addOutputModule(pb.structure.LinearLayer(N_output, name='output'))

    # connections
    nn.addConnection(pb.structure.FullConnection(nn['bias'],  nn['hidden'] if N_hidden else nn['output']))
    nn.addConnection(pb.structure.FullConnection(nn['input'], nn['hidden'] if N_hidden else nn['output']))
    if N_hidden:
        nn.addConnection(pb.structure.FullConnection(nn['hidden'], nn['output']))

    nn.sortModules()
    return nn


def ann_from_ds(ds=None, N_input=3, N_hidden=0, N_output=1):
    N_input = getattr(ds, 'indim', N_input)
    N_output = getattr(ds, 'outdim', N_output)
    N_hidden = getattr(ds, 'paramdim', N_hidden + N_input + N_output) - N_hidden - N_output

    return build_ann(N_input=N_input, N_hidden=N_hidden, N_output=N_output)


def dataset_from_dataframe(df, delays=[1,2,3], inputs=[1, 2, -1], outputs=[-1], normalize=True, verbosity=1):
    """Compose a pybrain.dataset from a pandas DataFrame

    Arguments:
        delays (list of int): sample delays to use for the input tapped delay line
            Positive and negative values are treated the same as sample counts into the past.
            default: [1, 2, 3], in z-transform notation: z^-1 + z^-2 + z^-3
        inputs (list of int or list of str): column indices or labels for the inputs
        outputs (list of int or list of str): column indices or labels for the outputs
        normalize (bool): whether to divide each input to be normally distributed about 0 with std 1

    Returns:
        3-tuple: tuple(dataset, list of means, list of stds)
            means and stds allow normalization of new inputs and denormalization of the outputs
    """

    if isinstance(delays, int):
        delays = range(1, delays+1)
    delays = np.abs(np.array([int(i) for i in delays]))
    inputs = [df.columns[int(inp)] if isinstance(inp, (float, int)) else str(inp) for inp in inputs]
    outputs = [df.columns[int(out)] if isinstance(out, (float, int)) else str(out) for out in outputs]

    N_inp = len(inputs)
    N_out = len(outputs)

    inp_outs = inputs + outputs
    if verbosity > 0:
        print("inputs: {}\noutputs: {}\ndelays: {}\n".format(inputs, outputs, delays))
    means, stds = np.zeros(len(inp_outs)), np.ones(len(inp_outs))
    if normalize:
        means, stds = df[inp_outs].mean(), df[inp_outs].std()

    if verbosity > 0:
        print("input means: {}".format(means[:N_inp]))
    ds = pb.datasets.SupervisedDataSet(N_inp * len(delays), N_out)
    print("Dataset dimensions are {}x{}".format(ds.indim, ds.outdim))
    for i, out_vec in enumerate(df[outputs].values):
        if verbosity > 1:
            print(i, out_vec)
        if i < max(delays):
            continue
        inp_vec = []
        for delay in delays:
            inp_vec += list((df[inputs].values[i-delay] - means[:N_inp]) / stds[:N_inp])
        ds.addSample(inp_vec, (out_vec - means[N_inp:]) / stds[N_inp:])
    print("Dataset now has {} samples".format(len(ds)))
    return ds, means, stds


def build_trainer(nn, ds, verbosity=1):
    """Configure neural net trainer from a pybrain dataset"""
    return pb.supervised.trainers.rprop.RPropMinusTrainer(nn, dataset=ds, batchlearning=True, verbose=bool(verbosity))


def weight_matrices(nn):
    """ Extract list of weight matrices from a Network, Layer (module), Trainer, Connection or other pybrain object"""

    if isinstance(nn, ndarray):
        return nn

    try:
        return weight_matrices(nn.connections)
    except:
        pass

    try:
        return weight_matrices(nn.module)
    except:
        pass

    # Network objects are ParameterContainer's too, but won't reshape into a single matrix, 
    # so this must come after try nn.connections
    if isinstance(nn, (ParameterContainer, Connection)):
        return reshape(nn.params, (nn.outdim, nn.indim))

    if isinstance(nn, basestring):
        try:
            fn = nn
            nn = NetworkReader(fn, newfile=False)
            return weight_matrices(nn.readFrom(fn))
        except:
            pass
    # FIXME: what does NetworkReader output? (Module? Layer?) need to handle it's type here

    try:
        return [weight_matrices(v) for (k, v) in nn.iteritems()]
    except:
        try:
            connections = nn.module.connections.values()
            nn = []
            for conlist in connections:
                nn += conlist
            return weight_matrices(nn)
        except:
            return [weight_matrices(v) for v in nn]



# # FIXME: resolve all these NLP dependencies and get this working

# def dataset_from_time_series(df, N_inp=None, features=('moy',), verbosity=1):
#     """Build a pybrains.dataset from the time series contained in a dataframe"""
#     N_inp = N_inp or len(df.columns)
#     features = features or []
#     # Add features to input vector in reverse alphabetical order by feature name,
#     #   so woy will be added first, and date will be added last.
#     # The order that the feature vectors should appear in the input vector to remain consistent 
#     #   and neural net architecture can have structure that anticipates this.
#     sorted_features = nlp.sort_strings(features, ('dat', 'dow', 'moy', 'dom', 'moy', 'mor'), case_sensitive=False)
#     if verbosity > 0:
#         print('dataset_from_thresh(features={0})'.format(features))

#     samples, mean, std, thresh = simple_dataset_from_thresh(thresh, N=N, max_window=max_window, normalize=normalize, ignore_below=ignore_below)

#     name = getattr(thresh, 'name', None)
#     if name:
#         name = normalize_building_name(name)
#     if name:
#         series = get_series(name)
#     else:
#         if isinstance(series, basestring):
#             name = normalize_building_name(series.strip()) or thresh.name or 'Unknown'
#             series = get_series(name)
#         elif isinstance(series, pd.DataFrame):
#             name = normalize_building_name(series.columns[0]) or thresh.name or 'Unknown'
#             series = series[name]
#         elif isinstance(series, pd.Series):
#             name = normalize_building_name(series.name) or thresh.name or 'Unknown'
#         else:
#             name = None

#     # Compute the length of extra features added on to the vector from the rolling window of previous threshold values
#     # TODO: pre-process features list of strings in a separate function
#     morn = 0

#     # if the building name isn't known, you can't retrieve the morning load values for it
#     if name:
#         for s in features:
#             if s.startswith('morn'):
#                 try:
#                     morn = int(s[4:])
#                 except:
#                     if verbosity > 0:
#                         warnings.warn('Unable to determine morning length from feature named "{0}" so using default (8 am = 8 * 4 = 32)')
#                     morn = 32  # default to 9 am morning ending
#                 break

#     if verbosity > 0:
#         print('In dataset_from_thresh() using {0} morning load values for Building {1} because series arg is of type {2}'.format(morn, name, type(series)))

#     extras = (+ int('dow' in features) * 7
#               + int('moy' in features) * 12
#               + int('woy' in features)
#               + int('date' in features)
#               + morn)

#     if verbosity > 0:
#         print('The total input vector length (dimension) is now {0}'.format(N + extras))
#     ds = pb.datasets.SupervisedDataSet(N + extras, 1)
#     first_date = samples[0]['target'].index[0].date().toordinal()
#     last_date = samples[-1]['target'].index[0].date().toordinal()
#     date_range = (last_date - first_date) or 1

#     bit_scale = 5  # number of standard deviations for the magnitude of bit            

#     if verbosity > 0:
#         print('Adding features for building {3}, {0}, and a morning time series of len {2}, to each of the {1} vectors (samples)'.format(features, len(samples), morn, name))

#     for sampnum, sample in enumerate(samples):
#         # sample['input'] and ['output'] are pd.Series tables so convert them to normal list()
#         inputs = list(sample['input'].values)
#         # the date we're trying to predict the rhreshold for
#         timestamp = sample['target'].index[0]
#         for feature_name in sorted_features:
#             if feature_name.startswith('morn'):
#                 day = get_day(series, date=timestamp.date())
#                 morning_loads = (day.values[:morn] - mean) / std
#                 if verbosity > 1:
#                     print('day = {0} and morning = {1}'.format(len(day), len(morning_loads)))
#                 inputs = list(morning_loads) + inputs
#             elif feature_name == 'dow':
#                 dow_bits = [0] * 7
#                 dow_bits[timestamp.weekday()] = bit_scale
#                 inputs = dow_bits + inputs
#             elif feature_name == 'moy':
#                 moy_bits = [0] * 12
#                 moy_bits[timestamp.month - 1] = bit_scale
#                 inputs = moy_bits + inputs
#             elif feature_name == 'woy':
#                 inputs = [(timestamp.weekofyear - 26.) * 3 * bit_scale / 52] + inputs
#             elif feature_name == 'date':
#                 inputs = [(timestamp.date().toordinal() - first_date - date_range / 2.) * 3 * bit_scale / date_range ] + inputs

#             if pd.isnull(inputs).any():
#                 msg = 'Feature "{0}" within the feature list: {1} created null/NaN input values\nFor sample {2} and date {3}\nInput vector positions {4}:\nInput vector: {5}'.format(
#                     feature_name, sorted_features, sampnum, timestamp, ann.table_nan_locs(inputs), inputs)
#                 msg += '\nBuilding load Series:\n{0}\n'.format(series)
#                 if ignore_nans:
#                     warnings.warn(msg)
#                 else:
#                     raise ValueError(msg)
#         ds.addSample(inputs, list(sample['target'].values))

#     return ds, mean, std, thresh



def dataset_nan_locs(ds):
    """
    from http://stackoverflow.com/a/14033137/623735
    # gets the indices of the rows with nan values in a dataframe
    pd.isnull(df).any(1).nonzero()[0]
    """
    ans = []
    for sampnum, sample in enumerate(ds):
        if pd.isnull(sample).any():
            ans += [{
                'sample': sampnum,
                'input':  pd.isnull(sample[0]).nonzero()[0],
                'output': pd.isnull(sample[1]).nonzero()[0],
                }]
    return ans


def table_nan_locs(table):
    """
    from http://stackoverflow.com/a/14033137/623735
    # gets the indices of the rows with nan values in a dataframe
    pd.isnull(df).any(1).nonzero()[0]
    """
    ans = []
    for rownum, row in enumerate(table):
        try:
            if pd.isnull(row).any():
                colnums = pd.isnull(row).nonzero()[0]
                ans += [(rownum, colnum) for colnum in colnums]
        except AttributeError:  # table is really just a sequence of scalars
            if pd.isnull(row):
                ans += [(rownum, 0)]
    return ans


def plot_network_results(network, ds=None, mean=0, std=1, title='', show=True, save=True):
    """Identical to plot_trainer except `network` and `ds` must be provided separately"""
    df = sim_network(network=network, ds=ds, mean=mean, std=std)
    df.plot()
    plt.xlabel('Date')
    plt.ylabel('Threshold (kW)')
    plt.title(title)

    if show:
        try:
            # ipython notebook overrides plt.show and doesn't have a block kwarg
            plt.show(block=False)
        except TypeError:
            plt.show()
    if save:
        filename = 'ann_performance_for_{0}.png'.format(title).replace(' ', '_')
        if isinstance(save, basestring) and os.path.isdir(save):
            filename = os.path.join(save, filename) 
        plt.savefig(filename)
    if not show:
        plt.clf()

    return network, mean, std


def trainer_results(trainer, mean=0, std=1, title='', show=True, save=True):
    """Plot the performance of the Network and SupervisedDataSet in a pybrain Trainer

    DataSet target and output values are denormalized before plotting with:

        output * std + mean

    Which inverses the normalization 

        (output - mean) / std

    Args:
        trainer (Trainer): a pybrain Trainer instance containing a valid Network and DataSet
        ds (DataSet): a pybrain DataSet to override the one contained in `trainer`. 
          Required if trainer is a Network instance rather than a Trainer instance.
        mean (float): mean of the denormalized dataset (default: 0)
          Only affects the scale of the plot
        std (float): std (standard deviation) of the denormalized dataset (default: 1)
        title (str): title to display on the plot.

    Returns:
        3-tuple: (trainer, mean, std), A trainer/dataset along with denormalization info
    """
    return plot_network_results(network=trainer.module, ds=trainer.ds, mean=mean, std=std, title=title, show=show, save=save)




def sim_trainer(trainer, mean=0, std=1):
    """Simulate a trainer by activating its DataSet and returning DataFrame(columns=['Output','Target'])
    """
    return sim_network(network=trainer.module, ds=trainer.ds, mean=mean, std=std)


def sim_network(network, ds=None, index=None, mean=0, std=1):
    """Simulate/activate a Network on a SupervisedDataSet and return DataFrame(columns=['Output','Target'])

    The DataSet's target and output values are denormalized before populating the dataframe columns:

        denormalized_output = normalized_output * std + mean

    Which inverses the normalization that produced the normalized output in the first place: 

        normalized_output = (denormalzied_output - mean) / std

    Args:
        network (Network): a pybrain Network instance to activate with the provided DataSet, `ds`
        ds (DataSet): a pybrain DataSet to activate the Network on to produce an output sequence
        mean (float): mean of the denormalized dataset (default: 0)
          Output is scaled
        std (float): std (standard deviation) of the denormalized dataset (default: 1)
        title (str): title to display on the plot.

    Returns:
        DataFrame: DataFrame with columns "Output" and "Target" suitable for df.plot-ting
    """
    # just in case network is a trainer or has a Module-derived instance as one of it's attributes
    if hasattr(network, 'module') and hasattr(network.module, 'activate'):  # isinstance(network.module, (networks.Network, modules.Module))
        network = network.module
    ds = ds or network.ds
    if not ds:
        raise RuntimeError("Unable to find a `pybrain.DataSet` instance to activate the Network with in order to plot the outputs. A dataset can be provided as part of a network instance or as a separate kwarg if `network` is used to provide the `pybrain.Network` instance directly.")
    results_generator = ((network.activate(ds['input'][i])[0] * std + mean, ds['target'][i][0] * std + mean) for i in xrange(len(ds['input'])))
    
    return pd.DataFrame(results_generator, columns=['Output', 'Target'], index=index or range(len(ds['input'])))
