"""Maniuplate, analyze and plot `pybrain` `Network` and `DataSet` objects

TODO:
    Incorporate into pybrain fork so pug doesn't have to depend on pybrain

"""
from __future__ import print_function
import os
import warnings

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

from pug.nlp.util import tuplify, fuzzy_get
from pug.ann.data import weather

#import pug.nlp.util as nlp

# print(os.path.realpath(__file__))
DATA_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(DATA_PATH, '..', 'data')
LAYER_TYPES = set([getattr(pybrain.structure, s) for s in
                  dir(pybrain.structure) if s.endswith('Layer')])

FAST = False
try:
    from arac.pybrainbridge import _FeedForwardNetwork as FeedForwardNetwork
    FAST = True
except:
    from pybrain.structure import FeedForwardNetwork
    print("No fast (ARAC, a FC++ library) FeedForwardNetwork was found, "
          "so using the slower pybrain python implementation of FFNN.")


def normalize_layer_type(layer_type):
    try:
        if layer_type in LAYER_TYPES:
            return layer_type
    except TypeError:
        pass
    try:
        return getattr(pb.structure, layer_type.strip())
    except AttributeError:
        try:
            return getattr(pb.structure, layer_type.strip() + 'Layer')
        except AttributeError:
            pass
    return [normalize_layer_type(lt) for lt in layer_type]


def build_ann(N_input=None, N_hidden=2, N_output=1, hidden_layer_type='Linear', verbosity=1):
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
    N_hidden = N_hidden or tuple()
    if isinstance(N_hidden, (int, float, basestring)):
        N_hidden = (int(N_hidden),)

    hidden_layer_type = hidden_layer_type or tuple()
    hidden_layer_type = tuplify(normalize_layer_type(hidden_layer_type))

    if verbosity > 0:
        print(N_hidden, ' layers of type ', hidden_layer_type)

    assert(len(N_hidden) == len(hidden_layer_type))
    nn = pb.structure.FeedForwardNetwork()

    # layers
    nn.addInputModule(pb.structure.BiasUnit(name='bias'))
    nn.addInputModule(pb.structure.LinearLayer(N_input, name='input'))
    for i, (Nhid, hidlaytype) in enumerate(zip(N_hidden, hidden_layer_type)):
        Nhid = int(Nhid)
        nn.addModule(hidlaytype(Nhid, name=('hidden-{}'.format(i) if i else 'hidden')))
    nn.addOutputModule(pb.structure.LinearLayer(N_output, name='output'))

    # connections
    nn.addConnection(pb.structure.FullConnection(nn['bias'],  nn['hidden'] if N_hidden else nn['output']))
    nn.addConnection(pb.structure.FullConnection(nn['input'], nn['hidden'] if N_hidden else nn['output']))
    for i, (Nhid, hidlaytype) in enumerate(zip(N_hidden[:-1], hidden_layer_type[:-1])):
        Nhid = int(Nhid)
        nn.addConnection(pb.structure.FullConnection(nn[('hidden-{}'.format(i) if i else 'hidden')],
                         nn['hidden-{}'.format(i + 1)]))
    i = len(N_hidden) - 1
    nn.addConnection(pb.structure.FullConnection(nn['hidden-{}'.format(i) if i else 'hidden'], nn['output']))

    nn.sortModules()
    if FAST:
        try:
            nn.convertToFastNetwork()
        except:
            if verbosity > 0:
                print('Unable to convert slow PyBrain NN to a fast ARAC network...')
    if verbosity > 0:
        print(nn.connections)
    return nn


def ann_from_ds(ds=None, N_input=3, N_hidden=0, N_output=1, verbosity=1):
    N_input = getattr(ds, 'indim', N_input)
    N_output = getattr(ds, 'outdim', N_output)
    N_hidden = N_hidden or getattr(ds, 'paramdim', N_hidden + N_input + N_output) - N_input - N_output
    N_hidden = max(round(min(N_hidden, len(ds) / float(N_input) / float(N_output) / 5.)), N_output)

    return build_ann(N_input=N_input, N_hidden=N_hidden, N_output=N_output, verbosity=verbosity)


def prepend_dataset_with_weather(samples, location='Fresno, CA', weather_columns=None, use_cache=True, verbosity=0):
    """ Prepend weather the values specified (e.g. Max TempF) to the samples[0..N]['input'] vectors

    samples[0..N]['target'] should have an index with the date timestamp

    If you use_cache for the curent year, you may not get the most recent data.

    Arguments:
        samples (list of dict): {'input': np.array(), 'target': pandas.DataFrame}
    """
    if verbosity > 1:
        print('Prepending weather data for {} to dataset samples'.format(weather_columns))
    if not weather_columns:
        return samples
    timestamps = pd.DatetimeIndex([s['target'].index[0] for s in samples])
    years = range(timestamps.min().date().year, timestamps.max().date().year + 1)
    weather_df = weather.daily(location=location, years=years, use_cache=use_cache)
    # FIXME: weather_df.resample('D') fails
    weather_df.index = [d.date() for d in weather_df.index]
    if verbosity > 1:
        print('Retrieved weather for years {}:'.format(years))
        print(weather_df)
    weather_columns = [label if label in weather_df.columns else weather_df.columns[int(label)]
                       for label in (weather_columns or [])]
    for sampnum, sample in enumerate(samples):
        timestamp = timestamps[sampnum]
        try:
            weather_day = weather_df.loc[timestamp.date()]
        except:
            from traceback import print_exc
            print_exc()
            weather_day = {}
            if verbosity >= 0:
                warnings.warn('Unable to find weather for the date {}'.format(timestamp.date()))
        NaN = float('NaN')
        sample['input'] = [weather_day.get(label, None) for label in weather_columns] + list(sample['input'])
        if verbosity > 0 and NaN in sample['input']:
            warnings.warn('Unable to find weather features {} in the weather for date {}'.format(
                [label for i, label in enumerate(weather_columns) if sample['input'][i] == NaN], timestamp))
    return samples


def dataset_from_dataframe(df, delays=(1, 2, 3), inputs=(1, 2, -1), outputs=(-1,), normalize=False, verbosity=1):
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

    TODO:

        Detect categorical variables with low dimensionality and split into separate bits
            Vowpel Wabbit hashes strings into an int?
        Detect ordinal variables and convert to continuous int sequence
        SEE: http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    """
    if isinstance(delays, int):
        if delays:
            delays = range(1, delays + 1)
        else:
            delays = [0]
    delays = np.abs(np.array([int(i) for i in delays]))
    inputs = [df.columns[int(inp)] if isinstance(inp, (float, int)) else str(inp) for inp in inputs]
    outputs = [df.columns[int(out)] if isinstance(out, (float, int)) else str(out) for out in (outputs or [])]

    inputs = [fuzzy_get(df.columns, i) for i in inputs]
    outputs = [fuzzy_get(df.columns, o) for o in outputs]

    N_inp = len(inputs)
    N_out = len(outputs)

    inp_outs = inputs + outputs
    if verbosity > 0:
        print("inputs: {}\noutputs: {}\ndelays: {}\n".format(inputs, outputs, delays))
    means, stds = np.zeros(len(inp_outs)), np.ones(len(inp_outs))
    if normalize:
        means, stds = df[inp_outs].mean(), df[inp_outs].std()

    if normalize and verbosity > 0:
        print("Input mean values (used to normalize input biases): {}".format(means[:N_inp]))
        print("Output mean values (used to normalize output biases): {}".format(means[N_inp:]))
    ds = pb.datasets.SupervisedDataSet(N_inp * len(delays), N_out)
    if verbosity > 0:
        print("Dataset dimensions are {}x{}x{} (records x indim x outdim) for {} delays, {} inputs, {} outputs".format(
              len(df), ds.indim, ds.outdim, len(delays), len(inputs), len(outputs)))
    # FIXME: normalize the whole matrix at once and add it quickly rather than one sample at a time
    if delays == np.array([0]) and not normalize:
        if verbosity > 0:
            print("No tapped delay lines (delays) were requested, so using undelayed features for the dataset.")
        assert(df[inputs].values.shape[0] == df[outputs].values.shape[0])
        ds.setField('input', df[inputs].values)
        ds.setField('target', df[outputs].values)
        ds.linkFields(['input', 'target'])
        # for inp, outp in zip(df[inputs].values, df[outputs].values):
        #     ds.appendLinked(inp, outp)
        assert(len(ds['input']) == len(ds['target']))
    else:
        for i, out_vec in enumerate(df[outputs].values):
            if verbosity > 0 and i % 100 == 0:
                print("{}%".format(i / .01 / len(df)))
            elif verbosity > 1:
                print('sample[{i}].target={out_vec}'.format(i=i, out_vec=out_vec))
            if i < max(delays):
                continue
            inp_vec = []
            for delay in delays:
                inp_vec += list((df[inputs].values[i - delay] - means[:N_inp]) / stds[:N_inp])
            ds.addSample(inp_vec, (out_vec - means[N_inp:]) / stds[N_inp:])
    if verbosity > 0:
        print("Dataset now has {} samples".format(len(ds)))
    if normalize:
        return ds, means, stds
    else:
        return ds


# def dataset_from_feature_names(df, delays=(1, 2,), quantiles=(), input_columns=(0,), target_columns=(-1,),
#                                weather_columns=(), features=(), verbosity=1):
#     """FIXME: Untested. Transform Datafram columns and append to input_columns as additional features

#     Arguments:
#         features (seq of str): names of feautures to be appended to the input vector (feature set)

#     """
#     raise NotImplementedError("Better to implement this as a separate feature transformation function")
#     # Compute the length of extra features added on to the vector from the rolling window of previous
#     # threshold values
#     extras = (+ int('dow' in features) * 7
#               + int('moy' in features) * 12
#               + int('woy' in features)
#               + int('date' in features)
#               + len(quantiles)
#               + len(weather_columns)
#               )

#     N = max(list(delays) + list(quantiles))
#     last_date = first_date = 0
#     try:
#         if verbosity > 0:
#             print('The total input vector length (dimension) is now {0}'.format(len(N) + extras))
#             print('Starting to augment {0}x{1} sample inputs by adding adding {2} additional features'.format(
#                 len(df), len(input_columns), extras))
#         if verbosity > 0:
#             print('The first sample input was {0}'.format(
#                 df[input_columns].iloc[0]))
#             print('The first sample target was {0}'.format(
#                 df[target_columns].iloc[0]))
#             print('The last sample input was {0}'.format(
#                 df[input_columns].iloc[-1]))
#             print('The last sample target was {0}'.format(
#                 df[target_columns].iloc[-1]))
#         first_date = df.index.iloc[0].date().toordinal()
#         last_date = df.index.iloc[-1].date().toordinal()
#     except:
#         if verbosity > -1:
#             from traceback import format_exc
#             warnings.warn(format_exc())
#         if verbosity > 1:
#             import ipdb
#             ipdb.set_trace()
#     date_range = (last_date - first_date) + 1 or 1

#     # FIXME: scale each feature/column/dimension independently using pug.ann.util.dataset_from_dataframe
#     #        but mean and std become vectors of the same dimension as the feature/input vector.
#     #        scikit-learn has transformations that do this more reasonably
#     bit_scale = 5  # number of standard deviations for the magnitude of bit

#     # convert the list of dicts ((input, output) supervised dataset pairs) into a pybrains.Dataset
#     ds = pybrain.datasets.SupervisedDataSet(len(N) + extras, 1)
#     sorted_features = sorted(features)
#     for sampnum, (input_vector, target_vector) in enumerate(
#             zip(df[input_columns].values, df[target_columns].values)):
#         # sample['input'] and ['output'] are pd.Series tables so convert them to normal list()
#         # inputs = list(sample['input'])

#         # the date we're trying to predict the threshold for
#         timestamp = target_vector.index[0]
#         for feature_name in sorted_features:
#             if feature_name.startswith('morn'):
#                 day = get_day(series, date=timestamp.date())
#                 morning_loads = (day.values[:morn] - mean) / std
#                 if verbosity > 2:
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
#                 inputs = [(timestamp.date().toordinal() - first_date - date_range / 2.) * 3 * bit_scale / date_range
#                           ] + inputs

#             if pd.isnull(inputs).any():
#                 msg = 'Feature "{0}" within the feature list: {1} created null/NaN input values\n'.format(
#                     feature_name, sorted_features)
#                 msg += 'For sample {} and date {}\n'.format(sampnum, timestamp)
#                 msg += 'Input vector positions {}:\nInput vector: {}'.format(
#                     ann.table_nan_locs(inputs), inputs)
#                 msg += '\nBuilding load Series:\n{0}\n'.format(series)
#                 if ignore_nans:
#                     warnings.warn(msg)
#                 else:
#                     raise ValueError(msg)

#         ds.addSample(inputs, list(target_vector.values))


def input_dataset_from_dataframe(df, delays=(1, 2, 3), inputs=(1, 2, -1), outputs=None, normalize=True, verbosity=1):
    """ Build a dataset with an empty output/target vector

    Identical to `dataset_from_dataframe`, except that default values for 2 arguments:
        outputs: None
    """
    return dataset_from_dataframe(df=df, delays=delays, inputs=inputs, outputs=outputs,
                                  normalize=normalize, verbosity=verbosity)


def inputs_from_dataframe(df, delays=(1, 2, 3), inputs=(1, 2, -1), outputs=None, normalize=True, verbosity=1):
    """ Build a sequence of vectors suitable for "activation" by a neural net

    Identical to `dataset_from_dataframe`, except that only the input vectors are
    returned (not a full DataSet instance) and default values for 2 arguments are changed:
        outputs: None

    And only the input vectors are return
    """
    ds = input_dataset_from_dataframe(df=df, delays=delays, inputs=inputs, outputs=outputs,
                                      normalize=normalize, verbosity=verbosity)
    return ds['input']


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

#     samples, mean, std, thresh = simple_dataset_from_thresh(thresh, N=N, max_window=max_window,
#                                                             normalize=normalize, ignore_below=ignore_below)

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

#     # Compute the length of extra features added on to the vector from the rolling window of previous
#     # threshold values
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
#                         warnings.warn('Unable to determine morning length from feature named "{0}" so using default '
#                                       '(8 am = 8 * 4 = 32)')
#                     morn = 32  # default to 9 am morning ending
#                 break

#     if verbosity > 0:
#         print('In dataset_from_thresh() using {0} morning load values for Building {1}'
#               ' because series arg is of type {2}'.format(morn, name, type(series)))

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
#         print('Adding features for building {3}, {0}, and a morning time series of len {2}, '
#               'to each of the {1} vectors (samples)'.format(features, len(samples), morn, name))

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
#                 inputs = [(timestamp.date().toordinal() - first_date - date_range / 2.) * 3 * bit_scale /
#                            date_range ] + inputs

#             if pd.isnull(inputs).any():
#                 msg = 'Feature "{0}" within the feature list: {1} created null/NaN input values\nFor sample {2}'
#                       ' and date {3}\nInput vector positions {4}:\nInput vector: {5}'.format(
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
    return plot_network_results(network=trainer.module, ds=trainer.ds, mean=mean, std=std, title=title,
                                show=show, save=save)


def sim_trainer(trainer, mean=0, std=1):
    """Simulate a trainer by activating its DataSet and returning DataFrame(columns=['Output','Target'])
    """
    return sim_network(network=trainer.module, ds=trainer.ds, mean=mean, std=std)


def sim_network(network, ds=None, index=None, mean=0, std=1):
    """Simulate/activate a Network on a SupervisedDataSet and return DataFrame(columns=['Output','Target'])

    The DataSet's target and output values are denormalized before populating the dataframe columns:

        denormalized_output = normalized_output * std + mean

    Which inverses the normalization that produced the normalized output in the first place: \

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
    # just in case network is a trainer or has a Module-derived instance as one of it's attribute
       # isinstance(network.module, (networks.Network, modules.Module))
    if hasattr(network, 'module') and hasattr(network.module, 'activate'):
        # may want to also check: isinstance(network.module, (networks.Network, modules.Module))
        network = network.module
    ds = ds or network.ds
    if not ds:
        raise RuntimeError("Unable to find a `pybrain.datasets.DataSet` instance to activate the Network with, "
                           " to plot the outputs. A dataset can be provided as part of a network instance or "
                           "as a separate kwarg if `network` is used to provide the `pybrain.Network`"
                           " instance directly.")
    results_generator = ((network.activate(ds['input'][i])[0] * std + mean, ds['target'][i][0] * std + mean)
                         for i in xrange(len(ds['input'])))

    return pd.DataFrame(results_generator, columns=['Output', 'Target'], index=index or range(len(ds['input'])))
