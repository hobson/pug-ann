"""Example pybrain network training to predict the weather

Installation:

    pip install pug-ann

Usage:

    >>> from pug.ann import example
    >>> example.predict_weather('San Francisco, CA')
"""

from pug.ann.data import weather
from pug.ann import util

def predict_weather(
            location='Camas, WA',
            years=range(2012, 2015),
            delays=[1,2,3], 
            inputs=['Min TemperatureF', 'Max TemperatureF', 'Min Sea Level PressureIn', u'Max Sea Level PressureIn', 'WindDirDegrees'], 
            outputs=[u'Max TemperatureF'],
            epochs=30,
            verbosity=2):
    """Predict the weather for tomorrow based on the weather for the past few days

    Builds a linear single-layer neural net (multi-dimensional regression).
    The dataset is a basic SupervisedDataSet rather than a SequentialDataSet, so there may be
    "accuracy left on the table" or even "cheating" during training, because training and test
    set are selected randomly so historical data for one sample is used as target (furture data)
    for other samples.

    Uses CSVs scraped from wunderground (no api key required) to get daily weather for the years indicated.

    Arguments:
        location (str): City and state in standard US postal service format: "City, ST" or an airport code like "PDX"
        delays (list of int): sample delays to use for the input tapped delay line.
            Positive and negative values are treated the same as sample counts into the past.
            default: [1, 2, 3], in z-transform notation: z^-1 + z^-2 + z^-3
        years (int or list of int): list of 4-digit years to download weather from wunderground
        inputs (list of int or list of str): column indices or labels for the inputs
        outputs (list of int or list of str): column indices or labels for the outputs

    Returns:
        3-tuple: tuple(dataset, list of means, list of stds)
            means and stds allow normalization of new inputs and denormalization of the outputs

    """
    df = weather.daily(location, years=years, verbosity=verbosity).sort()
    ds, means, stds = util.dataset_from_dataframe(df, delays=delays, inputs=inputs, outputs=outputs)
    nn = util.ann_from_ds(ds)
    trainer = util.build_trainer(nn, ds)
    training_err, validation_err = trainer.trainUntilConvergence(maxEpochs=epochs, verbose=bool(verbosity))
    return trainer

if __name__ == '__main__':
    print(predict_weather())