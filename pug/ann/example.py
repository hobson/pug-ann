"""Example pybrain network training to predict the weather

Installation:

    pip install pug-ann

Usage:

    >>> from pug.ann import example
    >>> example.predict_weather('San Francisco, CA')
"""

from pug.data import weather
from pug.ann import util

def predict_weather(location='Camas, WA', years=range(2011, 2015), verbosity=2):
    df = weather.daily(location, years=years, verbosity=verbosity).sort()
    ds, means, stds = util.dataset_from_dataframe(df, delays=[1,2,3,4], inputs=['Min TemperatureF', 'Max TemperatureF', ' Min Sea Level PressureIn', u' Max Sea Level PressureIn', ' WindDirDegrees'], outputs=[u'Max TemperatureF'])
    nn = util.ann_from_ds(ds)
    trainer = util.build_trainer(nn, ds)
    training_err, validation_err = trainer.trainUntilConvergence(maxEpochs=200, verbose=bool(verbosity))
    return validation_err

if __name__ == '__main__':
    print(predict_weather())