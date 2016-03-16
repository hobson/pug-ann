import os
import urllib
# import re
import datetime
import json
import warnings

import pandas as pd

from pug.nlp import util, env

np = pd.np


DATA_PATH = os.path.dirname(os.path.realpath(__file__))
CACHE_PATH = os.path.join(DATA_PATH, 'cache')


def airport(location, default=None):
    return airport.locations.get(location, default)
airport.locations = json.load(open(os.path.join(CACHE_PATH, 'airport.locations.json'), 'rUb'))


def hourly(location='Fresno, CA', days=1, start=None, end=None, years=1, use_cache=True, verbosity=1):
    """ Get detailed (hourly) weather data for the requested days and location

    The Weather Underground URL for Fresno, CA on 1/1/2011 is:
    http://www.wunderground.com/history/airport/KFAT/2011/1/1/DailyHistory.html?MR=1&format=1

    This will fail periodically on Travis, b/c wunderground says "No daily or hourly history data available"
    >> df = hourly('Fresno, CA', verbosity=-1)
    >> 1 <= len(df) <= 24 * 2
    True
    The time zone of the client where this is used to compose the first column label, hence the ellipsis
    >> df.columns  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    Index([u'Time...

    >> df = hourly('Fresno, CA', days=5, verbosity=-1)
    >> 24 * 4 <= len(df) <= 24 * (5 + 1) * 2
    True
    """
    airport_code = airport(location, default=location)

    if isinstance(days, int):
        start = start or None
        end = end or datetime.datetime.today().date()
        days = pd.date_range(start=start, end=end, periods=days)

    # refresh the cache each calendar month or each change in the number of days in the dataset
    cache_path = 'hourly-{}-{}-{:02d}-{:04d}.csv'.format(airport_code, days[-1].year, days[-1].month, len(days))
    cache_path = os.path.join(CACHE_PATH, cache_path)
    if use_cache:
        try:
            return pd.DataFrame.from_csv(cache_path)
        except:
            pass

    df = pd.DataFrame()
    for day in days:
        url = ('http://www.wunderground.com/history/airport/{airport_code}/{year}/{month}/{day}/DailyHistory.html?MR=1&format=1'.format(
               airport_code=airport_code,
               year=day.year,
               month=day.month,
               day=day.day))
        if verbosity > 1:
            print('GETing *.CSV using "{0}"'.format(url))
        buf = urllib.urlopen(url).read()
        if verbosity > 0:
            N = buf.count('\n')
            M = (buf.count(',') + N) / float(N)
            print('Retrieved CSV for airport code "{}" with appox. {} lines and {} columns = {} cells.'.format(
                  airport_code, N, int(round(M)), int(round(M)) * N))
        if (buf.count('\n') > 2) or ((buf.count('\n') > 1) and buf.split('\n')[1].count(',') > 0):
            table = util.read_csv(buf, format='header+values-list', numbers=True)
            columns = [s.strip() for s in table[0]]
            table = table[1:]
            tzs = [s[4:] for s in columns if (s[5:] in ['ST', 'DT'] and s[4] in 'PMCE' and s[:4].lower() == 'time')]
            if tzs:
                tz = tzs[0]
            else:
                tz = 'UTC'
            for rownum, row in enumerate(table):
                try:
                    table[rownum] = [util.make_tz_aware(row[0], tz)] + row[1:]
                except ValueError:
                    pass
            dates = [row[-1] for row in table]
            if not all(isinstance(date, (datetime.datetime, pd.Timestamp)) for date in dates):
                dates = [row[0] for row in table]
            if len(columns) == len(table[0]):
                df0 = pd.DataFrame(table, columns=columns, index=dates)
                df = df.append(df0)
            elif verbosity >= 0:
                msg = "The number of columns in the 1st row of the table:\n    {}\n    doesn't match the number of column labels:\n    {}\n".format(
                    table[0], columns)
                msg += "Wunderground.com probably can't find the airport: {} ({})\n    or the date: {}\n    in its database.\n".format(
                    airport_code, location, day)
                msg += "Attempted a GET request using the URI:\n    {0}\n".format(url)
                warnings.warn(msg)
    try:
        df.to_csv(cache_path)
    except:
        if verbosity > 0 and use_cache:
            from traceback import print_exc
            print_exc()
            warnings.warn('Unable to write weather data to cache file at {}'.format(cache_path))
    return df


def api(feature='conditions', city='Portland', state='OR', key=None):
    """Use the wunderground API to get current conditions instead of scraping

    Please be kind and use your own key (they're FREE!):
    http://www.wunderground.com/weather/api/d/login.html

    References:
        http://www.wunderground.com/weather/api/d/terms.html

    Examples:
        >>> api('hurric', 'Boise', 'ID')  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        {u'currenthurricane': ...}}}

        >>> features = ('alerts astronomy conditions currenthurricane forecast forecast10day geolookup history hourly hourly10day ' +
        ...             'planner rawtide satellite tide webcams yesterday').split(' ')
        >> everything = [api(f, 'Portland') for f in features]
        >> js = api('alerts', 'Portland', 'OR')
        >> js = api('condit', 'Sacramento', 'CA')
        >> js = api('forecast', 'Mobile', 'AL')
        >> js = api('10day', 'Fairhope', 'AL')
        >> js = api('geo', 'Decatur', 'AL')
        >> js = api('hist', 'history', 'AL')
        >> js = api('astro')
    """
    features = ('alerts astronomy conditions currenthurricane forecast forecast10day geolookup history hourly hourly10day ' +
                'planner rawtide satellite tide webcams yesterday').split(' ')
    feature = util.fuzzy_get(features, feature)
    # Please be kind and use your own key (they're FREE!):
    # http://www.wunderground.com/weather/api/d/login.html
    key = key or env.get('WUNDERGROUND', None, verbosity=-1) or env.get('WUNDERGROUND_KEY', 'c45a86c2fc63f7d0', verbosity=-1)
    url = 'http://api.wunderground.com/api/{key}/{feature}/q/{state}/{city}.json'.format(
        key=key, feature=feature, state=state, city=city)
    return json.load(urllib.urlopen(url))


def daily(location='Fresno, CA', years=1, use_cache=True, verbosity=1):
    """Retrieve weather for the indicated airport code or 'City, ST' string.

    >>> df = daily('Camas, WA', verbosity=-1)
    >>> 365 <= len(df) <= 365 * 2 + 1
    True

    Sacramento data has gaps (airport KMCC):
        8/21/2013 is missing from 2013.
        Whole months are missing from 2014.
    >>> df = daily('Sacramento, CA', years=[2013], verbosity=-1)
    >>> 364 <= len(df) <= 365
    True
    >>> df.columns
    Index([u'PST', u'Max TemperatureF', u'Mean TemperatureF', u'Min TemperatureF', u'Max Dew PointF', u'MeanDew PointF', u'Min DewpointF', ...
    """
    this_year = datetime.date.today().year
    if isinstance(years, (int, float)):
        # current (incomplete) year doesn't count in total number of years
        # so 0 would return this calendar year's weather data
        years = np.arange(0, int(years) + 1)
    years = sorted(years)
    if not all(1900 <= yr <= this_year for yr in years):
        years = np.array([abs(yr) if (1900 <= abs(yr) <= this_year) else (this_year - abs(int(yr))) for yr in years])[::-1]

    airport_code = airport(location, default=location)

    # refresh the cache each time the start or end year changes
    cache_path = 'daily-{}-{}-{}.csv'.format(airport_code, years[0], years[-1])
    cache_path = os.path.join(CACHE_PATH, cache_path)
    if use_cache:
        try:
            return pd.DataFrame.from_csv(cache_path)
        except:
            pass

    df = pd.DataFrame()
    for year in years:
        url = ('http://www.wunderground.com/history/airport/{airport}/{yearstart}/1/1/' +
               'CustomHistory.html?dayend=31&monthend=12&yearend={yearend}' +
               '&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&MR=1&format=1').format(
            airport=airport_code,
            yearstart=year,
            yearend=year
            )
        if verbosity > 1:
            print('GETing *.CSV using "{0}"'.format(url))
        buf = urllib.urlopen(url).read()
        if verbosity > 0:
            N = buf.count('\n')
            M = (buf.count(',') + N) / float(N)
            print('Retrieved CSV for airport code "{}" with appox. {} lines and {} columns = {} cells.'.format(
                  airport_code, N, int(round(M)), int(round(M)) * N))
            if verbosity > 2:
                print(buf)
        table = util.read_csv(buf, format='header+values-list', numbers=True)
        # # clean up the last column (if it contains <br> tags)
        table = [util.strip_br(row) if len(row) > 1 else row for row in table]
        # numcols = max(len(row) for row in table)
        # table = [row for row in table if len(row) == numcols]
        columns = table.pop(0)
        tzs = [s for s in columns if (s[1:] in ['ST', 'DT'] and s[0] in 'PMCE')]
        dates = [float('nan')] * len(table)
        for i, row in enumerate(table):
            for j, value in enumerate(row):
                if not value and value is not None:
                    value = 0
                    continue
                if columns[j] in tzs:
                    table[i][j] = util.make_tz_aware(value, tz=columns[j])
                    if isinstance(table[i][j], datetime.datetime):
                        dates[i] = table[i][j]
                        continue
                try:
                    table[i][j] = float(value)
                    if not (table[i][j] % 1):
                        table[i][j] = int(table[i][j])
                except:
                    pass
        df0 = pd.DataFrame(table, columns=columns, index=dates)
        df = df.append(df0)

    if verbosity > 1:
        print(df)

    try:
        df.to_csv(cache_path)
    except:
        if verbosity > 0 and use_cache:
            from traceback import print_exc
            print_exc()
            warnings.warn('Unable to write weather data to cache file at {}'.format(cache_path))

    return df
