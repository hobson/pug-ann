import os
import urllib
import re
import datetime

import pandas as pd
np = pd.np
from pug.nlp import util

DATA_PATH = os.path.dirname(os.path.realpath(__file__))


# fresno = pd.DataFrame.from_csv(os.path.join(DATA_PATH, 'weather_fresno.csv'))

def airport(location='Fresno, CA', years=1, verbosity=1):
    """Retrieve weather for the indicated airport code or 'City, ST' string.

    >>> df = airport('Camas, WA', vebosity=0)
    >>> 365 <= len(df) <= 365 * 2 + 1
    True

    Sacramento data has gaps (airport KMCC), 2013 is missing 8/21/2013.
    Whole months are missing from 2014
    >>> df = airport('Sacramento, CA', years=[2013], verbosity=0)
    >>> len(df)
    364
    >>> df.columns
    Index([u'PST', u'Max TemperatureF', u'Mean TemperatureF', u'Min TemperatureF', u'Max Dew PointF', u'MeanDew PointF', u'Min DewpointF', u'Max Humidity', u' Mean Humidity', u' Min Humidity', u' Max Sea Level PressureIn', u' Mean Sea Level PressureIn', u' Min Sea Level PressureIn', u' Max VisibilityMiles', u' Mean VisibilityMiles', u' Min VisibilityMiles', u' Max Wind SpeedMPH', u' Mean Wind SpeedMPH', u' Max Gust SpeedMPH', u'PrecipitationIn', u' CloudCover', u' Events', u' WindDirDegrees'], dtype='object')
    """
    this_year = datetime.date.today().year
    if isinstance(years, (int, float)):
        # current (incomplete) year doesn't count in total number of years
        # so 0 would return this calendar year's weather data
        years = np.arange(0, int(years) + 1)
    years = sorted(years)
    if not all(1900 <= yr <= this_year for yr in years):
        years = np.array([abs(yr) if (1900 <= abs(yr) <= this_year) else (this_year - abs(int(yr))) for yr in years])[::-1]

    airport_code = airport.locations.get(location, location)
    df = pd.DataFrame()
    for year in years:
        url = ( 'http://www.wunderground.com/history/airport/{airport}/{yearstart}/1/1/'
                'CustomHistory.html?dayend=31&monthend=12&yearend={yearend}'
                '&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&MR=1&format=1').format(
               airport=airport_code, 
               yearstart=year,
               yearend=year)
        if verbosity > 1:
            print('GETing *.CSV using "{0}"'.format(url))
        buf = urllib.urlopen(url).read()
        if verbosity > 0:
            N = buf.count('\n')
            M = (buf.count(',') + N) / float(N)
            print('Retrieved CSV for airport code "{}" with appox. {} lines and {} columns = {} cells.'.format(
                  airport_code, N, int(round(M)), int(round(M)) * N))

        table = [s.strip() for s in (row.split(',') for row in buf.split('\n') if len(row)>1)]
        # clean up the last column (if it contains <br> tags)
        table = [row[:-1] + [re.sub(r'\s*<br\s*[/]?>\s*$','', row[-1])] for row in table]
        numcols = max(len(row) for row in table)
        table = [row for row in table if len(row) == numcols]
        columns = table.pop(0)
        tzs = [s for s in columns if (s[1:] in ['ST', 'DT'] and s[0] in 'PMCE')]
        dates = [float('nan')] * len(table)
        for i, row in enumerate(table):
            for j, col in enumerate(row):
                if not col and col != None:
                    col = 0
                    continue
                if columns[j] in tzs:
                    table[i][j] = util.make_tz_aware(col, tz=columns[j])
                    if isinstance(table[i][j], datetime.datetime):
                        dates[i] = table[i][j]
                        continue
                try:
                    table[i][j] = int(col)
                except:
                    try:
                        table[i][j] = float(col)
                    except:
                        pass
        df0 = pd.DataFrame(table, columns=columns, index=dates)
        df = df.append(df0)

    if verbosity > 1:
        print(df)
    return df
airport.locations = dict([(str(city) + ', ' + str(region)[-2:], str(ident)) for city, region, ident in pd.DataFrame.from_csv(os.path.join(DATA_PATH, 'airports.csv')).sort(ascending=False)[['municipality', 'iso_region', 'ident']].values])

city = airport
