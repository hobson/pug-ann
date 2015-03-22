import os
import urllib
from StringIO import StringIO
import re
import datetime

import pandas as pd
np = pd.np
from pug.nlp import util

DATA_PATH = os.path.dirname(os.path.realpath(__file__))


# fresno = pd.DataFrame.from_csv(os.path.join(DATA_PATH, 'weather_fresno.csv'))

def airport(location='Fresno, CA', years=1, verbosity=1):
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

        try:
            df0 = pd.DataFrame.from_csv(StringIO(buf))
        except IndexError:
            table = [row.split(',') for row in buf.split('\n') if len(row)>1]
            table = [row[:-1] + [re.sub(r'\s*<br\s*[/]?>\s*$','', row[-1])] for row in table]
            for i, row in enumerate(table):
                try:
                    row[-1] = int(row[-1])
                except:
                    try:
                        row[-1] = float(row[-1])
                    except:
                        pass
            numcols = max(len(row) for row in table)
            table = [row for row in table if len(row) == numcols]
            df0 = pd.DataFrame(table)
            df0.columns = [str(label) for label in df0.iloc[0].values]
            df0 = df0.iloc[1:]
        df0.columns = [label.strip() for label in df0.columns]

        # if verbosity > 0:
        #     print(df0.describe())
        # columns = df0.columns.values
        # columns[-1] = re.sub(r'<br\s*[/]?>','', columns[-1])
        # df0.columns = columns

        df = df.append(df0)

    tzs = [s for s in df.columns if (s[1:] in ['ST', 'DT'] and s[0] in 'PMCE')]
    df['Date'] = df.index
    if len(tzs) > 0:
        df['Date'] = [util.make_datetime(obj, tz=tzs[0]) if obj else float('nan') for obj in df[tzs[0]]]
        if len(tzs) == 2:
            nanmask = df.Date.isnull()
            df.Date[nanmask] = [util.make_datetime(obj, tz=tzs[0]) if obj else float('nan') for obj in df[tzs[1]][nanmask]]
    df.drop_duplicates(cols=['Date'], take_last=True, inplace=True)
    if not any(df.Date.isnull()):
        df.index = pd.DatetimeIndex(df.Date)


    if verbosity > 1:
        print(df)
    return df
airport.locations = dict([(str(city) + ', ' + str(region)[-2:], str(ident)) for city, region, ident in pd.DataFrame.from_csv(os.path.join(DATA_PATH, 'airports.csv')).sort(ascending=False)[['municipality', 'iso_region', 'ident']].values])

city = airport