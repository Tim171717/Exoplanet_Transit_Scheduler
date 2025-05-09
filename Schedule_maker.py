import numpy as np
import pandas as pd
from PyAstronomy import pyasl
import datetime
from astral import LocationInfo
from astral.sun import sun
import csv
import io
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, solar_system_ephemeris
from scipy.optimize import root_scalar
import ephem
from zoneinfo import ZoneInfo
import json
import urllib.request

from SNR_calculator import get_exptime




possible_headers = ['Planet', 'Planet_name', 'name', 'Name',
                    'dec_j2000', 'Declination', 'Dec',
                    'ra_j2000', 'Right Ascension', 'RA',
                    't0_bjd_tdb', 'ephemeris mid_time', 'ephemeris mid_time [BJD]', 'epoch',
                    't0_unc', 'ephemeris mid_time uncertainty', 'epoch_uncertainty',
                    'period_days', 'ephemeris period', 'ephemeris period [days]', 'period',
                    'period_unc', 'ephemeris period uncertainty', 'period_uncertainty',
                    'duration_hours', 'duration', 'duration [hours]',
                    'min_telescope_inches', 'minimmum size of telescope [inches]', 'minimmum size of telescope'
                    ]



def priority(char):
    if char == 'campaign': return 1
    if char == 'ttvs': return 2
    if char == 'alert': return 3
    if char == 'high': return 4
    if char == 'medium': return 5
    if char == 'low': return 6
    if char == 'no priority': return 7


def BJDtoJD(bjd_tdb, dec, ra, location):
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))

    # Define the function whose root we want to find
    def f(jd_utc_val):
        time = Time(jd_utc_val, format='jd', scale='utc', location=location)
        with solar_system_ephemeris.set('de432s'):
            ltt_bary = time.light_travel_time(coord)
        bjd_calc = time.tdb + ltt_bary
        return bjd_calc.jd - bjd_tdb  # We want this to be zero

    # Initial guess: BJD_TDB is very close to JD_UTC
    result = root_scalar(f, bracket=[bjd_tdb - 0.01, bjd_tdb + 0.01], method='brentq')

    if not result.converged:
        raise RuntimeError("Root finding for BJD→JD did not converge.")

    return result.root  # JD in UTC scale


def isinTransit(transittime, period, duration, dusk, dawn, dec, ra, city_astropy, transittime_unc, period_unc):
    counter = 0
    while transittime + counter * period < dusk + duration / 48 + 1 / 72:
        counter += 1
    transittime += counter * period
    if transittime + duration / 48 + 1 / 72 <= dawn:
        starttime = max(BJDtoJD(transittime - duration / 48 - np.sqrt(counter * period_unc ** 2 + transittime_unc ** 2),
                                dec, ra, city_astropy) - 1 / 72, dusk)
        endtime = min(BJDtoJD(transittime + duration / 48 + np.sqrt(counter * period_unc ** 2 + transittime_unc ** 2),
                              dec, ra, city_astropy) + 1 / 72, dawn)
        transittime = BJDtoJD(transittime, dec, ra, city_astropy)
        return (True,
                (pyasl.daycnv(starttime, mode='dt')),
                (pyasl.daycnv(transittime,mode='dt')),
                (pyasl.daycnv(endtime,mode='dt')))
    else:
        return False, 1, 1, 1


def Starisvisible(dec, ra, city_astropy, starttime, endtime, alt_limit, moon_distance):
    # alt_limit = Max's function
    object = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    # first check at the end
    astropy_time = Time(endtime)
    altaz_frame = AltAz(obstime=astropy_time, location=city_astropy)
    object_altaz = object.transform_to(altaz_frame)
    if object_altaz.alt.deg < alt_limit:
        return False

    time = starttime
    while time < endtime:
        astropy_time = Time(time)

        # Transform object coordinates to AltAz
        altaz_frame = AltAz(obstime=astropy_time, location=city_astropy)
        object_altaz = object.transform_to(altaz_frame)

        # Check altitude
        if object_altaz.alt.deg < alt_limit:
            return False

        # Get Moon position and compute separation
        moon_coord = get_body('moon', astropy_time, location=city_astropy).transform_to('icrs')
        if object.separation(moon_coord).deg < moon_distance:
            return False

        time += datetime.timedelta(minutes=10)
        # time += datetime.timedelta(minutes=15)

    return True


def obs_startend(dec, ra, city_astropy, starttime, endtime, alt_limit, moon_distance, add_time, dusk, dawn):
    object = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    key = np.zeros(5) + 1
    obsstart = starttime
    if dusk > starttime - datetime.timedelta(minutes=add_time - 20):
        timing1 = dusk
        key[0] = 0
    else:
        timing1 = starttime - datetime.timedelta(minutes=add_time - 20)
    while obsstart > timing1:
        astropy_time = Time(obsstart)

        # Transform object coordinates to AltAz
        altaz_frame = AltAz(obstime=astropy_time, location=city_astropy)
        object_altaz = object.transform_to(altaz_frame)

        # Check altitude
        if object_altaz.alt.deg < alt_limit:
            key[0] = 0
            break

        # Get Moon position and compute separation
        moon_coord = get_body('moon', astropy_time, location=city_astropy).transform_to('icrs')
        if object.separation(moon_coord).deg < moon_distance:
            key[0] = 0
            break
        obsstart -= datetime.timedelta(minutes=1)
        # obsstart -= datetime.timedelta(minutes=5)

    obsend = endtime
    if dawn < endtime + datetime.timedelta(minutes=add_time - 20):
        timing2 = dawn
        key[4] = 0
    else:
        timing2 = endtime + datetime.timedelta(minutes=add_time - 20)
    while obsend < timing2:
        astropy_time = Time(obsend)

        # Transform object coordinates to AltAz
        altaz_frame = AltAz(obstime=astropy_time, location=city_astropy)
        object_altaz = object.transform_to(altaz_frame)

        # Check altitude
        if object_altaz.alt.deg < alt_limit:
            key[4] = 0
            break

        # Get Moon position and compute separation
        moon_coord = get_body('moon', astropy_time, location=city_astropy).transform_to('icrs')
        if object.separation(moon_coord).deg < moon_distance:
            key[4] = 0
            break
        obsend += datetime.timedelta(minutes=1)
        # obsend += datetime.timedelta(minutes=5)
    return max(obsstart, timing1), min(obsend, timing2), key


def DuskandDawn(city_ephem, date):
    city_ephem.date = date
    dusk = city_ephem.next_setting(ephem.Sun()).datetime()
    city_ephem.date = date + datetime.timedelta(days=1)
    dawn = city_ephem.next_rising(ephem.Sun()).datetime()
    return dusk, dawn


def otherTargets(starttime1, endtime1, starttime2, endtime2):
    return starttime1 > endtime2 or starttime2 > endtime1


def Get_availabilities(date, city, elevation, df, alt_limit=30, moon_distance=30, add_time=60,
                       dusk_type='Nautical', aperture_size=20, progress_callback=None):
    city_astropy = EarthLocation(lat=city.latitude, lon=city.longitude, height=elevation * u.m)
    city_ephem = ephem.Observer()
    city_ephem.pressure = 0
    city_ephem.lat, city_ephem.lon = str(city.latitude), str(city.longitude)
    if dusk_type == 'Astronomical':
        city_ephem.horizon = '-18'
    elif dusk_type == 'Nautical':
        city_ephem.horizon = '-12'
    else:
        city_ephem.horizon = '-6'

    found_transits = []

    headers = []
    for header in possible_headers:
        if header in df.columns:
            headers.append(header)
    if len(headers) < 8:
        raise ValueError("Invalid import catalog. Please check the headers.")
    total_rows = len(df)

    dusk, dawn = DuskandDawn(city_ephem, date)
    for i, row in df.iterrows():
        name = row[headers[0]]
        dec = row[headers[1]]
        ra = row[headers[2]]
        transittime = float(row[headers[3]])
        transittime_unc = float(row[headers[4]])
        period = float(row[headers[5]])
        period_unc = float(row[headers[6]])
        duration = float(row[headers[7]])

        aperture = True
        if possible_headers[-1] in headers or possible_headers[-2] in headers:
            aperture_rec = float(row[headers[-1]])
            aperture = aperture_size >= aperture_rec

        if aperture:
            if np.isnan(transittime_unc): transittime_unc = 0
            if np.isnan(period_unc): period_unc = 0

            istransiting, starttime, midtime, endtime = isinTransit(transittime, period, duration, pyasl.jdcnv(dusk),
                                                                    pyasl.jdcnv(dawn), dec, ra, city_astropy,
                                                                    transittime_unc, period_unc)
            if istransiting:
                if Starisvisible(dec, ra, city_astropy, starttime, endtime, alt_limit, moon_distance):
                    obsstart, obsend, key = obs_startend(dec, ra, city_astropy, starttime, endtime, alt_limit,
                                                         moon_distance, add_time, dusk, dawn)
                    planet_info = [name,
                                   dec,
                                   ra,
                                   obsstart.replace(tzinfo=datetime.timezone.utc),
                                   (starttime + datetime.timedelta(minutes=20)).replace(tzinfo=datetime.timezone.utc),
                                   midtime.replace(tzinfo=datetime.timezone.utc),
                                   (endtime - datetime.timedelta(minutes=20)).replace(tzinfo=datetime.timezone.utc),
                                   obsend.replace(tzinfo=datetime.timezone.utc),
                                   key]
                    if 'priority' in df.columns:
                        planet_info.append(row['priority'])
                    else:
                        planet_info.append('no priority')

                    found_transits.append(planet_info)

            if progress_callback is not None:
                progress_callback(int((i + 1) / total_rows * 100))

    found_transits.sort(key=lambda x: x[3])
    return found_transits


def select_schedule(possible_transits):
    n = len(possible_transits)
    infos = np.zeros((n, 4))
    names = []
    for i in range(n):
        infos[i,0] = priority(possible_transits[i][9])
        infos[i,1] = pyasl.jdcnv(possible_transits[i][3])
        infos[i,2] = pyasl.jdcnv(possible_transits[i][7])
        infos[i,3] = min(possible_transits[i][8])
        names.append(possible_transits[i][0])

    prio = np.where(infos[:, 0] == np.min(infos[:, 0]), 1, 0)
    if np.sum(prio) == 1:
        selection = np.argmax(prio)
        otherobs = []
        for i in range(n):
            if otherTargets(infos[selection, 1], infos[selection, 2], infos[i, 1], infos[i, 2]):
                otherobs.append(possible_transits[i])
        selection = [selection]
        if len(otherobs) >= 1:
            addition = select_schedule(otherobs)
            for add in addition:
                selection.append(names.index(otherobs[add][0]))
    else:
        multipleobs = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if otherTargets(infos[i, 1], infos[i, 2], infos[j, 1], infos[j, 2]):
                    multipleobs[i] += 1
        multipleobs = multipleobs * prio
        multipleobs = np.where(multipleobs == np.max(multipleobs), 1, 0)
        if sum(multipleobs) == 1:
            selection = np.argmax(multipleobs)
            otherobs = []
            for i in range(n):
                if otherTargets(infos[selection, 1], infos[selection, 2], infos[i, 1], infos[i, 2]):
                    otherobs.append(possible_transits[i])
            selection = [selection]
            if len(otherobs) >= 1:
                addition = select_schedule(otherobs)
                for add in addition:
                    selection.append(names.index(otherobs[add][0]))
        else:
            key = np.zeros(n)
            for i in range(n):
                key[i] = infos[i, 3]
            key = key * multipleobs
            key = np.where(key == np.max(key), 1, 0)
            selection = np.argmax(key)
            otherobs = []
            for i in range(n):
                if otherTargets(infos[selection, 1], infos[selection, 2], infos[i, 1], infos[i, 2]):
                    otherobs.append(possible_transits[i])
            selection = [selection]
            if len(otherobs) >= 1:
                addition = select_schedule(otherobs)
                for add in addition:
                    selection.append(names.index(otherobs[add][0]))

    return sorted(selection)



def write_schedule(selected_transits, date, city, max_exp=120, bin=4,device_name='camera_hpp',tzone='UTC'):
    selected_transits.sort(key=lambda x: x[3])
    city_ephem = ephem.Observer()
    city_ephem.pressure = 0
    city_ephem.lat, city_ephem.lon = str(city.latitude), str(city.longitude)
    city_ephem.horizon = '-12'
    sunset = sun(city.observer, date=date)['sunset']
    sunrise = sun(city.observer, date=date + datetime.timedelta(days=1))['sunrise']
    dusk, dawn = DuskandDawn(city_ephem, date)
    dusk = dusk.replace(tzinfo=datetime.timezone.utc)
    dawn = dawn.replace(tzinfo=datetime.timezone.utc)
    dusk = min(dusk, selected_transits[0][3])
    dawn = max(dawn, selected_transits[-1][7])
    used_exptimes = []
    filters = []
    obsplan = [{'device_type': 'Camera', 'device_name': device_name, 'action_type': 'open',
                'action_value': {}, 'start_time': sunset.astimezone(ZoneInfo(tzone)).strftime('%Y-%m-%d %H:%M:%S.%f'),
                'end_time': (sunrise.astimezone(ZoneInfo(tzone))).strftime('%Y-%m-%d %H:%M:%S.%f')}]


    for transit in selected_transits:
        coord = SkyCoord(transit[2], transit[1], unit=(u.hourangle, u.deg))
        filter, exposure_time = get_exptime(transit[0][:-1], coord, transit[5].strftime('%Y-%m-%d %H:%M:%S'),
                                            max_exp=max_exp, binning=bin)
        obsplan.append({'device_type': 'Camera',
                        'device_name': device_name,
                        'action_type': 'object',
                        'action_value': str({'object': transit[0],
                                             'filter': filter,
                                             'ra': float(coord.ra.deg),
                                             'dec': float(coord.dec.deg),
                                             'exptime': float(exposure_time),
                                             'guiding': True,
                                             'pointing': False,
                                             'bin': bin}),
                        'start_time': transit[3].astimezone(ZoneInfo(tzone)).strftime('%Y-%m-%d %H:%M:%S.%f'),
                        'end_time': transit[7].astimezone(ZoneInfo(tzone)).strftime('%Y-%m-%d %H:%M:%S.%f')})
        used_exptimes.append(exposure_time)
        filters.append(filter)

    filters = np.unique(filters).tolist()
    nnf = (np.zeros(len(filters), dtype=int) + 20//len(filters)).tolist()
    obsplan.insert(1, {'device_type': 'Camera', 'device_name': device_name, 'action_type': 'flats',
                       'action_value': str({'filter': filters, 'n': nnf, 'bin': bin}),
                       'start_time': (sunset.astimezone(ZoneInfo(tzone)) + datetime.timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S.%f'),
                       'end_time': dusk.astimezone(ZoneInfo(tzone)).strftime('%Y-%m-%d %H:%M:%S.%f')})
    obsplan.append({'device_type': 'Camera',
                    'device_name': device_name,
                    'action_type': 'flats',
                    'action_value': str({'filter': filters,
                                         'n': nnf,
                                         'bin': bin}),
                    'start_time': dawn.astimezone(ZoneInfo(tzone)).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'end_time': sunrise.astimezone(ZoneInfo(tzone)).strftime('%Y-%m-%d %H:%M:%S.%f')})

    exptimes = np.linspace(0, max_exp, 8, dtype=int)
    deleting = []
    for ue in used_exptimes:
        deleting.append(np.argmin(abs(exptimes - ue)))
    exptimes = np.delete(exptimes, np.unique(deleting))
    exptimes = (np.unique(np.append(exptimes, used_exptimes))).tolist()
    nn = (np.zeros(len(exptimes), dtype=int) + 10).tolist()
    caltime = np.dot(exptimes, nn) + 0.7 * np.sum(nn)
    obsplan.append({'device_type': 'Camera',
                    'device_name': device_name,
                    'action_type': 'close',
                    'action_value': "{}",
                    'start_time': sunrise.astimezone(ZoneInfo(tzone)).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'end_time': (sunrise.astimezone(ZoneInfo(tzone)) + datetime.timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S.%f')})
    obsplan.append({'device_type': 'Camera',
                    'device_name': device_name,
                    'action_type': 'calibration',
                    'action_value': str({'exptime': exptimes, 'n': nn, 'bin': bin}),
                    'start_time': (sunrise.astimezone(ZoneInfo(tzone)) + datetime.timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'end_time': (sunrise.astimezone(ZoneInfo(tzone)) + datetime.timedelta(minutes=5, seconds=caltime)).strftime('%Y-%m-%d %H:%M:%S.%f')})

    output = io.StringIO()
    fieldnames = ['device_type', 'device_name', 'action_type', 'action_value', 'start_time', 'end_time']
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(obsplan)

    return output.getvalue()


def update_ExoClockCatalog():
    # Load ExoClock planet data
    url = 'https://www.exoclock.space/database/planets_json'
    response = urllib.request.urlopen(url)
    exoclock_planets = json.loads(response.read())

    # Define output CSV file and fieldnames
    output_file = 'ExoplanetCatalog_Exoclock.csv'
    fieldnames = [
        'name', 'priority', 'dec_j2000', 'ra_j2000', 'v_mag',
        't0_bjd_tdb', 't0_unc', 'period_days', 'period_unc',
        'min_telescope_inches', 'duration_hours'
    ]

    # Write to CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for planet in exoclock_planets.values():
            writer.writerow({
                'name': planet['name'],
                'priority': planet['priority'],
                'dec_j2000': planet['dec_j2000'],
                'ra_j2000': planet['ra_j2000'],
                'v_mag': planet['v_mag'],
                't0_bjd_tdb': planet['t0_bjd_tdb'],
                't0_unc': planet['t0_unc'],
                'period_days': planet['period_days'],
                'period_unc': planet['period_unc'],
                'min_telescope_inches': planet['min_telescope_inches'],
                #'depth_mmag': planet['depth_mmag'],
                'duration_hours': planet['duration_hours'],
            })

    print(f"Catalog saved to {output_file}")


if __name__ == '__main__':
    date = datetime.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    # date = datetime.datetime(2025,4,6)
    city = LocationInfo("Zurich", "Switzerland", "UTC", 47.2, 8.3)
    df = pd.read_excel('ExoplanetCatalog_Exoclock.xlsx')

    found_transits = Get_availabilities(date, city, 575, df)

    best_indices = select_schedule(found_transits)
    best_transits = [found_transits[i] for i in best_indices]

    csv_string = write_schedule(best_transits, date, city)
    csvname = 'Schedules/schedule_' + date.strftime('%Y-%m-%d') + '.csv'
    with open(csvname, "w", newline='', encoding='utf-8') as f:
        f.write(csv_string)