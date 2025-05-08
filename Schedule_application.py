import numpy as np
import pandas as pd
from PyAstronomy import pyasl
import datetime
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder
from astral import LocationInfo
from astral.sun import sun
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, solar_system_ephemeris
from scipy.optimize import root_scalar
import ephem
import streamlit as st
from geopy.geocoders import Nominatim
import requests
import io
import csv


from SNR_calculator import get_exptime

possible_headers = ['Planet', 'Planet_name', 'name', 'Name',
                    'Declination', 'Dec',
                    'Right Ascension', 'RA',
                    'ephemeris mid_time', 'ephemeris mid_time [BJD]', 'epoch',
                    'ephemeris mid_time uncertainty', 'epoch_uncertainty',
                    'ephemeris period', 'ephemeris period [days]', 'period',
                    'ephemeris period uncertainty', 'period_uncertainty',
                    'duration', 'duration [hours]',
                    'minimmum size of telescope [inches]', 'minimmum size of telescope'
                    ]

colormapping = {
    'campaign': 'turquoise',
    'ttvs': 'darkturquoise',
    'alert': 'blueviolet',
    'high': 'red',
    'medium': 'orange',
    'low': 'limegreen'
}


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
    key = np.zeros(5)
    obsstart = starttime
    if dusk > starttime - datetime.timedelta(minutes=add_time - 20):
        timing1 = dusk
        key[0] = 1
    else:
        timing1 = starttime - datetime.timedelta(minutes=add_time - 20)
    while obsstart > timing1:
        astropy_time = Time(obsstart)

        # Transform object coordinates to AltAz
        altaz_frame = AltAz(obstime=astropy_time, location=city_astropy)
        object_altaz = object.transform_to(altaz_frame)

        # Check altitude
        if object_altaz.alt.deg < alt_limit:
            key[0] = 1
            break

        # Get Moon position and compute separation
        moon_coord = get_body('moon', astropy_time, location=city_astropy).transform_to('icrs')
        if object.separation(moon_coord).deg < moon_distance:
            key[0] = 1
            break
        obsstart -= datetime.timedelta(minutes=1)
        # obsstart -= datetime.timedelta(minutes=5)

    obsend = endtime
    if dawn < endtime + datetime.timedelta(minutes=add_time - 20):
        timing2 = dawn
        key[4] = 1
    else:
        timing2 = endtime + datetime.timedelta(minutes=add_time - 20)
    while obsend < timing2:
        astropy_time = Time(obsend)

        # Transform object coordinates to AltAz
        altaz_frame = AltAz(obstime=astropy_time, location=city_astropy)
        object_altaz = object.transform_to(altaz_frame)

        # Check altitude
        if object_altaz.alt.deg < alt_limit:
            key[4] = 1
            break

        # Get Moon position and compute separation
        moon_coord = get_body('moon', astropy_time, location=city_astropy).transform_to('icrs')
        if object.separation(moon_coord).deg < moon_distance:
            key[4] = 1
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


def Get_availabilities(date, city, elevation, alt_limit, add_time, dusk_type, df, aperture_size,
                       progress_callback=None):
    city_astropy = EarthLocation(lat=city.latitude, lon=city.longitude, height=elevation * u.m)
    city_ephem = ephem.Observer()
    city_ephem.pressure = 0;
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


def write_schedule(selected_transits, date, city, max_exp=120, bin=4,device_name='camera_hpp',tzone='UTC'):
    selected_transits.sort(key=lambda x: x[3])
    city_ephem = ephem.Observer()
    city_ephem.pressure = 0;
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
    nn = (np.zeros(len(filters), dtype=int) + 20//len(filters)).tolist()
    obsplan.insert(1, {'device_type': 'Camera', 'device_name': device_name, 'action_type': 'flats',
                       'action_value': str({'filter': filters, 'n': nn, 'bin': bin}),
                       'start_time': (sunset.astimezone(ZoneInfo(tzone)) + datetime.timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S.%f'),
                       'end_time': dusk.astimezone(ZoneInfo(tzone)).strftime('%Y-%m-%d %H:%M:%S.%f')})

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
                    'action_type': 'flats',
                    'action_value': str({'filter': [filter],
                                         'n': [20],
                                         'bin': bin}),
                    'start_time': dawn.astimezone(ZoneInfo(tzone)).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'end_time': sunrise.astimezone(ZoneInfo(tzone)).strftime('%Y-%m-%d %H:%M:%S.%f')})
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


# --- PAGE CONFIG ---
st.set_page_config(page_title="Transit Scheduler", layout="centered")

st.title("🪐 Exoplanet Transit Scheduler")

# Create two columns: left for inputs, right for output
col_left, col_right = st.columns([1, 2])  # Adjust widths if needed

with col_left:
    # --- INPUTS ---
    st.subheader("🔭 Observation Constraints")

    # Date selection
    date = st.date_input("📅 Select a date", value=datetime.date.today())
    st.session_state['date'] = date

    # --- LOCATION SEARCH ---
    location_query = st.text_input("📍 Location Selection", value='ETH Hönggerberg')

    location = None
    city = None
    elevation = None
    uploaded_file = None

    # Additional inputs
    add_time = st.slider('⏳ Additional Observation Time (min)', min_value=20, max_value=60, value=60)
    alt_limit = st.number_input("🌌 Altitude Limit (°)", min_value=0.0, max_value=90.0, value=30.0, step=1.0)
    moon_distance = st.number_input("🌙 Min Moon Distance (°)", min_value=0.0, max_value=180.0, value=30.0, step=1.0)
    aperture_size = st.number_input("📏 Aperture (inch)", min_value=0.0, max_value=150.0, value=20.0, step=1.0)
    dusk_type = st.selectbox("🌅 Dusk Type", options=["Astronomical", "Nautical", "Civil"])

    # --- CATALOG SELECTION ---
    catalog_options = {
        "🪐 ExoClock Database": "ExoClock",
        "🚀 NASA Exoplanet Archive": "NASA",
        "📂 Custom Catalog": "Custom"
    }

    selected_label = st.radio("Select a Catalog:", list(catalog_options.keys()))
    catalog_selected = catalog_options[selected_label]

    # --- LOAD CATALOG BASED ON SELECTION ---
    df = None
    uploaded_file = None

    if catalog_selected == "ExoClock":
        uploaded_file = 'ExoplanetCatalog_Exoclock.xlsx'
        df = pd.read_excel(uploaded_file, engine='openpyxl')

    elif catalog_selected == "NASA":
        uploaded_file = 'NASA_Exoplanet_Archive_database.csv'
        df = pd.read_csv(uploaded_file)

    elif catalog_selected == "Custom":
        uploaded_file = st.file_uploader("📂 Upload Custom Catalog", type=["csv", "xlsx"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, engine='openpyxl')

    # Submit button and trigger calculation + store results in session state

    #additional inputs
    with st.expander("Additional Settings"):
        # Add inputs inside the expander
        device_name = st.text_input("Device Name: ", value='camera_hpp')
        st.session_state['device_name'] = device_name
        max_exp = st.number_input("Maximal Exposure Time", value=120, min_value=0, step=10)
        st.session_state['max_exp'] = max_exp
        bin = st.number_input("Binning", value=4, min_value=0, step=1)
        st.session_state['bin'] = bin
        display = st.checkbox('Display Transits in UTC', value=False)
        st.session_state['display'] = display
        utc = st.checkbox('Write Schedule in UTC', value=True)
        st.session_state['utc'] = utc

    if st.button("Submit"):
        if location_query:
            geolocator = Nominatim(user_agent="streamlit-location-search")
            try:
                location = geolocator.geocode(location_query)
            except Exception as e:
                st.error(f"Geocoding error: {e}")

            if location:
                city_info = location.address.split(",")[0]
                country = location.address.split(",")[-1].strip()

                # Get elevation (optional)
                elev_url = f"https://api.open-elevation.com/api/v1/lookup?locations={location.latitude},{location.longitude}"
                response = requests.get(elev_url)
                if response.status_code == 200:
                    results = response.json()['results']
                    elevation = results[0]['elevation']
                else:
                    st.warning("Could not get elevation data.")
                    elevation = None
                elevation = elevation if elevation is not None else 0

                city = LocationInfo(name=city_info, region=country, timezone='UTC',
                                    latitude=location.latitude, longitude=location.longitude)
                if display is False:
                    tf = TimezoneFinder()
                    timezone = tf.timezone_at(lat=location.latitude, lng=location.longitude)
                else: timezone = 'UTC'
                st.session_state['city'] = city
                st.session_state['elevation'] = elevation
                st.session_state['timezone'] = timezone

            else:
                st.error("Location not found. Please try another query.")

        if uploaded_file is None:
            st.error("Please upload a catalog.")

        if uploaded_file and city is not None:
            with col_right:
                progress_bar = st.progress(0)
                status_text = st.empty()


                def progress_update(pct):
                    progress_bar.progress(pct)


                found_transits = Get_availabilities(date, city, elevation, alt_limit, add_time, dusk_type, df,
                                                    aperture_size,
                                                    progress_callback=progress_update)

                progress_bar.empty()  # remove progress bar after done

                st.session_state['found_transits'] = found_transits
                st.session_state['selected_transits'] = []  # reset selections

                # Reset all transit checkboxes to False
                for i in range(len(found_transits)):
                    st.session_state[f"transit_{i}"] = False

    # Initialize session state for transits if not present
    if 'found_transits' not in st.session_state:
        st.session_state['found_transits'] = []

with col_right:
    # --- OUTPUT: Show found transits ---
    found_transits = st.session_state['found_transits']

    if found_transits:
        city = st.session_state.get('city')
        timezone_str = st.session_state.get('timezone')
        elevation = st.session_state.get('elevation')

        tzinfo = f"🕒 Timezone: **{timezone_str}**"
        if timezone_str != 'UTC':
            date = datetime.datetime.combine(st.session_state.get('date'), datetime.datetime.min.time())
            date = (date + datetime.timedelta(hours=12)).replace(tzinfo=datetime.timezone.utc)
            tz = ZoneInfo(timezone_str)
            now = date.astimezone(tz)
            offset_sec = tz.utcoffset(now).total_seconds()
            hours = int(offset_sec // 3600)
            minutes = int((abs(offset_sec) % 3600) // 60)
            sign = "+" if offset_sec >= 0 else "-"
            utc_offset = f"UTC{sign}{abs(hours):02d}:{minutes:02d}"
            tzinfo += f" **({utc_offset})**"

        location_info = f"**{city.name}/{city.region}, {elevation} m**"

        st.success(f"✅ Found {len(found_transits)} observable transits for {location_info}")
        st.markdown(tzinfo)

        # Initialize selected_transits if not present
        if 'selected_transits' not in st.session_state:
            st.session_state['selected_transits'] = []

        # Step 1: Gather checkbox states from session state or default to False
        checked_states = []
        for i, t in enumerate(found_transits):
            checked_states.append(st.session_state.get(f"transit_{i}", False))

        # Step 2: Build new selected list from checked_states
        new_selected = [t for i, t in enumerate(found_transits) if checked_states[i]]

        # Step 3: Render transits with compatibility check and disabling incompatible ones
        for i, t in enumerate(found_transits):
            name, dec, ra, obsstart, start, mid, end, obsend, key, priority = t

            # Check if this transit overlaps with any *other* selected transit
            is_compatible = True
            for sel in new_selected:
                if sel == t:
                    continue
                sel_start, sel_end = sel[3], sel[7]
                if not otherTargets(obsstart, obsend, sel_start, sel_end):
                    is_compatible = False
                    break

            # Show disabled checkbox if not compatible and not already selected
            disabled = (not is_compatible) and (t not in new_selected)

            # Render checkbox with disabled state
            checked = st.checkbox(
                f"Select", key=f"transit_{i}", value=(t in new_selected), disabled=disabled
            )
            if disabled:
                colors = ['#FFA07A' if i == 1 else '#bbb' for i in key]
            else:
                colors = ['red' if i == 1 else 'black' for i in key]

            timezone = st.session_state['timezone']

            priority_tag = ""
            if priority.lower() != "no priority":
                priority_color = colormapping.get(priority.lower(), "#cccccc")
                priority_tag = (
                    f"<span style='"
                    f"background-color: {priority_color}; "
                    f"color: white; "
                    f"padding: 2px 6px; "
                    f"border-radius: 6px; "
                    f"font-size: 0.85em; "
                    f"margin-left: 8px;'>"
                    f"{priority}</span>"
                )

            box_style = (
                "background-color: #f0f0f0; color: #999;" if disabled else "background-color: #f9f9f9; color: #000;"
            )
            box_content = f"""
            <div style="
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 15px;
                box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
                {box_style};
            ">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <strong>{name}{priority_tag}</strong>
                    {"<span style='color: #FF6347;'>⚠ Overlaps with selected transit</span>"
            if disabled else "<span style='visibility: hidden;'>.</span>"}
                </div>
                    <div style="margin-top: 5px;">
                    <span style="color:{colors[0]};" title="Observation Start">{obsstart.astimezone(ZoneInfo(timezone)).strftime('%H:%M')}</span> &nbsp;–&nbsp;
                    <span style="color:{colors[1]};" title="Transit Start">{start.astimezone(ZoneInfo(timezone)).strftime('%H:%M')}</span> &nbsp;–&nbsp;
                    <span style="color:{colors[2]};" title="Transit Mid-Time">{mid.astimezone(ZoneInfo(timezone)).strftime('%H:%M')}</span> &nbsp;–&nbsp;
                    <span style="color:{colors[3]};" title="Transit End">{end.astimezone(ZoneInfo(timezone)).strftime('%H:%M')}</span> &nbsp;–&nbsp;
                    <span style="color:{colors[4]};" title="Observation End">{obsend.astimezone(ZoneInfo(timezone)).strftime('%H:%M')}</span>
                </div>
            </div>
            """
            st.markdown(box_content, unsafe_allow_html=True)


        # Step 4: Update selected transits in session state after rendering all
        st.session_state['selected_transits'] = new_selected

        # Show selected transits summary
        date = st.session_state.get('date', None)
        city = st.session_state.get('city', None)
        dusk_type = st.session_state.get('dusk_type', None)
        max_exp = st.session_state.get('max_exp', None)
        bin = st.session_state.get('bin', None)
        device_name = st.session_state.get('device_name', None)
        utc = st.session_state.get('utc', None)
        if utc: timezone = 'UTC'
        else:
            tf = TimezoneFinder()
            timezone = tf.timezone_at(lat=city.latitude, lng=city.longitude)

        if len(st.session_state['selected_transits']) > 0:
            if st.button("Create Schedule CSV"):
                try:
                    # Call your function to write schedule
                    csv_data = write_schedule(
                        st.session_state['selected_transits'],
                        date,
                        city,
                        max_exp=max_exp,
                        bin=bin,
                        device_name=device_name,
                        tzone=timezone
                    )
                    st.success("Schedule CSV created successfully!")

                    st.download_button(
                        label="Download Schedule CSV",
                        data=csv_data,
                        file_name=f'schedule_{date}.csv',
                        mime='text/csv'
                    )

                except Exception as e:
                    st.error(f"Error creating schedule: {e}")
        else:
            st.info("No transits selected to create a schedule.")

    else:
        st.info("No observable transits found for the selected date and location.")