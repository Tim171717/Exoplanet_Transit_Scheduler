import numpy as np
import pandas as pd
from PyAstronomy import pyasl
import datetime
from astral import LocationInfo
from astral.sun import sun
import csv
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun, get_body
from scipy.constants import c, au
import ephem
import streamlit as st
from geopy.geocoders import Nominatim
import requests


possible_headers = ['Planet', 'Planet_name', 'name', 'Name',
                    'Declination', 'Dec',
                    'Right Ascension', 'RA',
                    'ephemeris mid_time', 'ephemeris mid_time [BJD]', 'epoch',
                    'ephemeris mid_time uncertainty', 'epoch_uncertainty',
                    'ephemeris period', 'ephemeris period [days]', 'period',
                    'ephemeris period uncertainty', 'period_uncertainty',
                    'duration', 'duration [hours]'
                    ]

colormapping = {
    'campaign': 'turquoise',
    'ttvs': 'darkturquoise',
    'alert': 'blueviolet',
    'high': 'red',
    'medium': 'orange',
    'low': 'limegreen'
}


def BJDtoJD(time, dec, ra, city_astropy, input='BJD'):
    sun = get_sun(Time(pyasl.daycnv(time, mode='dt'))).transform_to(AltAz(location=city_astropy, obstime=pyasl.daycnv(time, mode='dt')))
    obj = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    object = obj.transform_to(AltAz(obstime=pyasl.daycnv(time, mode='dt'), location=city_astropy))
    n_sun = np.array([np.cos(sun.alt.rad)*np.sin(sun.az.rad),
                      np.cos(sun.alt.rad)*np.cos(sun.az.rad),
                      np.sin(sun.alt.rad)])
    n_object = np.array([np.cos(object.alt.rad)*np.sin(object.az.rad),
                      np.cos(object.alt.rad)*np.cos(object.az.rad),
                      np.sin(object.alt.rad)])
    if input == 'JD':
        k = -1
    elif input == 'BJD':
        k = 1
    else:
        raise ValueError("Invalid input. Must be 'JD' or 'BJD'.")
    return time + k * np.dot(n_sun, n_object) * au / c / 3600 / 24


def isinTransit(transittime, period, duration, dusk, dawn, dec, ra, city_astropy, transittime_unc, period_unc, add_time):
    counter = 0
    while transittime < dusk + duration/48 + 1/72:
        transittime += period
        counter += 1
    if transittime + duration/48 + 1/72 <= dawn:
        starttime = max(BJDtoJD(transittime - duration/48 - counter * period_unc - transittime_unc, dec, ra, city_astropy) - add_time/60/24, dusk)
        endtime = min(BJDtoJD(transittime + duration/48 + counter * period_unc + transittime_unc, dec, ra, city_astropy) + add_time/60/24, dawn)
        return endtime <= dawn, pyasl.daycnv(starttime, mode='dt'), pyasl.daycnv(endtime, mode='dt')
    else:
        return False, pyasl.daycnv(2400000, mode='dt'), pyasl.daycnv(2400000, mode='dt')


def Starisvisible(dec, ra, city_astropy, starttime, endtime):
    # alt_limit = Max's function
    object = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    time = starttime
    while time <= endtime:
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

    return True


def DuskandDawn(city_ephem, date):
    city_ephem.date = date
    dusk = city_ephem.next_setting(ephem.Sun()).datetime()
    city_ephem.date = date + datetime.timedelta(days=1)
    dawn = city_ephem.next_rising(ephem.Sun()).datetime()
    return dusk, dawn


def otherTargets(starttime1, endtime1, starttime2, endtime2):
    return starttime1 > endtime2 or starttime2 > endtime1


def Get_availabilities(date, city, elevation, add_time, dusk_type, import_catalog, progress_callback=None):
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

    # Detect file type and read accordingly
    if str(import_catalog)[0:2] == 'D:':
        file_name = import_catalog.split("/")[-1]
        if file_name.endswith('.csv'):
            df = pd.read_csv(import_catalog)
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(import_catalog, engine='openpyxl')
    else:
        if import_catalog.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif import_catalog.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')

    headers = []
    for header in possible_headers:
        if header in df.columns:
            headers.append(header)
    if 'priority' in df.columns:
        headers.append('priority')
    if len(headers) < 8:
        raise ValueError("Invalid import catalog. Please check the headers.")

    total_rows = len(df)

    dusk, dawn = DuskandDawn(city_ephem, date)
    for i, row in df.iterrows():
        name = row[headers[0]]
        dec = row[headers[1]]
        ra = row[headers[2]]
        transittime = row[headers[3]]
        transittime_unc = row[headers[4]]
        period = row[headers[5]]
        period_unc = row[headers[6]]
        duration = row[headers[7]]


        if np.isnan(transittime_unc): transittime_unc = 0
        if np.isnan(period_unc): period_unc = 0

        istransiting, starttime, endtime = isinTransit(transittime, period, duration, pyasl.jdcnv(dusk),
                                                       pyasl.jdcnv(dawn), dec, ra, city_astropy,
                                                       transittime_unc, period_unc, add_time)
        if istransiting:
            if Starisvisible(dec, ra, city_astropy, starttime, endtime):
                planet_info = [name, dec, ra, starttime, endtime]
                if len(headers) == 9:
                    planet_info.append(row[headers[8]])
                else:
                    planet_info.append('no priority')

                found_transits.append(planet_info)

        if progress_callback is not None:
            progress_callback(int((i + 1) / total_rows * 100))

    found_transits.sort(key=lambda x: x[3])
    return found_transits


def write_schedule(selected_transits, date, city, dusk_type, exposure_time=120, filter='G', bin=4):
    selected_transits.sort(key=lambda x: x[3])
    city_ephem = ephem.Observer()
    city_ephem.pressure = 0;
    city_ephem.lat, city_ephem.lon = str(city.latitude), str(city.longitude)
    if dusk_type == 'Astronomical':
        city_ephem.horizon = '-18'
    elif dusk_type == 'Nautical':
        city_ephem.horizon = '-12'
    else:
        city_ephem.horizon = '-6'
    sunset = sun(city.observer, date=date)['sunset'].replace(tzinfo=None)
    sunrise = sun(city.observer, date=date + datetime.timedelta(days=1))['sunrise'].replace(tzinfo=None)
    dusk, dawn = DuskandDawn(city_ephem, date)

    obsplan = [{'device_type': 'Camera', 'device_name': 'camera_hpp', 'action_type': 'open',
                'action_value': {}, 'start_time': sunset, 'end_time': sunrise},
               {'device_type': 'Camera', 'device_name': 'camera_hpp', 'action_type': 'flats',
                'action_value': str({'filter': [filter], 'n': [20], 'bin': bin}),
                'start_time': sunset + datetime.timedelta(minutes=5), 'end_time': dusk}]

    for transit in selected_transits:
        obsplan.append({'device_type': 'Camera',
                        'device_name': 'camera_hpp',
                        'action_type': 'object',
                        'action_value': str({'object': transit[0],
                                             'filter': filter,
                                             'ra': SkyCoord(transit[2], transit[1], unit=(u.hourangle, u.deg)).ra.deg,
                                             'dec': SkyCoord(transit[2], transit[1], unit=(u.hourangle, u.deg)).dec.deg,
                                             'exptime': exposure_time,
                                             'guiding': True,
                                             'pointing': False,
                                             'bin': bin}),
                        'start_time': str(transit[3]),
                        'end_time': str(transit[4])})
    exptimes = (np.unique(np.append(np.linspace(0,120,8, dtype=int), exposure_time))).tolist()
    nn = (np.zeros(len(exptimes), dtype=int) + 10).tolist()
    obsplan.append({'device_type': 'Camera',
                    'device_name': 'camera_hpp',
                    'action_type': 'flats',
                    'action_value': str({'filter': [filter],
                                         'n': [20],
                                         'bin': bin}),
                    'start_time': dawn,
                    'end_time': sunrise})
    obsplan.append({'device_type': 'Camera',
                    'device_name': 'camera_hpp',
                    'action_type': 'close',
                    'action_value': "{}",
                    'start_time': sunrise,
                    'end_time': sunrise + datetime.timedelta(minutes=5)})
    obsplan.append({'device_type': 'Camera',
                    'device_name': 'camera_hpp',
                    'action_type': 'calibration',
                    'action_value': str({'exptime': exptimes, 'n': nn, 'bin': bin}),
                    'start_time': sunrise + datetime.timedelta(minutes=5),
                    'end_time': sunrise + datetime.timedelta(minutes=65)})

    with open('Schedules\schedule_' + str(date) + '.csv', 'w', newline='') as csvfile:
        fieldnames = ['device_type', 'device_name', 'action_type', 'action_value', 'start_time', 'end_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(obsplan)


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

    if location_query:
        geolocator = Nominatim(user_agent="streamlit-location-search")



    # Additional inputs
    add_time = st.slider('⏳ Additional Observation Time (min)', min_value=20, max_value=60, value=60)
    alt_limit = st.number_input("🌌 Altitude Limit (°)", min_value=0.0, max_value=90.0, value=30.0, step=1.0)
    moon_distance = st.number_input("🌙 Min Moon Distance (°)", min_value=0.0, max_value=180.0, value=30.0, step=1.0)
    dusk_type = st.selectbox("🌅 Dusk Type", options=["Astronomical", "Nautical", "Civil"])
    st.session_state['dusk_type'] = dusk_type

    # File uploader
    st.markdown("<h2 style='font-size:25px;'>📂 Catalog Upload</h2>", unsafe_allow_html=True)
    use_default = st.checkbox("Use default Catalog")
    if use_default:
        # Load default file from your directory
        uploaded_file = "D:/timlf/Tim Daten/Documents/GitHub/Semester_Project_Tim/ExoplanetCatalog_Exoclock_usefull.xlsx"
    else:
        uploaded_file = st.file_uploader("Upload a catalog (CSV, Excel)", type=["csv", "xlsx"])

    # Submit button and trigger calculation + store results in session state
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
                timezone = "UTC"

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

                city = LocationInfo(name=city_info, region=country, timezone=timezone,
                                    latitude=location.latitude, longitude=location.longitude)
                st.session_state['city'] = city

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


                found_transits = Get_availabilities(date, city, elevation, add_time, dusk_type, uploaded_file,
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
        st.success(f"Found {len(found_transits)} observable transits!")

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
            name, dec, ra, start, end = t[:5]
            priority = t[5] if len(t) == 6 else "no priority"

            # Check if this transit overlaps with any *other* selected transit
            is_compatible = True
            for sel in new_selected:
                if sel == t:
                    continue
                sel_start, sel_end = sel[3], sel[4]
                if not otherTargets(start, end, sel_start, sel_end):
                    is_compatible = False
                    break

            # Show disabled checkbox if not compatible and not already selected
            disabled = (not is_compatible) and (t not in new_selected)

            # Render checkbox with disabled state
            checked = st.checkbox(
                f"Select", key=f"transit_{i}", value=(t in new_selected), disabled=disabled
            )

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
                "background-color: #f0f0f0; color: #999;" if disabled else "background-color: #f9f9f9;"
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
                    {"<em style='font-style: italic; color: #999; margin-left: 10px;'>"
                     "(Overlaps with selected transit)</em>" if disabled else "<span style='color: white;'>.</span>"}
                </div>
                <div style="margin-top: 5px;">
                    Time: {start.strftime('%Y-%m-%d %H:%M')} – {end.strftime('%Y-%m-%d %H:%M')}
                </div>
            </div>
            """
            st.markdown(box_content, unsafe_allow_html=True)

        #coordinates display: <br>
        #Coordinates: Dec {dec}, RA {ra}

        # Step 4: Update selected transits in session state after rendering all
        st.session_state['selected_transits'] = new_selected

        # Show selected transits summary
        date = st.session_state.get('date', None)
        city = st.session_state.get('city', None)
        dusk_type = st.session_state.get('dusk_type', None)

        if len(st.session_state['selected_transits']) > 0:
            if st.button("Create Schedule CSV"):
                try:
                    # Call your function to write schedule
                    write_schedule(
                        st.session_state['selected_transits'],
                        date,
                        city,
                        dusk_type,
                        exposure_time=120,  # you can add inputs for these if you want
                        filter='G',
                        bin=4
                    )
                    st.success("Schedule CSV created successfully!")

                    # Optional: Provide download link
                    with open(f'Schedules/schedule_{date}.csv', 'rb') as f:
                        st.download_button(
                            label="Download Schedule CSV",
                            data=f,
                            file_name=f'schedule_{date}.csv',
                            mime='text/csv'
                        )
                except Exception as e:
                    st.error(f"Error creating schedule: {e}")
        else:
            st.info("No transits selected to create a schedule.")

    else:
        st.info("No observable transits found for the selected date and location.")

