import pandas as pd
import datetime
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder
from astral import LocationInfo
import streamlit as st
from geopy.geocoders import Nominatim
import requests
import time

from Schedule_maker import Get_availabilities, select_schedule, write_schedule, otherTargets
from airmass_plotter import plot_airmass_interactive, UTCtoTimeZone
from timeline import make_timeline

colormapping = {
    'campaign': 'cyan',
    'ttvs': 'darkturquoise',
    'alert': 'blueviolet',
    'high': 'red',
    'medium': 'orange',
    'low': 'limegreen'
}

priority_info = {
    'campaign': 'CAMPAIGN: Planet included in one of the current observing campaings',
    'ttvs': 'TTVs: Planet exhibiting strong Trasit Time Variations due to another planet or orbital decay',
    'alert': 'ALERT: Observations in the last 2 years show an O-C greater than 10 minutes',
    'high': 'HIGH: Prediction uncertainty greater than the target or less than 3 epochs observed in the last 2 years',
    'medium': 'MEDIUM: Less than 3 epochs observed in the last year',
    'low': 'LOW: Else'
}

# --- PAGE CONFIG ---
st.set_page_config(page_title="Transit Scheduler", layout="centered")

st.title("🪐 Exoplanet Transit Scheduler")

tab1, tab2 = st.tabs(["Schedule Maker", "Timeline"])

with tab1:
    # Create two columns: left for inputs, right for output
    col_left, col_right = st.columns([1, 2])  # Adjust widths if needed

    with col_left:
        # --- INPUTS ---
        st.subheader("🔭 Observation Constraints")

        # Date selection
        date = st.date_input("**📅 Select a date**", value=datetime.date.today())
        st.session_state['date'] = date

        # --- LOCATION SEARCH ---
        location_query = st.text_input("**📍 Location Selection**", value='ETH Hönggerberg')

        location = None
        city = None
        elevation = None
        uploaded_file = None

        # additional time input
        addmode = st.radio('**⏳ Additional Observation Time (min)**', ['fixed time', '% of transit'], horizontal=True)
        if addmode == '% of transit':
            add_time = st.slider('.', min_value=20, max_value=100, value=50, label_visibility='collapsed')
            add_percent = True
        else:
            add_time = st.slider('.', min_value=20, max_value=60, value=60, label_visibility='collapsed')
            add_percent = False

        # aperture
        aperturemode = st.radio('**📏 Aperture**', ['mm', 'inch'], horizontal=True)
        if aperturemode == 'mm':
            aperture_size = st.number_input(".", min_value=0, max_value=3500, value=508,
                                            label_visibility='collapsed') / 25.4
        else:
            aperture_size = st.number_input('.', min_value=0, max_value=150, value=20, label_visibility='collapsed')

        # dusk selection
        dusk_type = st.selectbox("**🌅 Dusk Type**", options=["Astronomical", "Nautical", "Civil"])

        # --- CATALOG SELECTION ---
        catalog_options = {
            "🪐 ExoClock Database": "ExoClock",
            "🚀 NASA Exoplanet Archive": "NASA",
            "📂 Custom Catalog": "Custom"
        }

        selected_label = st.radio("**Select a Catalog**", list(catalog_options.keys()))
        catalog_selected = catalog_options[selected_label]

        # --- LOAD CATALOG BASED ON SELECTION ---
        df = None

        if catalog_selected == "ExoClock":
            uploaded_file = 'ExoClock_Exoplanet_Database.csv'
            df = pd.read_csv(uploaded_file)

        elif catalog_selected == "NASA":
            uploaded_file = 'NASA_Exoplanet_Archive_Database.csv'
            df = pd.read_csv(uploaded_file)

        elif catalog_selected == "Custom":
            uploaded_file = st.file_uploader("📂 Upload Custom Catalog", type=["csv", "xlsx"])
            if uploaded_file is not None:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file, engine='openpyxl')

        # additional inputs
        with st.expander("Additional Settings"):
            display = st.checkbox('Display Transits in UTC', value=False)
            st.session_state['display'] = display
            utc = st.checkbox('Write Schedule in UTC', value=True)
            st.session_state['utc'] = utc
            endearly = st.checkbox('end early (if possible)', value=True)
            st.session_state['endearly'] = endearly
            alt_limit = st.number_input("Altitude Limit (°)", min_value=0.0, max_value=90.0, value=30.0, step=1.0)
            st.session_state['alt_limit'] = alt_limit
            moon_distance = st.number_input("Min Moon Distance (°)", min_value=0.0, max_value=180.0, value=30.0,
                                            step=1.0)
            device_name = st.text_input("Device Name: ", value='camera_hpp')
            st.session_state['device_name'] = device_name
            max_exp = st.number_input("Maximal Exposure Time", value=120, min_value=0, step=10)
            st.session_state['max_exp'] = max_exp
            bin = st.number_input("Binning", value=4, min_value=0, step=1)
            st.session_state['bin'] = bin

            filters = [
                "Clear (UV to IR)",
                "Luminance (B to R)",
                "U (Johnson)",
                "B (Johnson)",
                "V (Johnson)",
                "R (Cousins)",
                "I (Cousins)",
                "H (2MASS)",
                "J (2MASS)",
                "Ks (2MASS)",
                "u' (SDSS)",
                "g' (SDSS)",
                "r' (SDSS)",
                "i' (SDSS)",
                "z' (SDSS)",
                "Astrodon ExoPlanet-BB (V to IR)"
            ]
            default_selection = ['Clear (UV to IR)', "V (Johnson)", "R (Cousins)", "g' (SDSS)"]
            # selected_filters = st.multiselect("Select filters:", filters, default=default_selection)

        if st.button("Submit"):
            if location_query:
                if time.time() - st.session_state.get('lastcall', 0) <= 1:
                    time.sleep(1)
                st.session_state['lastcall'] = time.time()
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
                    else:
                        timezone = 'UTC'
                    st.session_state['city'] = city
                    st.session_state['elevation'] = elevation
                    st.session_state['timezone'] = timezone

                else:
                    st.error("Location not found. Please try another query.")

            if uploaded_file is None:
                st.error("Please upload a catalog.")

            if uploaded_file and city is not None:
                st.session_state['firsttime'] = True
                with col_right:
                    progress_bar = st.progress(0)
                    status_text = st.empty()


                    def progress_update(pct):
                        progress_bar.progress(pct)


                    found_transits = Get_availabilities(date, city, elevation, df, alt_limit=alt_limit,
                                                        moon_distance=moon_distance, add_time=add_time,
                                                        dusk_type=dusk_type, aperture_size=aperture_size,
                                                        add_perc=add_percent, progress_callback=progress_update)

                    progress_bar.empty()  # remove progress bar after done

                    st.session_state['found_transits'] = found_transits
                    st.session_state['selected_transits'] = []  # reset selections

                    if found_transits:
                        st.session_state['plots'] = {t[0]: None for t in found_transits}
                        st.session_state['maxdepht'] = max([t[-1] for t in found_transits])
                        st.session_state['airmass_show'] = [False for t in found_transits]

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
            # reruns after hitting submit to avoid flickering when pressing best selection for the first time and setting the plots up
            if st.session_state.get('firsttime', False):
                st.session_state['firsttime'] = False
                st.rerun()

            city = st.session_state.get('city')
            timezone = st.session_state.get('timezone')
            elevation = st.session_state.get('elevation')

            tzinfo = f"🕒 **{timezone}**"
            if timezone != 'UTC':
                date = datetime.datetime.combine(st.session_state.get('date'), datetime.datetime.min.time())
                date = (date + datetime.timedelta(hours=12)).replace(tzinfo=datetime.timezone.utc)
                utc_offset = UTCtoTimeZone(timezone, date)
                tzinfo += f" **({utc_offset})**"

            location_info = f"**{city.name}/{city.region}, {elevation} m**"

            st.success(f"✅ Found **{len(found_transits)}** observable transits for {location_info}")

            subcol_left, subcol_right = st.columns([1.9, 1])
            with subcol_left:
                st.markdown(tzinfo)

            # Initialize selected_transits if not present
            if 'selected_transits' not in st.session_state:
                st.session_state['selected_transits'] = []

            # Step 1: Gather checkbox states from session state or default to False
            checked_states = []
            for i, t in enumerate(found_transits):
                checked_states.append(st.session_state.get(f"transit_{i}", False))

            with subcol_right:
                if True not in [st.session_state[f"transit_{i}"] for i in range(len(found_transits))]:
                    if st.button("💡 Best Selection"):
                        try:
                            key2 = [int((t[4] - t[3] + t[7] - t[6]).total_seconds()) for t in found_transits]
                            best_indices = select_schedule(found_transits, key2)
                            best_transits = [found_transits[i] for i in best_indices]

                            # Update session state
                            st.session_state['selected_transits'] = best_transits
                            for i in range(len(found_transits)):
                                st.session_state[f"transit_{i}"] = i in best_indices

                            st.rerun()
                        except Exception as e:
                            st.error(f"Error selecting best transit(s): {e}")
                else:
                    if st.button("❌ Deselect All"):
                        st.session_state['selected_transits'] = []
                        for i in range(len(found_transits)):
                            st.session_state[f"transit_{i}"] = False
                        st.rerun()

            # Step 2: Build new selected list from checked_states
            new_selected = [t for i, t in enumerate(found_transits) if checked_states[i]]

            # Step 3: Render transits with compatibility check and disabling incompatible ones
            for i, t in enumerate(found_transits):
                name, dec, ra, obsstart, start, mid, end, obsend, key, priority = t[:-1]

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
                subcol_left, subcol_right = st.columns([1.81, 1])
                with subcol_left:
                    checked = st.checkbox(
                        f"Select", key=f"transit_{i}", disabled=disabled
                    )

                with subcol_right:
                    airmass_show = st.checkbox(
                        f"Show Airmass Plot", key=f"airmass_{i}", value=st.session_state['airmass_show'][i]
                    )
                st.session_state['airmass_show'][i] = airmass_show
                if disabled:
                    colors = ['#FFA07A' if i == 0 else '#bbb' for i in key]
                else:
                    colors = ['red' if i == 0 else 'black' for i in key]

                timezone = st.session_state['timezone']

                priority_tag = ""
                if priority.lower() != "no priority":
                    priority_color = colormapping.get(priority.lower(), "#cccccc")
                    priority_tag = (
                        f"<span title='{priority_info[priority]}' style='"
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
                        <span style="color:{colors[0]};" title="Observation Start">
                        {obsstart.astimezone(ZoneInfo(timezone)).strftime('%H:%M')}</span> &nbsp;–&nbsp;
                        <span style="color:{colors[1]};" title="Transit Start">
                        {start.astimezone(ZoneInfo(timezone)).strftime('%H:%M')}</span> &nbsp;–&nbsp;
                        <span style="color:{colors[2]};" title="Transit Mid-Time">
                        {mid.astimezone(ZoneInfo(timezone)).strftime('%H:%M')}</span> &nbsp;–&nbsp;
                        <span style="color:{colors[3]};" title="Transit End">
                        {end.astimezone(ZoneInfo(timezone)).strftime('%H:%M')}</span> &nbsp;–&nbsp;
                        <span style="color:{colors[4]};" title="Observation End">
                        {obsend.astimezone(ZoneInfo(timezone)).strftime('%H:%M')}</span>
                    </div>
                </div>
                """
                st.markdown(box_content, unsafe_allow_html=True)

                if st.session_state.get(f"airmass_{i}", False):
                    if st.session_state['plots'][name] is None:
                        maxdepht = st.session_state['maxdepht']
                        city = st.session_state['city']
                        elevation = st.session_state['elevation']
                        timezone = st.session_state['timezone']
                        alt_limit = st.session_state['alt_limit']
                        print(
                            (
                                dec, ra,
                                obsstart, obsend,
                                city.latitude, city.longitude, elevation,
                                start, end,
                                t[-1], name,
                                maxdepht,
                                timezone,
                                alt_limit,
                            )
                        )
                        fig = plot_airmass_interactive(
                            dec, ra,
                            obsstart, obsend,
                            city.latitude, city.longitude, elevation,
                            start, end,
                            t[-1], name,
                            maxdepht=maxdepht,
                            timezone=timezone,
                            alt_limit=alt_limit,
                        )
                        st.session_state['plots'][name] = fig
                    else:
                        fig = st.session_state['plots'][name]
                    st.plotly_chart(fig, use_container_width=True)

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
            endearly = st.session_state.get('endearly', None)
            if utc:
                timezone = 'UTC'
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
                            tzone=timezone,
                            endearly=endearly
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
                st.info("No transits selected.")


        else:
            st.info("No observable transits found for the selected date and location.")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        startdate = st.date_input("**📅 Select Startdate**", value=datetime.date.today())
        st.session_state['startdate'] = startdate
    with col2:
        enddate = st.date_input("**📅 Select Enddate**", value=startdate + datetime.timedelta(days=30),
                                min_value=startdate)
        st.session_state['enddate'] = enddate

    col1, col2 = st.columns(2)
    with col1:
        location_query = st.text_input("**📍  Location Selection**", value='ETH Hönggerberg')
    with col2:
        dusk_type = st.selectbox("**🌅  Dusk Type**", options=["Civil", "Nautical", "Astronomical"])

    location = None
    city = None
    elevation = None
    uploaded_file = None

    col1, col2 = st.columns(2)
    with col1:
        alt_limit = st.number_input("**📐 Altitude Limit / 🌙 Moon Distance**", min_value=0.0, max_value=90.0, value=20.0,
                                    step=1.0)
        moon_distance = st.number_input("Min Moon", min_value=0.0, max_value=90.0, value=30.0, step=1.0,
                                        label_visibility='collapsed')
    with col2:
        aperturemode = st.radio('**📏  Aperture**', ['mm', 'inch'], horizontal=True)
        if aperturemode == 'mm':
            aperture_size = st.number_input("..", min_value=0, max_value=3500, value=508,
                                            label_visibility='collapsed') / 25.4
        else:
            aperture_size = st.number_input('..', min_value=0, max_value=150, value=20, label_visibility='collapsed')

    priorities = [
        'campaign',
        'ttvs',
        'alert',
        'high',
        'medium',
        'low'
    ]
    default_sel = ['campaign', 'ttvs', 'alert', 'high', 'medium']
    selected_priorities = st.multiselect("**⭐ Select Priorities**", priorities, default=default_sel)

    if st.button("Create Timeline"):
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
                tf = TimezoneFinder()
                timezone = tf.timezone_at(lat=location.latitude, lng=location.longitude)

            else:
                st.error("Location not found. Please try another query.")

            if city:
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #ccc;
                        border-radius: 10px;
                        padding: 16px;
                        margin-bottom: 1em;
                    ">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <strong>City:</strong> {city.name}<br>
                                <strong>Region/Country:</strong> {city.region}
                            </div>
                            <div>
                                <strong>Timezone:</strong> <code>{timezone}</code><br>
                                <strong>Elevation:</strong> {elevation} m
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if city is not None:
            df = pd.read_csv('ExoClock_Exoplanet_Database.csv')
            filtered_df = df[df['priority'].isin(selected_priorities)]
            progress_bar = st.progress(0)
            status_text = st.empty()


            def progress_update(pct, text):
                progress_bar.progress(pct)
                status_text.text(text)


            fig = make_timeline(startdate, enddate, city, elevation, filtered_df, alt_limit=alt_limit,
                                moon_distance=moon_distance,
                                dusk_type=dusk_type, aperture_size=aperture_size, timezone=timezone,
                                progress_callback=progress_update)
            st.session_state['fig'] = fig

            progress_bar.empty()
            status_text.empty()

    if st.session_state.get('fig', None) is not None:
        fig = st.session_state['fig']
        st.plotly_chart(fig)