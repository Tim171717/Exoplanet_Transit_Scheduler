import pandas as pd
import datetime
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder
from astral import LocationInfo
import streamlit as st
from geopy.geocoders import Nominatim
import requests

from Schedule_maker import Get_availabilities, select_schedule, write_schedule, otherTargets

colormapping = {
    'campaign': 'turquoise',
    'ttvs': 'darkturquoise',
    'alert': 'blueviolet',
    'high': 'red',
    'medium': 'orange',
    'low': 'limegreen'
}



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

    if catalog_selected == "ExoClock":
        uploaded_file = 'ExoClock_Exoplanet_Database.csv'
        df = pd.read_csv(uploaded_file)

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
        display = st.checkbox('Display Transits in UTC', value=False)
        st.session_state['display'] = display
        utc = st.checkbox('Write Schedule in UTC', value=True)
        st.session_state['utc'] = utc
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
        #selected_filters = st.multiselect("Select filters:", filters, default=default_selection)

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
            st.session_state['firsttime'] = True
            with col_right:
                progress_bar = st.progress(0)
                status_text = st.empty()


                def progress_update(pct):
                    progress_bar.progress(pct)


                found_transits = Get_availabilities(date, city, elevation, df, alt_limit=alt_limit,
                                                    moon_distance=moon_distance, add_time=add_time,
                                                    dusk_type=dusk_type, aperture_size=aperture_size,
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
    #reruns after hitting submit to avoid flickering when pressing best selection for the first time
    if st.session_state.get('firsttime', False):
        st.session_state['firsttime'] = False
        st.rerun()

    # --- OUTPUT: Show found transits ---
    found_transits = st.session_state['found_transits']

    if found_transits:
        city = st.session_state.get('city')
        timezone_str = st.session_state.get('timezone')
        elevation = st.session_state.get('elevation')

        tzinfo = f"🕒 **{timezone_str}**"
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

        st.success(f"✅ Found **{len(found_transits)}** observable transits for {location_info}")

        subcol_left, subcol_right = st.columns([2, 1])
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
            if True not in checked_states:
                if st.button("💡 Best Selection"):
                    try:
                        best_indices = select_schedule(found_transits)
                        best_transits = [found_transits[i] for i in best_indices]

                        # Update session state
                        st.session_state['selected_transits'] = best_transits
                        for i in range(len(found_transits)):
                            st.session_state[f"transit_{i}"] = i in best_indices

                        st.rerun()
                    except Exception as e:
                        st.error(f"Error selecting best transit(s): {e}")

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
                colors = ['#FFA07A' if i == 0 else '#bbb' for i in key]
            else:
                colors = ['red' if i == 0 else 'black' for i in key]

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
            st.info("No transits selected.")


    else:
        st.info("No observable transits found for the selected date and location.")