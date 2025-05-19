import numpy as np
import datetime
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_sun, get_body
from astropy.time import Time
from astroplan import moon_illumination
from zoneinfo import ZoneInfo
import astropy.units as u
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import plotly.io as pio
pio.templates.default = "none"

def UTCtoTimeZone(timezone, date):
    tz = ZoneInfo(timezone)
    now = date.astimezone(tz)
    offset_sec = tz.utcoffset(now).total_seconds()
    hours = int(offset_sec // 3600)
    minutes = int((abs(offset_sec) % 3600) // 60)
    sign = "+" if offset_sec >= 0 else "-"
    return f"UTC{sign}{abs(hours):02d}:{minutes:02d}"


def get_moon(start, end, city_astropy):
    if start.minute != 0 or start.second != 0 or start.microsecond != 0:
        start = (start.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1))
    end = end.replace(minute=0, second=0, microsecond=0)

    dt_list = []
    current = start
    while current <= end:
        dt_list.append(current)
        current += datetime.timedelta(hours=1)
    time_list = Time(dt_list)

    moon_coord = get_body('moon', time_list, location=city_astropy).transform_to('icrs')
    altaz_frame = AltAz(obstime=time_list, location=city_astropy)
    moon_altaz = moon_coord.transform_to(altaz_frame)
    moon_alts = moon_altaz.alt.deg
    illums = moon_illumination(time_list)

    moon = []
    for time, moon_alt, illum in zip(dt_list, moon_alts, illums):
        if moon_alt > 0:
            moon.append([time, moon_alt, illum])

    return moon


def sky_color_from_solar_altitude(alt):
    """
    Return an RGB hex color that smoothly transitions from sky blue (day)
    to deep night blue as the Sun goes from +10° to -18° altitude.
    """
    # RGB in 0–1 range
    day_rgb = np.array([69, 179, 224]) / 255
    night_rgb = np.array([17, 10, 79]) / 255

    if alt >= 10:
        return day_rgb
    elif alt <= -18:
        return night_rgb
    else:
        # Linear interpolation factor from 0 (day) to 1 (night)
        f = (-alt + 10) / 28  # 10 to -18 spans 28 degrees
        f = np.clip(f, 0, 1)
        color = (1 - f) * day_rgb + f * night_rgb
        return color

def plot_airmass_interactive(
        dec_hour, ra_deg,
        start_time_utc, end_time_utc,
        observer_lat, observer_lon, observer_elev,
        transit_start,
        transit_end,
        transit_depth,
        title="Airmass Plot",
        maxdepht=0.7,
        timezone='UTC',
        alt_limit = 30
    ):
    start_time_utc = start_time_utc.replace(second=0, microsecond=0)
    end_time_utc = end_time_utc.replace(second=0, microsecond=0)
    location = EarthLocation(lat=observer_lat * u.deg, lon=observer_lon * u.deg, height=observer_elev * u.m)
    dt = datetime.timedelta(minutes=1)
    datetimes = []
    current = start_time_utc
    while current <= end_time_utc:
        datetimes.append(current)
        current += dt
    times = Time(datetimes)

    # Define alt/az frame and target
    target = SkyCoord(ra_deg, dec_hour, unit=(u.hourangle, u.deg))
    altaz_frame = AltAz(obstime=times, location=location)
    target_altaz = target.transform_to(altaz_frame)
    airmass = target_altaz.secz
    airmass[airmass < 1] = np.nan

    # Get sun altitude
    sun_altaz = get_sun(times).transform_to(altaz_frame)
    sun_alts = sun_altaz.alt.deg
    sky_colors = [sky_color_from_solar_altitude(alt) for alt in sun_alts]

    #get moon altitudes
    moon_data = get_moon(start_time_utc, end_time_utc, location)
    times = [entry[0].astimezone(ZoneInfo(timezone)) for entry in moon_data]
    altitudes = [entry[1] for entry in moon_data]
    illuminations = [entry[2] * 100 for entry in moon_data]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='rgb(15, 17, 23)',
    )

    y_base = 90
    y_transit = 90 - transit_depth / maxdepht / 1.2 * 90
    fig.add_trace(go.Scatter(
        x=[start_time_utc.astimezone(ZoneInfo(timezone)),
           transit_start.astimezone(ZoneInfo(timezone)),
           transit_start.astimezone(ZoneInfo(timezone)),
           transit_end.astimezone(ZoneInfo(timezone)),
           transit_end.astimezone(ZoneInfo(timezone)),
           end_time_utc.astimezone(ZoneInfo(timezone))],
        y=[y_base, y_base, y_transit, y_transit, y_base, y_base, y_base],
        fill='toself',
        fillcolor='rgba(128,128,128,0.3)',
        line=dict(color='gray', dash='dash'),
        name='Transit',
        hoveron='fills',
        hoverinfo='text',
        text=(
            f"<b>Transit Window</b><br>"
            f"Start: {transit_start.astimezone(ZoneInfo(timezone)).strftime('%H:%M:%S')}<br>"
            f"End: {transit_end.astimezone(ZoneInfo(timezone)).strftime('%H:%M:%S')}<br>"
            f"Depth: {transit_depth:.2f} ppt"
        ),
    ), secondary_y=True)

    fig.add_trace(
        go.Scatter(
            x=times,
            y=altitudes,
            mode='markers',
            name='Moon Info',
            marker=dict(symbol=116, size=12, color='royalblue'),
            text=[f"{t.strftime('%H:%M')}<br>Moon Alt: {alt:.1f}°<br>Illum: {illum:.1f}%"
                  for t, alt, illum in zip(times, altitudes, illuminations)],
            hovertemplate='%{text}',
            showlegend=True
        ),
        secondary_y=True
    )
    fig.update_yaxes(
        showgrid=False,
        range=[0,91],
        title_text="Moon Altitude [deg]",
        secondary_y=True,
        showline=True,
        ticks="outside"
    )

    datetimes = [t.astimezone(ZoneInfo(timezone)) for t in datetimes]
    # Airmass curve
    fig.add_trace(go.Scatter(
        x=datetimes,
        y=airmass,
        name="Airmass",
        line=dict(color="gold", width=2)
    ), secondary_y=False)

    step = 6
    for i in range(0, len(datetimes) - 1, step):
        c = sky_colors[i]
        rgba = f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},0.3)"
        fig.add_vrect(
            x0=datetimes[i], x1=datetimes[min(i + step, len(datetimes) - 1)],
            fillcolor=rgba, line_width=0, layer="below"
        )
    low_limit = 1 / np.sin(np.deg2rad(alt_limit))
    fig.update_yaxes(title_text="Airmass", range=[low_limit,1], secondary_y=False)

    if timezone == 'UTC':
        utc_offset = '(UTC)'
    else:
        date = datetimes[0]
        utc_offset = '(' + UTCtoTimeZone(timezone, date) + ')'

    fig.update_xaxes(title_text="Time " + utc_offset,
                     range=[start_time_utc.astimezone(ZoneInfo(timezone)),
                            end_time_utc.astimezone(ZoneInfo(timezone))],
                     showline=True, ticks="outside")

    fig.update_layout(title=title,
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1),
                      font=dict(color="white"),
    )

    return fig

if __name__ == "__main__":
    st.set_page_config()
    st.title("🌌 Interactive Airmass Plot with Sky Brightness")
    fig = plot_airmass_interactive(
            '+76:33:11.250', '02:57:18.2591',
            datetime.datetime(2025,5,10,14,5), datetime.datetime(2025,5,11,0,49),
            47.40772222222222, 8.510972222222222, 575,
            datetime.datetime(2025,5,10,23,5),
            datetime.datetime(2025,5,10,23,49),
            0.5, "WASP-36b",
    )
    st.plotly_chart(fig, use_container_width=True)