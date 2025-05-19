import copy
import numpy as np
import pandas as pd
import datetime
from astral import LocationInfo
from pygments.lexers import go
import ephem
import plotly.graph_objects as go
from zoneinfo import ZoneInfo
import streamlit as st

from Schedule_maker import Get_availabilities, DuskandDawn


def priority(char):
    if char == 'campaign': return 1
    if char == 'ttvs': return 2
    if char == 'alert': return 3
    if char == 'high': return 4
    if char == 'medium': return 5
    if char == 'low': return 6


def getnames(data):
    low_data = []
    medium_data = []
    high_data = []
    alert_data = []
    ttvs_data = []
    campaign_data = []
    priority = []
    for d in data:
        if d[3] == 'low':
            low_data.append(d[2])
        if d[3] == 'medium':
            medium_data.append(d[2])
        if d[3] == 'high':
            high_data.append(d[2])
        if d[3] == 'alert':
            alert_data.append(d[2])
        if d[3] == 'ttvs':
            ttvs_data.append(d[2])
        if d[3] == 'campaign':
            campaign_data.append(d[2])
        priority.append(d[3])
    low_data = np.unique(np.array(low_data))
    medium_data = np.unique(np.array(medium_data))
    high_data = np.unique(np.array(high_data))
    alert_data = np.unique(np.array(alert_data))
    ttvs_data = np.unique(np.array(ttvs_data))
    campaign_data = np.unique(np.array(campaign_data))
    priority = np.unique(np.array(priority))

    return np.concatenate((campaign_data, ttvs_data, alert_data, high_data, medium_data, low_data))


def remap_time(dt: datetime.datetime, a: datetime.datetime, b: datetime.datetime) -> datetime.datetime:
    """
    Map dt in the time interval [a, b] (spanning midnight) to [13:00, 11:00 next day].

    Parameters:
        dt: datetime.datetime – the datetime to transform
        a, b: datetime.time – start and end of the original time window (must span midnight, a > b)

    Returns:
        datetime.datetime – a new datetime with remapped hour (date preserved, hour remapped)
    """
    # Convert times to fractional hours
    def to_hours(t: datetime.datetime) -> float:
        return t.hour + t.minute / 60 + t.second / 3600

    a_hour = to_hours(a)
    b_hour = to_hours(b) + 24  # b is after midnight (next day)
    dt_hour = to_hours(dt)

    # Ensure dt is in the original window, adjust if before `a`
    if dt_hour < a_hour:
        dt_hour += 24

    # Normalize to 0–1 range
    scale = (dt_hour - a_hour) / (b_hour - a_hour)

    # Map to [13, 35] (11 + 24)
    mapped_hour = 13 + scale * (35 - 13)
    if mapped_hour >= 13:
        new_date = day_of(a) + datetime.timedelta(days=1)
        mapped_hour -= 24
    else:
        new_date = day_of(a)

    remapped_dt = datetime.datetime.combine(new_date, datetime.time()) + datetime.timedelta(hours=mapped_hour)
    return remapped_dt


def day_of(date: datetime.datetime) -> datetime.date:
    if date.hour < 12:
        return date.date() - datetime.timedelta(days=1)
    else:
        return date.date()


def make_timeline(
        startdate,
        enddate,
        city,
        elevation,
        df,
        alt_limit=20,
        moon_distance = 30,
        dusk_type='Civil',
        aperture_size=20,
        timezone='Europe/Zurich',
        progress_callback=None
):
    date = startdate
    days = (enddate - startdate).days + 1
    n = 0
    data = []

    while date <= enddate:
        if progress_callback is not None:
            def progress_update2(pct):
                pctt = (n + pct/100) / days
                progress_callback(pctt, f"Progress: {int(pctt * 100)}% — Observing the {date.strftime('%Y-%m-%d')}")
        else:
            progress_update2 = None
        found_transits = Get_availabilities(date, city, elevation, df, alt_limit=alt_limit, moon_distance=moon_distance,
                                            dusk_type=dusk_type, aperture_size=aperture_size, progress_callback=progress_update2)
        for t in found_transits:
            data.append([t[3].astimezone(ZoneInfo(timezone)), t[7].astimezone(ZoneInfo(timezone)), t[0], t[9]])
        date += datetime.timedelta(days=1)
        n += 1

    names = getnames(data)
    planets = {t: (len(names) - i) for i, t in enumerate(names)}

    colormapping = {'campaign' : 'cyan', 'ttvs': 'darkturquoise', 'alert': 'blueviolet', 'high': 'red',
                    'medium': 'orange', 'low': 'limegreen'}

    fig = go.Figure()

    fig.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='rgb(15, 17, 23)',
    )

    city_ephem = ephem.Observer(); city_ephem.pressure = 0
    city_ephem.lat, city_ephem.lon = str(city.latitude), str(city.longitude)
    if dusk_type == 'Astronomical':
        city_ephem.horizon = '-18'
    elif dusk_type == 'Nautical':
        city_ephem.horizon = '-12'
    else:
        city_ephem.horizon = '-6'

    start_date = startdate #day_of(data[0][0])
    end_date = enddate #day_of(data[-1][1]) + datetime.timedelta(days=1)
    days = (end_date - start_date).days - 1

    data_edited = copy.deepcopy(data)
    for i, d in enumerate(data_edited):
        dusk, dawn = DuskandDawn(city_ephem, day_of(d[0]))
        dusk = (dusk.replace(tzinfo=datetime.timezone.utc)).astimezone(ZoneInfo(timezone))
        dawn = (dawn.replace(tzinfo=datetime.timezone.utc)).astimezone(ZoneInfo(timezone))
        d[0] = remap_time(d[0], dusk, dawn)
        d[1] = remap_time(d[1], dusk, dawn)


    # Loop through each date and add a vertical line at 12:00
    for n in range((end_date - start_date).days + 1):
        midday = datetime.datetime.combine(start_date + datetime.timedelta(days=n), datetime.datetime.min.time()) + datetime.timedelta(hours=12)
        fig.add_vline(
            x=midday,
            line=dict(color='darkslategrey', width=1, dash="dash"),
            layer="below",
            opacity=0.7
        )

    for d_initial, d in zip(data, data_edited):
        y_center = planets[d[2]]
        fig.add_trace(go.Scatter(
            x=[d[0], d[1], d[1], d[0], d[0]],
            y=[y_center - 0.4, y_center - 0.4, y_center + 0.4, y_center + 0.4, y_center - 0.4],
            fill="toself",
            fillcolor=colormapping[d[3]],
            name=d[2],
            mode="lines",
            hoveron='fills',
            hoverinfo='text',
            line=dict( color=colormapping[d[3]]),
            showlegend=False,
            text=f"<b>{d_initial[2]}</b><br>Start: {d_initial[0].strftime('%Y-%m-%d %H:%M')}<br>End: {d_initial[1].strftime('%Y-%m-%d %H:%M')}<br>Priority: {d_initial[3]}"
        ))

        # Unique legend items
    for label, color in colormapping.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=label.capitalize(),
            showlegend=True
        ))

    fig.update_layout(
        height=200 + len(names) * 20,
        width=100 + days * 150,
        font=dict(color="white"),
        yaxis=dict(
            tickmode='array',
            showline=True,
            tickvals=list(planets.values()),
            ticktext=[name for name in names],
            color='white'
        ),
        xaxis=dict(
            type='date',
            tickmode="linear",
            title="Date",
            showgrid=False,
            tickformat="%Y-%m-%d",
            dtick=86400000 * (days // 15 + 1),
            ticklabelmode="period",
            ticks="outside",
            color="white"
        ),
        title="Future Transit Opportunities",
        legend=dict(title='Priorities',
                    orientation="h",
                    yanchor="bottom",
                    y=1,  # Slightly above the plot
                    xanchor="center",
                    x=0.5,
                    font=dict(color="white")),
        margin=dict(t=110)
    )
    return fig


    # custom_lines = [Line2D([0], [0], color='cyan', lw=4),
    #                 Line2D([0], [0], color='darkturquoise', lw=4),
    #                 Line2D([0], [0], color='blueviolet', lw=4),
    #                 Line2D([0], [0], color='red', lw=4),
    #                 Line2D([0], [0], color='orange', lw=4),
    #                 Line2D([0], [0], color='limegreen', lw=4)]
    #
    # verts = []
    # colors = []
    #
    # for d in data:
    #     v = [(mdates.date2num(d[0] - datetime.timedelta(days=1)), planets[d[2]] - .4),
    #          (mdates.date2num(d[0] - datetime.timedelta(days=1)), planets[d[2]] + .4),
    #          (mdates.date2num(d[1] - datetime.timedelta(days=1)), planets[d[2]] + .4),
    #          (mdates.date2num(d[1] - datetime.timedelta(days=1)), planets[d[2]] - .4),
    #          (mdates.date2num(d[0] - datetime.timedelta(days=1)), planets[d[2]] - .4)]
    #     verts.append(v)
    #     colors.append(colormapping[d[3]])
    #
    # bars = PolyCollection(verts, facecolors=colors)
    #
    # fig = plt.figure(figsize=(25,25))
    # ax = fig.subplots()
    # ax.add_collection(bars)
    # ax.autoscale()
    # loc = mdates.DayLocator(interval=5)
    # ax.xaxis.set_major_locator(loc)
    # ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
    # ax.tick_params(axis='x', which='major', labelsize=15)
    #
    # ax.set_yticks((np.arange(1, len(names) + 1, dtype=int)))
    #
    # ax.set_yticklabels([t for t in reversed(names)])
    #
    # plt.legend(custom_lines, ['Campaign', 'TTVs', 'Alert', 'High', 'Medium', 'Low'], fontsize=20, loc="lower right")
    # # plt.title('Future Transit Opportunities', fontsize=25)
    # plt.xlabel('Date', fontsize=25)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("future_schedule.png")
    # plt.show()



if __name__ == '__main__':
    df = pd.read_csv('ExoClock_Exoplanet_Database.csv')
    startdate = datetime.datetime(2025, 5, 17)
    enddate = datetime.datetime(2025, 5, 31)
    city = LocationInfo("Zurich", "Switzerland", "UTC", 47.2, 8.3)

    data = Get_availabilities(startdate, enddate, city, 575, df)
    fig = make_graph(data)
    fig.show()
