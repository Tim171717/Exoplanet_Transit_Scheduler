import numpy as np
import numpy.ma as ma
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astropy.coordinates import Distance
from mphot import core


### PARAMETERS

def get_exptime(name,                   # star name of the target
                coords,                 # target coordinates as SkyCoord object
                obstime,                # observation time and date in the format 'YYYY-MM-DD HH:MM:SS'
                pwv=30,                 # precipitable water vapor in mm, default 30 mm
                seeing=4.5,             # FWHM of star spread function in ", default 4.5 "
                max_exp=120,            # maximum exposure time in s
                gain_setting=0,
                binning=4):
    info = get_star_info(name)
    if info["source"] == 'default':
        return ('clear', max_exp)
    Teff = info['temperature_K']
    distance = info['distance_pc']

    system_response, results = get_models(
        instrument_efficiency_path='resources/systems/generic_IMX461.csv',
        filter_paths=get_filters(),
        target_Teffs=[Teff],
        target_dists=[distance],
        obs_times=obstime,
        pwv=pwv,
        seeing=seeing,
        max_exp=[max_exp],
        target_coords=[coords],
        target_names=[name],
        binning=binning,
        gain_setting=[gain_setting],
        status_updates=False,
        precision_curve=False
    )

    max_i = find_best_params(results)
    result = results[max_i]
    if result['system_name'] == 'generic_IMX461_baader_r':
        return 'R', round(result['precisions'][2]['t [s]'], 3)
    elif result['system_name'] == 'generic_IMX461_baader_g':
        return 'G', round(result['precisions'][2]['t [s]'], 3)
    elif result['system_name'] == 'generic_IMX461_baader_b':
        return 'B', round(result['precisions'][2]['t [s]'], 3)
    else:
        return 'clear', round(result['precisions'][2]['t [s]'], 3)




### FUNCTIONS


def get_star_info(star_name, default_teff=5778.0, default_distance=10.0):
    """
    Retrieve temperature and distance for a star by name.
    Attempts Gaia DR3, then Gaia DR2, then TOI IDs via SIMBAD.
    Falls back to TESS Input Catalog if Gaia data is unavailable.
    Assigns default values if all queries fail.

    Parameters:
    star_name (str): Name of the star.
    default_teff (float): Default effective temperature in Kelvin.
    default_distance (float): Default distance in parsecs.

    Returns:
    dict: Dictionary containing 'source', 'temperature_K', and 'distance_pc'.
    """
    Simbad.TIMEOUT = 120
    try:
        # Query SIMBAD for object IDs
        ids_table = Simbad.query_objectids(star_name)
        if ids_table is None:
            raise ValueError(f"No identifiers found for {star_name} in SIMBAD.")

        # Extract IDs from the table
        ids = np.array(ids_table)

        # Search for Gaia DR3 ID
        for idn in ids:
            if idn[0].startswith('Gaia DR3'):
                gaia_id = idn[0].split()[-1]
                return query_gaia(gaia_id, default_teff, default_distance)

        # Search for Gaia DR2 ID
        for idn in ids:
            if idn[0].startswith('Gaia DR2'):
                gaia_id = idn[0].split()[-1]
                return query_gaia(gaia_id, default_teff, default_distance)

        # Search for TOI ID (TESS Object of Interest)
        for idn in ids:
            if idn[0].startswith('TOI'):
                toi_id = idn[0].split()[-1]
                return query_tess(toi_id, default_teff, default_distance)

        raise ValueError(f"No Gaia or TOI ID found for {star_name}.")

    except Exception as e:
        print(f"Error: {e}")
        return {
            "source": "default",
            "temperature_K": default_teff,
            "distance_pc": default_distance
        }

def query_gaia(gaia_id, default_teff, default_distance):
    """
    Query Gaia DR3 catalog for temperature and distance using Gaia ID.

    Parameters:
    gaia_id (str): Gaia DR3 source ID.
    default_teff (float): Default effective temperature in Kelvin.
    default_distance (float): Default distance in parsecs.

    Returns:
    dict: Dictionary containing 'source', 'temperature_K', and 'distance_pc'.
    """
    try:
        query = f"""
            SELECT source_id, teff_gspphot, parallax
            FROM gaiadr3.gaia_source
            WHERE source_id = {gaia_id}
        """
        job = Gaia.launch_job(query)
        result = job.get_results()

        if len(result) == 0:
            raise ValueError(f"No data found in Gaia DR3 for ID {gaia_id}.")

        teff = result["teff_gspphot"][0]
        parallax = result["parallax"][0]

        # Check for masked or invalid values
        if ma.is_masked(teff) or ma.is_masked(parallax) or parallax <= 0:
            raise ValueError("Incomplete or invalid data in Gaia DR3.")

        distance = Distance(parallax=parallax * u.mas).pc

        return {
            "source": "Gaia DR3",
            "temperature_K": teff,
            "distance_pc": distance
        }

    except Exception as e:
        print(f"Gaia query error: {e}")
        return {
            "source": "default",
            "temperature_K": default_teff,
            "distance_pc": default_distance
        }

def query_tess(toi_id, default_teff, default_distance):
    """
    Query TESS Input Catalog for temperature and distance using TOI ID.

    Parameters:
    toi_id (str): TESS Object of Interest ID.
    default_teff (float): Default effective temperature in Kelvin.
    default_distance (float): Default distance in parsecs.

    Returns:
    dict: Dictionary containing 'source', 'temperature_K', and 'distance_pc'.
    """
    try:
        result = Catalogs.query_object(f"TOI {toi_id}", catalog="TIC")
        if len(result) == 0:
            raise ValueError(f"No data found in TESS catalog for TOI {toi_id}.")

        star = result[0]
        teff = star.get("Teff")
        distance = star.get("d")

        # Check for None or invalid values
        if teff is None or distance is None:
            raise ValueError("Incomplete data in TESS catalog.")

        return {
            "source": "TESS",
            "temperature_K": teff,
            "distance_pc": distance
        }

    except Exception as e:
        print(f"TESS query error: {e}")
        return {
            "source": "default",
            "temperature_K": default_teff,
            "distance_pc": default_distance
        }



def get_airmasses(
        target_coords,
        obs_times,
        make_valid=False
    ):

    airmasses = []

    if(np.isscalar(obs_times)):
        obs_times = [Time(obs_times)] * len(target_coords)
    else:
        for i in range(len(obs_times)):
            obs_times[i] = Time(obs_times[i])

    HPP = EarthLocation(lat=47.40780226034046*u.deg, lon=8.510942690829427*u.deg, height=547*u.m)

    for target, obs_time in zip(target_coords, obs_times):
        altaz = target.transform_to(AltAz(obstime=Time(obs_time, scale='utc'), location=HPP))
        airmass = altaz.secz

        if airmass < 1 and make_valid:
            print(f"WARNING: Calculated airmass < 1 for target coordinates {target.to_string()}, assigning value 1.")
            airmass = 1
        elif not(altaz.zen.is_within_bounds(upper=60 * u.deg)):
            print(f"WARNING: Zenith angle {altaz.zen.to_string(unit=u.deg, decimal=True)}° > 60° for target coordinates {target.to_string()}. Calculated Airmass = {airmass}")

        airmasses.append(airmass)
    
    return airmasses



def get_models(
        instrument_efficiency_path,
        filter_paths,
        target_Teffs,
        target_dists,
        obs_times,
        pwv,
        seeing,
        max_exp,
        airmasses = None,
        target_coords = None,
        target_names = None,
        gain_setting = [0, 1, 2, 3],
        binning = 4,
        status_updates = False,
        precision_curve = True
    ):

    # variables
    results = []
    system_responses = []
    props_sky = []
    status = "Obtained all precision results for the system(s) "



    # get target airmasses and initialize variables
    if airmasses is None:
        airmasses = get_airmasses(target_coords=target_coords, obs_times=obs_times, make_valid=True)
    elif np.isscalar(airmasses):
        airmasses = [airmasses] * len(target_Teffs)

    if target_names is None:
        target_names = np.char.mod('%d', np.arange(len(target_Teffs)))

    if np.isscalar(binning):
        binning = [binning]

    if np.isscalar(seeing):
        seeing = [seeing] * len(target_Teffs)

    

    # create sky properties array
    for airmass, see in zip(airmasses, seeing):
        props = {
            "pwv" : pwv,
            "airmass" : airmass,
            "seeing" : see
        }

        props_sky.append(props)



    # get instrument data
    for filter_path in filter_paths:
        name, system_response = core.generate_system_response(instrument_efficiency_path, filter_path)
        response = {
            "name" : name,
            "system_response" : system_response
        }
        system_responses.append(response)
        status = status + name + ", "

        well_depth = [50000, 16500, 16500, 11200]

        for bin in binning:
            for setting in gain_setting:
                if status_updates:
                    print(f'Obtaining precision results for system {name}, a binning of {bin}x{bin} pix and gain setting {setting}...')

                plate_scale = 0.225 * bin

                # get precision results
                for target_name, Teff, dist, props_s, m_exp in zip(target_names, target_Teffs, target_dists, props_sky, max_exp):
                    if status_updates:
                        print(f'Current Target: {target_name}')

                    ap_rad = (1 + 10 * 14 / 29) * bin * plate_scale / props_s["seeing"]

                    props_i = {
                        "name" : name,                                # name to get SR/precision grid from file
                        "plate_scale" : plate_scale,                  # pixel plate scale ["]
                        "N_dc" : 0.00058 * bin**2,                    # dark current [e/pix/s]
                        "N_rn" : 2.117 * bin,                         # read noise [e_rms/pix]
                        "well_depth" : well_depth[setting] * bin**2,  # well depth [e/pix]
                        "well_fill" : 0.7,                            # fractional value to fill central target pixel, assuming gaussian (width function of seeing^)
                        "read_time" : 5,                              # read time between images [s]
                        "r0" : 0.254,                                 # radius of telescope's primary mirror [m]
                        "r1" : 0.099,                                 # radius of telescope's secondary mirror [m]
                        # "min_exp" : 0,                              # optional, minimum exposure time [s]
                        "max_exp" : m_exp,                            # optional, maximum exposure time [s]
                        "ap_rad" : ap_rad                             # optional, aperture radius [FWHM, seeing] for photometry -- 3 default == 7 sigma of Gaussian
                    }



                    precision = core.get_precision(props_i, props_s, Teff, dist, extended=True)

                    # mphot may return quantity type
                    for i, el in enumerate(precision):
                        for key in el.keys():
                            if type(el[key]) is u.quantity.Quantity:
                                precision[i][key] = el[key].to_value()

                    image_precision, binned_precision, components = precision
                


                    # get precision vs flux data
                    exp_time = components["t [s]"]
                    base_N_star = components["N_star [e/s]"]

                    flux_factor = np.logspace(-4, 4, 200)
                    N_stars = [base_N_star * i for i in flux_factor]

                    precision_array = []
                    flux_array = []

                    if precision_curve:
                        for i in N_stars:
                            image_precision_i, binned_precision_i, components_i = core.get_precision(props_i, props_s, Teff, dist, exp_time=exp_time, N_star=i, extended=True)
                            
                            # mphot may return quantity type
                            for key in binned_precision_i.keys():
                                if type(binned_precision_i[key]) is u.quantity.Quantity:
                                    binned_precision_i[key] = binned_precision_i[key].to_value()

                            precision_array.append(binned_precision_i)
                            flux_array.append(components_i['N_star [e/s]'])



                    # get SNR
                    single_frame_snr = 1 / image_precision["All"]      
                    binned_snr = 1 / binned_precision["All"]



                    # create results dictionary
                    result = {
                        "target_name" : target_name,
                        "system_name" : name,
                        "binning" : bin,
                        "gain_setting" : setting,
                        "precisions" : precision,
                        "precision_vs_flux" : (precision_array, flux_array),
                        "single_frame_snr" : single_frame_snr,
                        "10min_binned_snr" : binned_snr,
                        "ap_rad" : (1 + 10 * 14 / 29) * bin
                    }
                    results.append(result)    



    if status_updates:
        status = status + "with binning(s) of "

        for bin in binning:
            status = status + str(bin) + "x" + str(bin) + " pix, "

        status = status + "and gain setting(s)"

        for setting in gain_setting:
            status = status + " " + str(setting) + ","

        status = status[:-1] + "."
        print(status)

    return system_responses, results



def get_filters():
    filter_clear = 'resources/filters/clear.csv'
    filter_baader_r = 'resources/filters/baader/baader_r.csv'
    filter_baader_g = 'resources/filters/baader/baader_g.csv'
    filter_baader_b = 'resources/filters/baader/baader_b.csv'

    return [filter_clear, filter_baader_r, filter_baader_g, filter_baader_b]



def find_best_params(results):
    precisions = []

    for sys in results:
        precisions.append(sys['precisions'][1]['All'])

    i_min = np.argmin(np.array(precisions))

    return i_min



def print_results(indices, results):
    for i in indices:
        result = results[i]
        
        system = result['system_name']
        precisions = result['precisions']
        ppt_bin = precisions[1]['All'] * 1000
        snr_bin = result['10min_binned_snr']
        exp_time = precisions[2]['t [s]']

        print('--------------------')
        print(f'System: {system}')
        print(f'Exposure time: {exp_time:.2f}s')
        print(f'ppt (10min binned): {ppt_bin:.2f}')
        print(f'SNR (10min binned): {snr_bin:.2f}')



