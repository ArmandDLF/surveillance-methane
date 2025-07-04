import ee
import wxee
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from codepropre import Rterre, taillecarre

SATS = ["ERA5", "GEOS"]
SATELLITES = {}
SATELLITES["ERA5"]  = ['ECMWF/ERA5_LAND/HOURLY', 11132, "surface_pressure", "u_component_of_wind_10m", "v_component_of_wind_10m"] 
#name, pixel_resolution, pressure, east wind component, north wind component
SATELLITES["GEOS"] = ['NASA/GEOS-CF/v1/rpl/tavg1hr', 27750, "PS", "U10M", "V10M"]
#SATELLITES["GFS"] =["NOAA/GFS0P25", 27830, None, "u_component_of_wind_10m_above_ground", "v_component_of_wind_10m_above_ground"]


def selection_carre(lat : float, lon : float) -> list[float] :
    """
    Returns a list of coordinates that delimit a square of half size 250km, centered around (lat, long)

    Parameters
    ----------
    lat : float
        Latitude of the center point of the meteorological data
    long : float
        Longitude of the center point of the meteorological data

    Returns
    -------
    list[float]
        List of coordinates that delimit a square of half size 250km, centered around (lat, long)
    
    """
    
    lat_max = min(lat + (taillecarre * 180 / (Rterre * np.pi)), 90)
    lat_min = max(lat - (taillecarre * 180 / (Rterre * np.pi)), -90)

    lat_rad = lat * np.pi / 180
    lon_delta = taillecarre * 180 / (Rterre * np.pi * np.cos(lat_rad))
    lon_max = (lon + lon_delta + 180) % 360 - 180
    lon_min = (lon - lon_delta + 180) % 360 - 180

    return [lon_min, lon_max, lat_min, lat_max]


def initialize_api():
    """
    Initializes the Earth Engine API. 
    Requires to have a Google Earth 
    """
    # initialize ee API library
    ee.Initialize()
    # initialize the wxee
    wxee.Initialize()

 
def get_meteo(lat : float, long : float, date : str, time : int, plot=False) -> list[xr.Dataset] :
    """
    Returns a list of xarrays, each of them containing meteorological data from different models (ERA5 and GEOS at the moment).
    Pixels size is different from one model to another.
    Meteorological data stored is surface pressure and wind speed components
    
    Parameters
    ----------
    lat : float
        Latitude of the center point of the meteorological data
    long : float
        Longitude of the center point of the meteorological data
    date : str
        Date of the meteorological data
        Eg. : "2023-06-27"
    time : int
        Time of the day for the meteorological data
        Eg. : 16 for 15:00
    plot : bool
        Plots the meteorological data if True


    Returns
    -------
    DATA : list[xr.Dataset]
        List of xarrays, each of them contains data related to a meteorological model (ERA5 and GEOS at the moment)
    
    """
    hour = time
    ext = selection_carre(lat, long)
    rectangle = [[ext[0],ext[2]],
            [ext[1],ext[2]],
            [ext[1],ext[3]],
            [ext[0],ext[3]],
            [ext[0],ext[2]]]

    DATA = []

    for sat in SATS:

        name = SATELLITES[sat][0]
        resolution = SATELLITES[sat][1]
        ind_pressure = SATELLITES[sat][2]
        ind_u, ind_v = SATELLITES[sat][3], SATELLITES[sat][4]

        ee_rect = ee.Geometry.Polygon(rectangle, None, False)
        datenext = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        ee_date1 = ee.Date(date)
        ee_date2 = ee.Date(datenext)
    
        collection_filtered_sorted = (ee.ImageCollection(name)
                                .filterBounds(ee_rect)
                                .filterDate(ee_date1, ee_date2)
                                .select([ind_pressure, ind_u, ind_v])).sort('system:time_start')

        ds = collection_filtered_sorted.wx.to_xarray(region=ee_rect, scale=resolution, crs='EPSG:4326', # crs = 4326 is lat,lon projection
                            masked=True, nodata=-999999)
        
        ds_time = ds.isel(time=hour)
        data = ds_time[[ind_pressure, ind_u, ind_v]]
        data = data.rename({"x": "longitude", "y": "latitude"})

        DATA.append(data)


    if plot:

        fig, axes = plt.subplots(
            1, 3,
            figsize=(15, 5),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        # Set the super title (move down `top` to make space)
        fig.suptitle(f"{sat} - Time: {date}T{time}h", fontsize=16)
        fig.subplots_adjust(top=0.85)  # Leave space for suptitle

        # Titles and data indices for the subplots
        titles = ["Pressure (Pa)", "Eastward wind (m/s)", "Northward wind (m/s)"]
        data_indices = [ind_pressure, ind_u, ind_v]

        # Plot each subplot
        for ax, title, idx in zip(axes, titles, data_indices):
            ax.set_title(title, fontsize=12, pad=15)
            data[idx].plot(
                x='longitude',
                y='latitude',
                antialiased=True,
                transform=ccrs.PlateCarree(),
                ax=ax,
                add_colorbar=True,
                add_labels=False
            )

        plt.tight_layout(w_pad=2)
        plt.show()
    
    return DATA


def interp_to_tropomi(DATA : list[xr.Dataset], grid : xr.Dataset):
    """
    Interpolates meteorological data from DATA xarrays to each pixel in grid

    Parameters
    ----------
    DATA : list[xr.Dataset]
        List of xarrays, each of them contains data related to a meteorological model (ERA5 and GEOS at the moment)
    grid : xr.Dataset
        Xarray provided by the reading of TROPOMI data, with a different meshgrid

    Returns
    -------
    grid : xr.Dataset
       Added meteorological variables contained in DATA to each pixel and for each model.
        
    """
    
    for i in range(len(DATA)):
        
        sat = SATS[i]
        data = DATA[i]

        ind_pressure = SATELLITES[sat][2]
        ind_u, ind_v = SATELLITES[sat][3], SATELLITES[sat][4]


        newdata = data.interp(
                longitude=grid.longitude,
                latitude=grid.latitude,
            method="linear"
        )

        def pressure(scan, ground):
            out = newdata[ind_pressure].sel(scanline=scan, ground_pixel=ground)
            return out

        def uwind(scan, ground):
            out = newdata[ind_u].sel(scanline=scan, ground_pixel=ground)
            return out

        def vwind(scan, ground):
            out = newdata[ind_v].sel(scanline=scan, ground_pixel=ground)
            return out


        grid[f'surface_pressure{i}'] = xr.apply_ufunc(
            pressure,
            grid.scanline,
            grid.ground_pixel,
            input_core_dims=[[], []],
            output_core_dims=[[]],
            vectorize=True
        )

        grid[f'northward_wind{i}'] = xr.apply_ufunc(
            uwind,
            grid.scanline,
            grid.ground_pixel,
            input_core_dims=[[], []],
            output_core_dims=[[]],
            vectorize=True
        )

        grid[f'eastward_wind{i}'] = xr.apply_ufunc(
            vwind,
            grid.scanline,
            grid.ground_pixel,
            input_core_dims=[[], []],
            output_core_dims=[[]],
            vectorize=True
        )

    return grid


def get_fitted_meteo(grid : xr.Dataset) -> xr.Dataset:
    """
    Function that adds meteorological data to grid, pixel by pixel.
    
    Parameters
    ----------
    grid : xr.Dataset
        Xarray provided by the reading of TROPOMI data, with a different meshgrid

    Returns
    -------
    grid : xr.Dataset
       Added meteorological variables contained in DATA to each pixel and for each model.
        
    """
    lat = grid.attrs['latitude']
    lon = grid.attrs['longitude']
    date = grid.attrs['date']
    time = grid.attrs['time'].split(":")[0]  # Récupère l'heure 

    initialize_api()
    grid = grid.squeeze()  # Ensure grid is a 2D array
    return interp_to_tropomi(
        get_meteo(lat, lon, date, time, plot=False), 
        grid)


def tracer_variable(data, var : str, lat : float, long : float):
    """
    Plots data.var, centered on (lat, long)
    Parameters
    ----------
    donnees : xr.Dataset
        Data to be plotted

    """
    lon_min, lon_max, lat_min, lat_max = selection_carre(lat, long)
    img_extent = (lon_min, lon_max, lat_min, lat_max)

    fig = plt.figure(figsize = (7, 7))
    ax = plt.subplot(1,1,1,projection=ccrs.PlateCarree())
    ax.set_extent(img_extent, crs = ccrs.PlateCarree())
    ax.coastlines('10m', linewidth=1.5, color='black')
   
    data[var].plot(x='longitude',
                           y='latitude',
                           antialiased=True,
                           transform=ccrs.PlateCarree()
                           )
   
    plt.show()

