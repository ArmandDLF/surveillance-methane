import ee
import wxee
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy 
import xarray as xr

from codepropre import donnees
grid = donnees
square = 250 #size of the square that will be studied in kilometers
Rterre = 6378

def selection_carre(lat, lon):
    
    lat_max = min(lat + (square * 180 / (Rterre * np.pi)), 90)
    lat_min = max(lat - (square * 180 / (Rterre * np.pi)), -90)

    lat_rad = lat * np.pi / 180
    lon_delta = square * 180 / (Rterre * np.pi * np.cos(lat_rad))
    lon_max = (lon + lon_delta + 180) % 360 - 180
    lon_min = (lon - lon_delta + 180) % 360 - 180

    return [lon_min, lon_max, lat_min, lat_max]



# initialize ee API library
ee.Initialize()
# initialize the wxee
wxee.Initialize()
"""The function now works, and allows the user to get the necessary meteo data, i.e. pressure and 10m-wind components around a (lat, long) point.
A plotting function has been implemented via a boolean argument to allow visual representations."""

SATS = ["ERA5", "GEOS", "GFS"]
SATELLITES = {}
SATELLITES["ERA5"]  = ['ECMWF/ERA5_LAND/HOURLY', 11132, "surface_pressure", "u_component_of_wind_10m", "v_component_of_wind_10m"] 
#name, pixel_resolution, pressure, east wind component, north wind component
SATELLITES["GEOS"] = ['NASA/GEOS-CF/v1/rpl/tavg1hr', 27750, "PS", "U10M", "V10M"]
SATELLITES["GFS"] =["NOAA/GFS0P25", 27830, None, "u_component_of_wind_10m_above_ground", "v_component_of_wind_10m_above_ground"]
 

def get_meteo_geos(lat : float, long : float, sat_grid, date="2023-06-27", time=16, plot=False):

    hour = time
    ext = selection_carre(lat, long)
    rectangle = [[ext[0],ext[2]],
            [ext[1],ext[2]],
            [ext[1],ext[3]],
            [ext[0],ext[3]],
            [ext[0],ext[2]]]

    #TEST WITH GEOS

    name = 'NASA/GEOS-CF/v1/rpl/tavg1hr'
    resolution = 27750
    ind_pressure = 'PS'
    ind_u, ind_v = 'U10M', 'V10M'

    ee_rect = ee.Geometry.Polygon(rectangle, None, False)
    datenext="2023-06-28"
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


    # Interpolate data to the sat_grid's longitude and latitude
    data_interp = data.interp(
        x=np.sort(sat_grid["longitude"].values.flatten()),
        y=np.sort(sat_grid["latitude"].values.flatten()),
        method="linear"
    )

    if plot:
        fig, axes = plt.subplots(
            1, 3,
            figsize=(15, 5),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        fig.suptitle(f"GEOS - Time: {date}T{time}h", fontsize=16)
        fig.subplots_adjust(top=0.85)  # Leave space for suptitle

        titles = ["Pressure (Pa)", "Eastward wind (m/s)", "Northward wind (m/s)"]
        data_indices = [ind_pressure, ind_u, ind_v]

        for ax, title, idx in zip(axes, titles, data_indices):
            ax.set_title(title, fontsize=12, pad=15)
            data_interp[idx].plot(
                x='x',
                y='y',
                antialiased=True,
                transform=ccrs.PlateCarree(),
                ax=ax,
                add_colorbar=True,
                add_labels=False
            )

        plt.tight_layout(w_pad=2)
        plt.show()

    return data_interp



data = get_meteo_geos(-25.6, 28.67, grid, plot=True)
print(data)




def get_meteo_era(lat, long, date, time, plot=False):

    square = 250 #size of the square that will be studied in kilometers
    delta_angle =  (square/6378)*(180/np.pi)
    # Clamp latitude to [-90, 90]
    lat_min = max(-90, lat - delta_angle)
    lat_max = min(90, lat + delta_angle)

    # Longitude delta depends on latitude (converges at poles)
    if abs(lat) + delta_angle >= 90:
        # At or near the poles, longitude is undefined, so use full range
        long_min = -180
        long_max = 180
    else:
        delta_long = delta_angle / np.cos(np.radians(lat))
        long_min = (long - delta_long + 180) % 360 - 180
        long_max = (long + delta_long + 180) % 360 - 180

    ext = [long_min, long_max, lat_min, lat_max] 
    rectangle = [[ext[0],ext[2]],
            [ext[1],ext[2]],
            [ext[1],ext[3]],
            [ext[0],ext[3]],
            [ext[0],ext[2]]]

    data = []

    #TEST WITH ERA5
    SAT = SATELLITES["ERA5"]
    name = SAT[0]
    resolution = SAT[1]
    ind_pressure = SAT[2]
    ind_u, ind_v = SAT[3], SAT[4]

    ee_rect = ee.Geometry.Polygon(rectangle, None, False)
    ee_date = ee.Date(date)

    collection_filtered_sorted = (ee.ImageCollection(name)
                            .filterBounds(ee_rect)
                            .filterDate(ee_date)
                            .select([ind_pressure, ind_u, ind_v])).sort('system:time_start')
    
    ds = collection_filtered_sorted.wx.to_xarray(region=ee_rect, scale=resolution, crs='EPSG:4326', # crs = 4326 is lat,lon projection
                        masked=True, nodata=-999999)
    

    data = ds[[ind_pressure, ind_u, ind_v]]

    if plot:

        fig, axes = plt.subplots(
            1, 3,
            figsize=(15, 5),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        # Set the super title (move down `top` to make space)
        fig.suptitle(f"ERA5 - Time: {date}T{time}h", fontsize=16)
        fig.subplots_adjust(top=0.85)  # Leave space for suptitle

        # Titles and data indices for the subplots
        titles = ["Pressure (Pa)", "Eastward wind (m/s)", "Northward wind (m/s)"]
        data_indices = [ind_pressure, ind_u, ind_v]

        # Plot each subplot
        for ax, title, idx in zip(axes, titles, data_indices):
            ax.set_title(title, fontsize=12, pad=15)
            data[idx].plot(
                x='x',
                y='y',
                antialiased=True,
                transform=ccrs.PlateCarree(),
                ax=ax,
                add_colorbar=True,
                add_labels=False
            )

        plt.tight_layout(w_pad=2)
        plt.show()


    return data
    
data_era = get_meteo(0, 20, "2025-06-10", 12, plot=True)
print(data_era)


