import ee
import wxee
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy 

# initialize ee API library
ee.Initialize()
# initialize the wxee
wxee.Initialize()


SATS = ["ERA5", "GEOS", "GFS"]
SATELLITES = {}
SATELLITES["ERA5"]  = ['ECMWF/ERA5_LAND/HOURLY', 11132, "surface_pressure", "u_component_of_wind_10m", "v_component_of_wind_10m"] 
#name, pixel_resolution, pressure, east wind component, north wind component
SATELLITES["GEOS"] = ['NASA/GEOS-CF/v1/rpl/tavg1hr', 27750, "PS", "U10M", "V10M"]
SATELLITES["GFS"] =["NOAA/GFS0P25", 27830, None, "u_component_of_wind_10m_above_ground", "v_component_of_wind_10m_above_ground"]


def get_meteo(lat, long, date, time, plot=False):


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


    """
    ###TEST FOR ALL SATS
    for sat in SATS:
        
        name = SATELLITES[sat][0]
        resolution = SATELLITES[sat][1]
        ind_pressure = SATELLITES[sat][2]
        ind_u, ind_v = SATELLITES[sat][3], SATELLITES[sat][4]

        ee_rect = ee.Geometry.Polygon(rectangle, None, False)
        ee_date = ee.Date(date)

        collection_filtered_sorted = (ee.ImageCollection(name)
                              .filterBounds(ee_rect)
                              .filterDate(ee_date)
                              .select([resolution])).sort('system:time_start')
        
        ds = collection_filtered_sorted.wx.to_xarray(region=ee_rect, scale=resolution, crs='EPSG:4326', # crs = 4326 is lat,lon projection
                            masked=True, nodata=-999999)
        
        if ind_pressure != None:
            data.append(ds[ind_pressure])
        data.append(ds[ind_u])
        data.append(ds[ind_v])
    """
    #TEST WITH GEOS

    name = 'NASA/GEOS-CF/v1/rpl/tavg1hr'
    resolution = 27750
    ind_pressure = 'PS'
    ind_u, ind_v = 'U10M', 'V10M'

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

        fig = plt.figure()
        ax = plt.subplot(1,3,1,projection=ccrs.PlateCarree())
        ax.set_ylabel("Pressure (Pa)")
        data[ind_pressure].plot(x='x',
                    y='y',
                    antialiased =False,
                    transform=ccrs.PlateCarree())
        
        ax = plt.subplot(1, 3, 2, projection=ccrs.PlateCarree())
        ax.set_ylabel("Eastward wind (m/s)")
        data[ind_u].plot(x='x', 
                            y='y', 
                            antialiased=True, 
                            transform=ccrs.PlateCarree())
        
        ax = plt.subplot(1,3,3,projection=ccrs.PlateCarree())
        ax.set_ylabel("Northward wind (m/s)")
        data[ind_v].plot(x='x', 
                        y='y',
                        antialiased=True,
                        transform=ccrs.PlateCarree())
        
        plt.show()

        data.

    return data

print(get_meteo(45, 5, "2025-06-10", 12))



