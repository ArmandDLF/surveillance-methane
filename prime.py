import numpy as np
import xarray as xr 
import random as rd

###################################################### PARTIE DE LOUAN #################################################################


''' Constant hyperparameters '''

ic = 3 # Index of the dataset to use, change this to use another dataset
path = '\\Users\\Mon PC\\Documents\\Projet-info\\example_data\\data\\example_data_'+'%d'%ic+'.nc''' # Path to the dataset, change this to use another dataset
ds = xr.open_dataset(path) 


METHANE_COL = "methane_mixing_ratio_bias_corrected_destriped" # Name of the methane column in the dataset, change this if you use another dataset

alpha_1 = 0.5 # Coefficient for wind speed calculation

FACTOR = 1.25

pressure_std = ds.surface_pressure.std() # Standard deviation of surface pressure in the entire scene
methane_std = ds.methane_mixing_ratio_bias_corrected_destriped.std() # Standard deviation of methane mixing ratio in the entire scene
alpha_1_min = alpha_1*(1 - 0.05) # Minimum coefficient for wind speed calculation
alpha_1_max = alpha_1*(1 + 0.05) # Maximum Coefficient for wind speed calculation

FACTOR_MIN = 1.3 # Minimum factor for plume belonging
FACTOR_MAX = 2.3 # Maximum factor

M_CH4 = 16.04 # Molar mass of methane in g/mol
M_air = 28.97 # Molar mass of air in g/mol

g = 9.81 # Acceleration due to gravity in m/s^2

R = 6378000 # Radius of the Earth in meters

'''Some useful functions to calculate the area of a pixel'''

def distance_harvesine(lat1d, lon1d, lat2d, lon2d):
    """
    Calculate the distance between two points on the Earth using the Haversine formula.
    """
    lat1=lat1d*np.pi/180
    lat2=lat2d*np.pi/180
    lon1=lon1d*np.pi/180
    lon2=lon2d*np.pi/180

    return 2*R*np.arcsin(np.sqrt((np.sin((lat2-lat1)/2))**2 + np.cos(lat1)*np.cos(lat2)*(np.sin((lon2 - lon1)/2))**2))

vdistance_harvesine = np.vectorize(distance_harvesine)

def area_calculation(latitude_bounds, longitude_bounds):
    """
    Calculate the area of a pixel in square meters based on latitude and longitude.
    """

    lat1 = latitude_bounds[0,:,:]
    lon1 = longitude_bounds[0,:,:]
    lat2 = latitude_bounds[1,:,:]
    lon2 = longitude_bounds[1,:,:]
    lat3 = latitude_bounds[2,:,:]
    lon3 = longitude_bounds[2,:,:]

    return vdistance_harvesine(lat1, lon1, lat2, lon2) * vdistance_harvesine(lat2, lon2, lat3, lon3)

def add_pixel_area(ds): 
    """
    Add a pixel area column to the dataset. It calculate the area of each pixel in the dataset
    This will create a new variable 'pixel_area' in the dataset
    """
    ds['pixel_area'] = ("scanline", "ground_pixel"), area_calculation(ds.latitude_bounds, ds.longitude_bounds)

    return ds

'''Let's build the mask'''

def is_plume(methane, moy, std):
    return (methane > (moy + FACTOR * std))

def add_plume_mask(ds):
    """
    Add a column 'plume_belonging' to the dataset, indicating if the pixel belongs to a methane plume.
    This function calculates also the mean and standard deviation of the methane mixing ratio
    """
    # Calculate the mean and standard deviation of the methane mixing ratio

    moy, std = ds[METHANE_COL].mean(), ds[METHANE_COL].std()
    ds["plume_belonging"] = xr.apply_ufunc(is_plume, ds.methane_mixing_ratio_bias_corrected_destriped, moy, std)

    return ds

def plume_area(ds):
    """
    Calculate the Area of the plume in square meters
    """
    return (ds.plume_belonging * ds.pixel_area).sum() # Area of the plume in m^2

def wind_speed():
    """
    Calculate wind speed from 10m wind data
    """
    
    out = alpha_1*(ds.northward_wind**(2) + ds.eastward_wind**(2))**(1/2)
    return out

def add_wind_speed(ds):
    """
    Add a column 'wind_speed' to the dataset, indicating the wind speed in m/s
    """
    ds['wind_speed'] = xr.apply_ufunc(wind_speed) # Calculate the wind speed from the 10m wind data
    
    return ds

'''Let's calculate the emission rate'''

def calculate_emission_rate(ds):
    """
    Calculate the emission rate in tons/hour
    """
    ds_configured = add_plume_mask(add_pixel_area(add_wind_speed(ds)))

    mask = ds_configured.plume_belonging # Mask for the plume pixels

    concentration_rate_mean_ppb = ds_configured.methane_mixing_ratio_bias_corrected_destriped.where(~mask).mean() # Mean methane concentration in the scene without the plume in ppb
    
    wind_speed_mean = ds_configured.wind_speed.where(mask).mean()*(3.6*1e3) # Mean wind speed in the plume in m/h
    
    Q = ((wind_speed_mean)/(np.sqrt(plume_area(ds_configured))))*(ds_configured.pixel_area.where(mask)*ds_configured.surface_pressure.where(mask)*(ds_configured.methane_mixing_ratio_bias_corrected_destriped.where(mask) - concentration_rate_mean_ppb)).sum()*1e-9*M_CH4*1e-3/(g*M_air) # Emission rate in tons/hour
    
    return "The emission rate is", Q.data

def emission_rate_with_uncertainties(ds, n):
    """
    Calculate the emission rate in tons/hour with uncertainties
    """
    ds_configured = add_pixel_area(add_wind_speed(ds))

    wind_speed_max = ds_configured.wind_speed.mean() + ds_configured.wind_speed.std()# Minimum wind speed in the entire scene
    wind_speed_min = ds_configured.wind_speed.mean() - ds_configured.wind_speed.std()# Maximum wind speed in the entire scene

    Q_liste = np.array([])

    for i in range(n):

        FACTOR_rd = rd.uniform(FACTOR_MIN, FACTOR_MAX) # Random factor for plume belonging

        ds_configured = add_plume_mask(ds_configured) #Creation of the plume mask with random factor

        Plume_area_rd = (ds_configured.pixel_area*ds_configured.plume_belonging).sum() # Area of the plume in m^2

        alpha_1_rd = rd.uniform(alpha_1_min, alpha_1_max) # Random coefficient for wind speed calculation

        wind_speed_rd = rd.uniform(wind_speed_min, wind_speed_max)*(3.6*1e3) # Random wind speed in the plume in m/h
        
        mask = ds.plume_belonging

        concentration_rate_mean_ppb = ds_configured.methane_mixing_ratio_bias_corrected_destriped.where(~mask).mean() # Mean methane concentration in the scene without the plume in ppb

        Q_liste = np.append(Q_liste, 
                        (alpha_1_rd*wind_speed_rd) # Random wind speed in m/h
                        /
                        (np.sqrt(Plume_area_rd)) #Random plume area in m^2
                        *
                        (ds.pixel_area.where(mask)
                         *
                         rd.uniform(ds.surface_pressure.where(mask) - pressure_std,
                               ds.surface_pressure.where(mask) + pressure_std) # Random surface pressure in Pa
                               *
                               (rd.uniform(ds[METHANE_COL].where(mask) - 2*methane_std, 
                               ds[METHANE_COL].where(mask) + 2*methane_std) # Random methane mixing ratio
                                - concentration_rate_mean_ppb)
                               ).sum()*1e-9*M_CH4*1e-3/(g*M_air)) # Emission rate in tons/hour
        
    return("Emission rate mean: ", np.mean(Q_liste),"+/-", np.std(Q_liste), "t/h")

print(emission_rate_with_uncertainties(ds, 200))