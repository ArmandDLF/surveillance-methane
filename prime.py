import numpy as np
import xarray as xr 
import random as rd
from plume_mask import plume_mask, METHANE_COL
from codepropre import distance_harvesine, Rterre

###################################################### PARTIE DE LOUAN #################################################################


''' Constant hyperparameters '''

alpha_1 = 0.5 # Coefficient for wind speed calculation

FACTOR = 1.25
alpha_1_min = alpha_1*(1 - 0.05) # Minimum coefficient for wind speed calculation
alpha_1_max = alpha_1*(1 + 0.05) # Maximum Coefficient for wind speed calculation

FACTOR_MIN = 1.3 # Minimum factor for plume belonging
FACTOR_MAX = 2.3 # Maximum factor

M_CH4 = 16.04 # Molar mass of methane in g/mol
M_air = 28.97 # Molar mass of air in g/mol

g = 9.81 # Acceleration due to gravity in m/s^2

R = Rterre*1e3 # Radius of the Earth in meters

'''Some useful functions to calculate the area of a pixel'''

vdistance_harvesine = np.vectorize(lambda x1,y1,x2,y2 : distance_harvesine(x1,y1,x2,y2) * 1e3) # Convert distance from km to m

def area_calculation(latitude_bounds, longitude_bounds):
    """
    Calculate the area of a pixel in square meters based on latitude and longitude.
    """

    lat1 = latitude_bounds[:,:,0]
    lon1 = longitude_bounds[:,:,0]
    lat2 = latitude_bounds[:,:,1]
    lon2 = longitude_bounds[:,:,1]
    lat3 = latitude_bounds[:,:,2]
    lon3 = longitude_bounds[:,:,2]

    return vdistance_harvesine(lat1, lon1, lat2, lon2) * vdistance_harvesine(lat2, lon2, lat3, lon3)

def add_pixel_area(ds): 
    """
    Add a pixel area column to the dataset. It calculate the area of each pixel in the dataset
    This will create a new variable 'pixel_area' in the dataset
    """
    ds['pixel_area'] = ("scanline", "ground_pixel"), area_calculation(ds.latitude_bounds, ds.longitude_bounds)


def plume_area(ds):
    """
    Calculate the Area of the plume in square meters
    """
    return (ds.plume_belonging * ds.pixel_area).sum() # Area of the plume in m^2


def add_wind_speed(ds):
    """
    Add a column 'wind_speed' to the dataset, indicating the wind speed in m/s
    """
    ds['wind_speed'] = (ds.northward_wind**(2) + ds.eastward_wind**(2))**(1/2)
    

'''Let's calculate the emission rate'''

def calculate_emission_rate(ds):
    """
    Calculate the emission rate in tons/hour
    """
    ds_configured = ds.copy()
    add_wind_speed(ds_configured)
    add_pixel_area(ds_configured)
    plume_mask(ds_configured)

    mask = ds_configured.plume_belonging # Mask for the plume pixels

    concentration_rate_mean_ppb = ds_configured.methane_mixing_ratio_bias_corrected_destriped.where(~mask).mean() # Mean methane concentration in the scene without the plume in ppb
    
    wind_speed_mean = ds_configured.wind_speed.where(mask).mean()*(3.6*1e3) # Mean wind speed in the plume in m/h
    
    Q = alpha_1*((wind_speed_mean)/(np.sqrt(plume_area(ds_configured))))*(ds_configured.pixel_area.where(mask)*ds_configured.surface_pressure.where(mask)*(ds_configured.methane_mixing_ratio_bias_corrected_destriped.where(mask) - concentration_rate_mean_ppb)).sum()*1e-9*M_CH4*1e-3/(g*M_air) # Emission rate in tons/hour
    
    return Q.data

def emission_rate_with_uncertainties(ds, n):
    """
    Calculate the emission rate in tons/hour with uncertainties
    """
    ds_configured = ds.copy()
    add_wind_speed(ds_configured)
    add_pixel_area(ds_configured)

    pressure_std = ds_configured.surface_pressure.std()
    methane_std = ds_configured.methane_mixing_ratio_bias_corrected_destriped.std()

    wind_speed_max = ds_configured.wind_speed.mean() + ds_configured.wind_speed.std()# Minimum wind speed in the entire scene
    wind_speed_min = ds_configured.wind_speed.mean() - ds_configured.wind_speed.std()# Maximum wind speed in the entire scene

    Q_liste = np.array([])

    for _ in range(n):

        factor_rd = rd.uniform(FACTOR_MIN, FACTOR_MAX) # Random factor for plume belonging

        plume_mask(ds_configured, facteur=factor_rd) #Creation of the plume mask with random factor

        Plume_area_rd = (ds_configured.pixel_area*ds_configured.plume_mask).sum() # Area of the plume in m^2

        alpha_1_rd = rd.uniform(alpha_1_min, alpha_1_max) # Random coefficient for wind speed calculation

        wind_speed_rd = rd.uniform(wind_speed_min, wind_speed_max)*(3.6*1e3) # Random wind speed in the plume in m/h
        
        mask = ds_configured.plume_mask

        concentration_rate_mean_ppb = ds_configured.methane_mixing_ratio_bias_corrected_destriped.where(~mask).mean() # Mean methane concentration in the scene without the plume in ppb

        Q_liste = np.append(Q_liste, 
                        (alpha_1_rd*wind_speed_rd) # Random wind speed in m/h
                        /
                        (np.sqrt(Plume_area_rd)) #Random plume area in m^2
                        *
                        (ds_configured.pixel_area.where(mask)
                         *
                         rd.uniform(ds_configured.surface_pressure.where(mask) - pressure_std,
                               ds_configured.surface_pressure.where(mask) + pressure_std) # Random surface pressure in Pa
                               *
                               (rd.uniform(ds_configured[METHANE_COL].where(mask) - 2*methane_std, 
                               ds_configured[METHANE_COL].where(mask) + 2*methane_std) # Random methane mixing ratio
                                - concentration_rate_mean_ppb)
                               ).sum()*1e-9*M_CH4*1e-3/(g*M_air)) # Emission rate in tons/hour
        
    return np.mean(Q_liste), np.std(Q_liste)
