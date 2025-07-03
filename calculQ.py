import xarray as xr

''' Hyperparameters '''

alpha_1 = 0.5 # Coefficient for wind speed calculation
alpha_2 = 1.0 # Coefficient for wind speed calculation

M_CH4 = 16.04 # Molar mass of methane in g/mol
M_air = 28.97 # Molar mass of air in g/mol

g = 9.81 # Acceleration due to gravity in m/s^2

A = 5000*7500 # Area in m^2

METHANE_COL = "methane_mixing_ratio_bias_corrected_destriped"
FACTOR = 1.25

''' Load the dataset '''

ds = xr.open_dataset('data.nc')

'''Let's build the mask'''

def is_plume(methane, moy, std):
    return (methane > (moy + FACTOR * std))

def plume_mask(dataset):
    """
    Rajoute une colonne 'plume_mask' à dataset, indiquant si le pixel fait partie ou non d'un panache
    """
    moy, std = dataset[METHANE_COL].mean(), dataset[METHANE_COL].std()
    dataset["plume_belonging"] = xr.apply_ufunc(is_plume, dataset.methane_mixing_ratio_bias_corrected_destriped, moy, std)


'''Let's calculate the wind speed'''

def wind_speed(wind10):

    '''Calculate wind speed from 10m wind data.'''
    
    out = alpha_1* wind10 + alpha_2
    return out

ds['wind_speed'] = xr.apply_ufunc(wind_speed, # first function name
                                        ds.wind10) # then arguments

'''Let's calculate the emission rate'''

mask = ds.plume_belonging
mass_mean = ds.pressure[1 - mask]*ds.concentration_rate[1 - mask]*M_CH4/(g*M_air).mean() # Mean methane mass in the entire scene

Q = (wind_speed)*A*xr.sum(ds.pressure[mask]*ds.concentration_rate[mask]*M_CH4/(g*M_air) - mass_mean) #The emission rate

print (Q)
