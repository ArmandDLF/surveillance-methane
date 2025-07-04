import xarray as xr
import numpy as np
import scipy as sc
from codepropre import distance_harvesine

METHANE_COL = "methane_mixing_ratio_bias_corrected_destriped"
MAX_ITERATION = 20


def source(dataset):
    """
    Renvoie la ligne (le pixel) la plus proche de la source du dataset
    """
    
    lat_s, lon_s = dataset.attrs['latitude_source'], dataset.attrs['longitude_source']
    dataset["dist"] = xr.apply_ufunc(distance_harvesine, lat_s, lon_s, dataset.latitude, dataset.longitude)

    argmin = dataset["dist"].argmin(...)
    dataset.drop_vars(["dist"])
    return dataset.isel(argmin)


def plume_mask(dataset, facteur=1.25):
    """
    Rajoute une colonne 'plume_mask' à dataset, indiquant si le pixel fait partie ou non d'un panache
    """
    moy, std = dataset[METHANE_COL].mean(), dataset[METHANE_COL].std()
    sans_outliers = dataset[METHANE_COL].where(abs(dataset[METHANE_COL] - moy) < 2*std)
    moy_nouv, std_nouv = sans_outliers.mean(), sans_outliers.std()

    src = source(dataset)
    src_mask = dataset.longitude == src.longitude.data
    src_mask = src_mask.latitude == src.latitude.data

    candidats = (dataset[METHANE_COL] > (moy_nouv + facteur * std_nouv)) | src_mask
    dataset["candidats"] = candidats
    dataset["source"] = src_mask
    
    expansion = sc.ndimage.generate_binary_structure(2, 2) # Autorise les coins
    dataset["plume_mask"] = src_mask
    old = xr.zeros_like(dataset["plume_mask"]) # rempli de zéro mais bonne taille)
    i = 0
    
    while (dataset["plume_mask"] ^ old).any() and i < MAX_ITERATION:
        old = dataset["plume_mask"].copy()
        dataset["plume_mask"] = ("scanline","ground_pixel"), \
            sc.ndimage.binary_dilation(dataset["plume_mask"], structure=expansion)
        dataset["plume_mask"] = ("scanline","ground_pixel"), \
            np.where(candidats, dataset["plume_mask"], 0)
        i += 1