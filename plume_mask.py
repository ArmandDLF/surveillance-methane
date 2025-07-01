import xarray as xr
import numpy as np
import scipy as sc

METHANE_COL = "methane_mixing_ratio_bias_corrected_destriped"
FACTEUR = 1.25
MAX_ITERATION = 20


def dist2(lat1, lat2, lon1, lon2):
    """
    Distance naïve entre les coordonnées
    """
    return (lat1 - lat2)**2 + (lon1 - lon2)**2


def source(dataset):
    """
    Renvoie la ligne (le pixel) la plus proche de la source du dataset
    """
    
    lat_s, lon_s = dataset.attrs['SRON plume source lat,lon']
    dataset["dist2"] = xr.apply_ufunc(dist2, lat_s, dataset.latitude, lon_s, dataset.longitude)

    print("*"*20)
    argmin = dataset["dist2"].argmin(...)
    dataset.drop_vars(["dist2"])
    return dataset.isel(argmin)


def plume_mask(dataset):
    """
    Rajoute une colonne 'plume_mask' à dataset, indiquant si le pixel fait partie ou non d'un panache
    """
    moy, std = dataset[METHANE_COL].mean(), dataset[METHANE_COL].std()
    candidats = dataset[METHANE_COL] > (moy + FACTEUR * std)

    src = source(dataset)
    src_mask = dataset.longitude == src.longitude.data
    src_mask = src_mask.latitude == src.latitude.data

    dataset["candidats"] = candidats
    dataset["source"] = src_mask
    
    expansion = sc.ndimage.generate_binary_structure(2, 2) # Autorise les coins
    dataset["plume_mask"] = src_mask
    old = xr.zeros_like(dataset["plume_mask"]) # rempli de zéro mais bonne taille)
    i = 0
    
    while (dataset["plume_mask"] ^ old).any() and i < MAX_ITERATION:
        old = dataset["plume_mask"]
        dataset["plume_mask"] = ("scanline","ground_pixel"), \
            sc.ndimage.binary_dilation(dataset["plume_mask"], structure=expansion)
        dataset["plume_mask"] = ("scanline","ground_pixel"), \
            np.where(candidats, dataset["plume_mask"], 0)
        i += 1