import xarray as xr

METHANE_COL = "methane_mixing_ratio_bias_corrected_destriped"
FACTOR = 1.25

def is_plume(methane, moy, std):
    return (methane > (moy + FACTOR * std))

def plume_mask(dataset):
    """
    Rajoute une colonne 'plume_mask' à dataset, indiquant si le pixel fait partie ou non d'un panache
    """
    moy, std = dataset[METHANE_COL].mean(), dataset[METHANE_COL].std()
    dataset["plume_mask"] = xr.apply_ufunc(is_plume, dataset.methane_mixing_ratio_bias_corrected_destriped, moy, std)