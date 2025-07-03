import xarray as xr
import numpy as np
from scipy.ndimage import median_filter


def filtre_mediane_nan(data, size, axis):
    """Filtre mediane 1D filter sur l'axe donné, en ignorant les nan."""
    # Move the axis to the first dimension
    data_echang = np.moveaxis(data, axis, 0)
    resultat = np.empty_like(data_echang)

    for idx in range(data_echang.shape[1]):
        col = data_echang[:, idx]
        mask_nan = np.isnan(col)

        if np.all(mask_nan):
            resultat[:, idx] = np.nan
        else:
            filtree = median_filter(np.where(mask_nan, np.nanmedian(col), col), size=size, mode='nearest')
            resultat[:, idx] = np.where(mask_nan, np.nan, filtree)

    return np.moveaxis(resultat, 0, axis)
        
        
def destripe(ds):
    """
    Destriping des données : rajoute la colonne 'methane_mixing_ratio_bias_corrected_destriped'
    à l'objet xarray Dataset ds.
    Algo : https://www.mdpi.com/2072-4292/16/7/1208 Section 3.2
    """
    w1 = 7
    w2 = 20

    v = ds.methane_mixing_ratio_bias_corrected

    # Smoothing dans direction des bandes (axis=1)
    b = xr.apply_ufunc(
        filtre_mediane_nan, v, w1, 1,
        input_core_dims=[['scanline', 'ground_pixel'], [], []],
        output_core_dims=[['scanline', 'ground_pixel']],
    )

    r = v - b

    # Smoothing du residual dans direction du vol (axis=0)
    s = xr.apply_ufunc(
        filtre_mediane_nan, r, w2, 0,
        input_core_dims=[['scanline', 'ground_pixel'], [], []],
        output_core_dims=[['scanline', 'ground_pixel']],
    )

    ds["methane_mixing_ratio_bias_corrected_destriped"] = v - s