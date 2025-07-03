import codepropre # Traitement des données SRON
import meteo # Traitement des données Google Earth Engine
import plume_mask # Détection des panaches de méthane

import os
import xarray as xr


""" Paramètres globaux """

skip_traitement = True # Si False, nécessite de télécharger les données de TROPOMI
ajout_donnes_mto = False


""" Général """

if not skip_traitement:

    codepropre.panaches()

# lire chaque fichier dans ./work_data/traite/
files = os.listdir("./work_data/traite/")
files = [f for f in files if f.endswith('.nc')]


sources_emissions_sron = []
sources_emissions_cal = []
sources_incertitudes_sron = []
sources_incertitudes_cal = []

for i, file in enumerate(files):

    print(f"Traitement du fichier {i+1}/{len(files)} : {file}")
    dataset = xr.open_dataset(file)
    
    if ajout_donnes_mto:
        # Récupère wind et pression
        pass

    plume_mask.plume_mask(dataset)

    # Calcul émissions et incertitudes

    sources_emissions_sron.append(dataset.attrs['source_rate'])
    sources_incertitudes_sron.append(dataset.attrs['incertitude_source'])



