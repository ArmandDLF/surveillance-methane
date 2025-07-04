import codepropre # Traitement des données SRON
# import meteo # Traitement des données Google Earth Engine
import prime # Calcul émission et incertitudes

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
files = ["./work_data/traite/" + f for f in files if f.endswith('.nc')]


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

    # Calcul émissions et incertitudes

    sources_emissions_sron.append(dataset.attrs['source_rate'])
    sources_incertitudes_sron.append(dataset.attrs['incertitude'])

    emi , inc =  prime.emission_rate_with_uncertainties(dataset, 30)
    sources_emissions_cal.append(emi)
    sources_incertitudes_cal.append(inc)


# Plot des émissions calculées en fonction des émissions SRON avec incertitudes
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.errorbar(sources_emissions_sron, sources_incertitudes_sron, fmt='o', label='SRON', color='blue')
plt.errorbar(sources_emissions_cal, sources_incertitudes_cal, fmt='o', label='Calculé', color='red')
plt.xlabel('Émissions (t/h)')
plt.ylabel('Incertitudes (t/h)')
plt.title('Émissions et incertitudes SRON vs Calculées')
plt.plot()