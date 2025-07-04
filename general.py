import codepropre # Traitement des données SRON
import meteo # Traitement des données Google Earth Engine
import prime # Calcul émission et incertitudes

import os
import xarray as xr


""" Paramètres globaux """

skip_traitement = True # Si False, nécessite d'avoir les données de TROPOMI en local
ajout_donnes_mto = False # Si True, nécessite de connecter le compte Google Earth Engine
donnes_tropomi = True # Si False, utilise les données de Bremen

""" Général """

if not skip_traitement:
    codepropre.panaches(donnes_tropomi)

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
        # Rajoute les colonnes de wind et pression
        meteo.get_fitted_meteo(dataset)

    # Calcul émissions et incertitudes
    sources_emissions_sron.append(dataset.attrs['source_rate'])
    sources_incertitudes_sron.append(dataset.attrs['incertitude'])

    emi , inc =  prime.emission_rate_with_uncertainties(dataset, 10)
    sources_emissions_cal.append(emi)
    sources_incertitudes_cal.append(inc)


# Plot des émissions calculées en fonction des émissions SRON avec incertitudes
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.errorbar(sources_emissions_sron, sources_emissions_cal, 
             xerr=sources_incertitudes_sron, yerr=sources_incertitudes_cal, 
             fmt='o', capsize=5, capthick=1, elinewidth=1, markersize=6, 
             color='blue', ecolor='lightblue', label='Données')

# Add a diagonal line for reference (perfect correlation)
min_val = min(min(sources_emissions_sron), min(sources_emissions_cal))
max_val = max(max(sources_emissions_sron), max(sources_emissions_cal))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Corrélation parfaite')

plt.xlabel('Émissions SRON (t/h)')
plt.ylabel('Émissions Calculées (t/h)')
plt.title('Comparaison des émissions: SRON vs Calculées')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()