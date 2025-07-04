# Projet "Surveillance des sources de méthane par repérages et mesures satellitaires"


## Capacité numériques mobilisés

Notre projet consiste en la manipulation et le croisement de différents formats de données de mesures satellitaires et météorologiques via le module xarray de python. L'objectif est de parvenir à extraire des données d'intérêts parmi une large banque de données et de les mettre sous un format qui permet leur traitement analytique et calculatoire.

## Contexte scientifique

Aujourd'hui, la détection de fuites de méthane par voies satellitaires est importantes pour minimiser l'accroissement des effets nocifs du réchauffement climatique, et pour des raisons de sécurité publique. Cependant, leur détection manque encore de crédibilité aux yeux de certains organismes extérieurs qui contredisent les déductions fournies par les mesures. 
De sorte à améliorer cette crédibilité, il est important de faire converger les déductions qualitatives et quantitatives d'organismes différents. Notre projet cherche donc à répondre à cet objectif en comparant les flux de méthanes rapportés par les données fournies par le SRON (Space Research Organisation Netherlands) néerlandais et l'institut français de Brême en Allemagne. 
Pour cela, une implémentaion de la méthode de calcul par intégration discrète à été appliquée aux données traitées issues de ces sources.

## Modules python requis

pandas
random
xarray
matplotlib.pyplot
os
numpy
cartopy.crs
scipy

## Utilisation 

Le fichier `general.py` est le seul qu'il est nécessaire d'ouvrir pour obtenir les courbes de comparaison.

Ce fichier comprend trois paramètres principaux d'intérêt : 

- skip_traitement est un booléen qui indique s'il faut traiter les données de TROPOMI (satellite de SRON) difficilement accessibles - sinon des version déjà générées sont utilisables  
- ajout_donnes_mto est un booléen qui indique s'il faut télécharger ou non les données météo du Google Earth Engine. Cette étape est nécessaire dans le traitement des données de Brême, mais déjà existante pour les données de TROPOMI  
- donnes_tropomi est un booléen qui indique si l'on souhaite utiliser les données de TROPOMI ou celles de Brême pour le traitement.

## Fichiers et utilité

`codepropre.py` : Extraction et mise en forme des données brutes (étape 1)
`etape1_bremen.py` : De même mais pour celles de Brême avec un format différent (étape 1 bis)
`destriping.py` : Filtre appliqué sur les données pour enlever le phénomène de "stripes"
`meteo.py` : Récupération des données météorologiques de Google Earth Engine (étape 2)
`plume_mask.py` : Masque du panache de méthane (étape 3)
`prime.py` : Calcul des émissions et incertitudes associées (étape 4)
`general.py` : Workflow complet, dont le plot de comparaison (étape 5)