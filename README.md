# Projet "Surveillance des sources de méthane par repérages et mesures satelitaires"


Capacité numériques mobilisés : 

Notre projet consiste en la manipulation et le croisement de différents formats de données de mesures satelitaires et météorologiques via le module xarray de python. L'objectif est de parvenir à extraire des données d'intérêts parmi une large banque de données et de les mettre sous un format qui permet leur traitement analytique et calculatoire.

Contexte scientifique :

Aujourd'hui, la détection de fuites de méthane par voies satelitaires est importantes pour minimiser l'accroissement des effets nocifs du réchauffement climatique, et pour des raisons de sécurité publique. Cependant, leur détection manque encore de crédibilité aux yeux de certains organismes extérieurs qui contredisent les déductions fournies par les mesures. 
De sorte à améliorer cette crédibilité, il est important de faire converger les déductions qualitatives et quantitatives d'organismes différents. Notre projet cherche donc à répondre à cet objectif en comparant les flux de méthanes rapportés par les données fournies par le SRON (Space Research Organisation Netherlands) néerlandais et l'institut français Bremen en Allemagne. 
Pour cela, une implémentaion de la méthode de calcul par intégration discrète à été appliquée aux données traitées issues de ces sources.

Modules python requis : 

pandas
random
xarray
matplotlib.pyplot
os
numpy
cartopy.crs
destriping