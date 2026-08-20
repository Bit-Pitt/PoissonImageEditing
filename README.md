

# Poisson Image Editing


Implementazione del paper:  [Poisson Image Editing](https://dl.acm.org/doi/10.1145/882262.882269) paper by Perez et al. 2003.


- Sono state implementate tutte le diverse applicazioni descritte dal paper ovvero:
    - seamless cloning      [cloning_test.sh]
    - Seamless cloning con mixing gradients  [mixed_test.sh]
    - Texture flattening        [flattening_test.sh]
    - Local illumination change  [illumination_test.sh]
    - Local color change    [colorChange.sh]
    - Seamless tiling       [tiling_test.sh]

Ognuna di queste versioni testabile tramite lo script shell sopra elencato.

Aggiunta extra personale:
- feathering + Poisson: per una transizione più dolce tra target e source con l'idea di preservare il colore originale dell'oggetto   [featherCloning.sh]


Cartella "create_test":
- contiene due utility per la creazione veloce di una cartella di test:
- resize.py per ridimensionare l'immagine "source.jpeg" alle dimensioni di "target.jpeg".
- create_mask per generare la mask da GUI

