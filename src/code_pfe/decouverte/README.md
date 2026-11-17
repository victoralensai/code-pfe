# Découverte : FedAvg sur PathMNIST

Cette première expérience sert à comprendre et à relire chaque étape du fédéré.
Elle utilise PyTorch directement, sans Flower, FedML ni autre bibliothèque FL.
Les commentaires et ce guide sont en français.

## Installer avec uv

Depuis la racine du projet, dans votre environnement `.venv` :

```bash
uv pip install torch torchvision numpy medmnist matplotlib
uv pip install pytest  # seulement pour les tests
```

`uv pip install` installe dans l'environnement sans modifier `pyproject.toml`.
Les commandes suivantes utilisent `uv run --no-sync` pour garder ces paquets sans
resynchroniser l'environnement avec les dépendances encore vides du projet.
`PYTHONPATH=src` rend le code importable sans réinstaller le projet.

Pour une machine **sans GPU NVIDIA**, installer plutôt les versions CPU de
PyTorch, puis les autres dépendances (cela évite les paquets CUDA volumineux) :

```bash
uv pip install torch torchvision --index https://download.pytorch.org/whl/cpu
uv pip install numpy medmnist matplotlib pytest
```

## Lancer

Un premier essai sur 1 000 images d'entraînement, les dix hôpitaux et les trois
graines, avec seulement deux tours :

```bash
PYTHONPATH=src uv run --no-sync -m code_pfe.decouverte.experiment \
  --max-train 1000 --rounds 2 --output outputs/decouverte-rapide
```

Le téléchargement initial est celui du fichier PathMNIST complet, même pour cet
essai réduit. Il est conservé dans `data/medmnist/` pour les prochains lancements.
Une connexion Internet est nécessaire seulement pour ce téléchargement et
l'installation. Le test officiel reste entier dans le mode rapide.

L'expérience complète demandée :

```bash
PYTHONPATH=src uv run --no-sync -m code_pfe.decouverte.experiment \
  --clients 10 --rounds 30 --local-epochs 1 --alpha 0.5 --seeds 0 1 2 \
  --output outputs/decouverte-complete
```

Ajouter `--device cuda` si PyTorch dispose de CUDA. Le CPU est le choix par défaut,
avec deux threads ; adapter `--threads` si nécessaire. Le calcul complet comprend
six entraînements de trente tours : ce n'est pas le même coût que l'essai rapide.
Choisir un nouveau dossier de sortie pour chaque configuration. Le programme
refuse d'écraser un dossier non vide. Il n'y a pas de reprise automatique.

## Le protocole

- **Données** : PathMNIST, images RGB 28 × 28, neuf classes ; séparation officielle
  train/test. Les pixels passent de `[0, 255]` à `[-1, 1]`, sans augmentation.
- **IID** : permutation aléatoire du train, puis dix parts presque égales. Les
  proportions de classes sont proches en moyenne, sans être forcées identiques.
- **Non-IID** : pour chaque classe, on tire dix proportions selon
  `Dirichlet(0.5, ..., 0.5)`, puis on répartit ses images selon ces proportions
  par un tirage multinomial. Les proportions de classes et les tailles locales
  varient. On retire la partition entière si un client est vide (1 000 essais max).
- **Modèle** : deux convolutions (16 puis 32 canaux), deux max-poolings, puis deux
  couches linéaires (64 puis 9). Pas de BatchNorm ni de dropout.
- **Local** : SGD, taux d'apprentissage constant `0.01`, batch `64`, une époque
  locale par tour, aucun momentum. Tous les mini-batches, même incomplets, comptent.
- **Participation** : les dix hôpitaux participent à tous les tours, séquentiellement
  sur une seule machine. Un tour comprend dix apprentissages puis une agrégation.
- **Comparaison** : graines `0, 1, 2` ; modèle initial identique entre les deux
  scénarios d'une même graine. Les graines changent l'initialisation, la partition
  et le mélange des batches. Le sous-ensemble rapide est fixé séparément par
  `--subset-seed 12345` et reste commun aux six entraînements.
- **Mesure** : accuracy et entropie croisée sur tout le test au tour zéro puis
  après chaque agrégation. Aucun choix de modèle ou arrêt anticipé sur le test.
  Le split validation n'est pas utilisé ici ; l'utiliser pour régler les futurs
  hyperparamètres, plutôt que de choisir ceux qui maximisent le test.

## Comprendre FedAvg

Au tour `t`, le serveur possède les paramètres `w_t` et chaque hôpital `k` possède
`n_k` images. Les dix hôpitaux reçoivent **le même** `w_t`, font leurs pas de SGD,
puis renvoient leurs nouveaux paramètres `w_(t+1,k)`. Le serveur calcule :

```text
w_(t+1) = somme_k (n_k / somme_j n_j) × w_(t+1,k)
```

Exemple : un hôpital de 100 images et un de 300 images pèsent respectivement
`0.25` et `0.75`. Une moyenne simple des deux modèles serait incorrecte ici.
On moyenne les **paramètres après apprentissage**, pas les prédictions ni les
gradients. Au tour suivant, chaque hôpital repart du nouveau modèle global.

Pour lire le code dans l'ordre :

1. `fedavg.py` : le CNN, l'apprentissage d'un client, la moyenne et l'évaluation.
2. `data.py` : construire les hôpitaux à partir du train.
3. `experiment.py` : retrouver diffusion → apprentissages locaux → agrégation.
4. `plots.py` : passer du CSV aux courbes.

Les clients partagent physiquement la mémoire du processus de simulation : ce
code reproduit les échanges algorithmiques, pas un déploiement réseau sécurisé.
La Dirichlet simule une hétérogénéité des labels et des effectifs, pas des
différences réelles de scanners, de patients ou de pratiques hospitalières.
FedAvg seul ne fournit pas de garantie de confidentialité.

## Résultats à examiner

Dans chaque dossier de sortie :

```text
config.json                 # paramètres, versions, tailles, commit et état Git
metrics.csv                 # split, seed, round, test_accuracy, test_loss, elapsed_seconds
accuracy.png                # trois courbes par scénario + moyenne ± écart-type
iid_seed_0/                 # même structure pour les six couples split/graine
  partitions.npz            # client_0...client_9 : indices dans le train officiel
  class_counts.json         # matrice hôpital × classe des effectifs
  partitions.png            # proportions de classes et effectifs par hôpital
  model.pt                  # paramètres finaux du SmallCNN
```

L'accuracy du CSV est entre 0 et 1 ; la figure l'affiche en pourcentage. La bande
représente l'écart-type empirique entre les graines (`ddof=1`), pas un intervalle
de confiance. Les résultats réduits portent explicitement la mention « essai
réduit » dans le titre. Ils servent à vérifier le pipeline, pas à conclure.

Comparer la vitesse de progression, la dispersion entre graines et les valeurs
finales. Examiner les cartes des partitions pour relier les courbes aux données.
On ne suppose pas que le non-IID sera systématiquement moins bon : cela dépend
du tirage et des hyperparamètres. Trente tours est un point de départ, pas une
garantie de convergence. Ne pas inventer de conclusion avant d'avoir les mesures.
Les mêmes graines et versions sur une même machine permettent de reproduire les
calculs ; l'identité numérique entre matériels ou versions n'est pas garantie.

Retracer un CSV complet sans relancer l'apprentissage :

```bash
PYTHONPATH=src uv run --no-sync -m code_pfe.decouverte.plots \
  outputs/decouverte-complete/metrics.csv
```

Vérifier les propriétés essentielles (sans télécharger les données) :

```bash
PYTHONPATH=src uv run --no-sync -m pytest tests/test_decouverte.py -q
```

## Sources

- [McMahan et al., AISTATS 2017 — FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html).
- [MedMNIST : données et publication](https://medmnist.com/) et
  [API officielle](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/dataset.py).
- [PyTorch : reproductibilité](https://docs.pytorch.org/docs/stable/notes/randomness.html).

La répartition Dirichlet est le protocole demandé pour cette expérience ; elle
n'est pas présentée comme la partition originale du papier FedAvg.
