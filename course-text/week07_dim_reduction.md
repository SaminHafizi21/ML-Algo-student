# Week 6 — Dimensionality Reduction

> Notebooks: `dimensionality_reduction/dimensionality_reduction.ipynb`, `demo_facial_recognition.ipynb`

## Ethische opmerking

- De technieken in deze les zijn vaak gebruikt en legaal, maar kunnen onethisch gebruikt worden (bv. bevolkingscontrole in bepaalde landen).
- Gebruik **nooit facial recognition** op data van familie, vrienden, … zonder hun **expliciete toestemming**. Op het oneigenlijke gebruik van fotomateriaal bestaat wetgeving: dat mag in België niet zomaar.

## Features zijn dimensies

- Elk feature is een dimensie van de dataruimte; te veel dimensies brengt problemen mee:
  - Mensen zijn niet getraind in het interpreteren van meer dan 3 dimensies.
  - Bij heel veel dimensies zijn er heel veel mogelijkheden voor de datapunten — **groot risico op overfitting** ("the curse of dimensionality": er is in een hoog-dimensionale ruimte heel veel 'plaats').

## De oplossing: dimensies verlagen

- Doel: zoveel mogelijk **variatie (= 'informatie')** van de dataset bijhouden.
- Bv. schrap f1 in plaats van f2 of f3; of maak een nieuwe combinatie van f2 en f3 als nieuwe kolom en houd enkel die bij.

## Afbeeldingen: flattening

- Zwart-witfoto's zijn matrices (m rijen × n kolommen, pixelwaarde 0–256).
- Een foto (matrix) moet eerst naar een datapunt (vector) omgezet worden: alle kolommen na elkaar → 1 lange rij. De data blijft identiek, pixels verschuiven alleen.

```python
.reshape(-1)
```

- Bv. 125 × 94 pixels → een rij van 11750 waardes (dimensies).
- Een dataset = alle foto's als rijen onder elkaar; elke kolom = één pixel op dezelfde locatie in alle foto's.

```python
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=5, resize=1)
X = people.data
n_features = X.shape[1]
y = people.target
```

- PCA flattened de data automatisch — dat moet je niet zelf doen.

## PCA (Principal Component Analysis)

- Zoekt de meest **dominante richtingen** in een dataset en geeft die terug als **principal components**: hoe minder je er vraagt, hoe sterker de datareductie. Alle oorspronkelijke datapunten kunnen worden uitgedrukt als lineaire combinatie van die componenten.
- Achterliggend (voor 3 dimensies): de data ligt in de buurt van een vlak; PCA berekent automatisch het beste vlak om op te projecteren, de projecties van alle datapunten, en binnen dat vlak de dominante richtingen — de **eigenvectoren**.
- Voorbeelfoto's: de eigenvectoren ("eigenfaces") zijn **generieke gezichten**; elk gezicht is een lineaire combinatie van deze eigenfaces (bv. Kim Clijsters $= \sum_{i=1}^{200} c_i \cdot \text{eigenface}_i$).
- PCA behoudt zoveel mogelijk de **variantie**; de componenten staan in volgorde van belangrijkheid. De reductie = de laatste componenten "laten vallen" (weinig waarde).

```python
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)
eigenvectors = pca.components_.reshape((n_components, h, w))

# of: laat PCA zelf het aantal kiezen voor 95% van de variatie
pca = PCA(n_components=0.95, svd_solver='randomized', whiten=True).fit(X_train_scaled)
```

### Nadelen en varianten

- **Heel de dataset moet in memory** → oplossing: `IncrementalPCA`.
- PCA werkt met **lineaire projecties** → voor complexere datasets: `KernelPCA` met kernels zoals bij SVM.

## LLE (Locally Linear Embedding)

- Voor niet-lineaire datasets is PCA minder geschikt — bv. de **'swiss roll'**: een opgerold blad dat lokaal op een 2D-vlak lijkt.
- PCA zou die structuur verliezen: na projectie zou data van verschillende plaatsen op dezelfde regio landen.
- LLE is vooral geschikt voor "opgerolde datasets": punten die oorspronkelijk dicht bij mekaar lagen, blijven dicht bij mekaar (**distance-preserving op lokale schaal**) — op grotere afstand blijven afstanden echter níet bewaard.
- Werking (niet te kennen):
  1. Zoek voor elk punt de k dichtstbijzijnde buren.
  2. Schrijf elk punt als lineaire functie van die k buren (lokaal lineair modelleren) in een matrix W.
  3. Projecteer naar een lager-dimensionale ruimte met behoud van die lokale relaties.

```python
from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_unrolled = lle.fit_transform(X_swiss)
```

## Beste praktijken (Week 6)

- Reduceer dimensies bij veel features: het vermindert het overfit-risico en bewaart zoveel mogelijk variantie.
- **Schaal je data vóór PCA** (`X_train_scaled`); gebruik `n_components=0.95` om automatisch 95% variatie te behouden.
- Dataset past niet in memory? Gebruik `IncrementalPCA`.
- Niet-lineaire structuren (swiss roll e.d.): gebruik LLE of KernelPCA, niet gewone PCA.
- Respecteer privacy: geen facial recognition zonder expliciete toestemming.

## Kernpunten

- Curse of dimensionality: te veel features → overfit-risico.
- Flattening zet afbeeldingen om naar vectoren; PCA zoekt dominante richtingen (eigenfaces bij gezichten).
- PCA = lineair + in memory; IncrementalPCA/KernelPCA/LLE lossen die beperkingen op.