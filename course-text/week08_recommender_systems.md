# Week 8 — Recommender Systems

> Notebooks: `clustering/recommender_systems/recommender_system_lesversie.ipynb`, `recommender_system2.ipynb`

## Wat zijn recommender systems?

- Overal op het internet te vinden (streamingplatformen, digitale platformen met abonnementsformules).
- Een RS probeert op basis van vorige informatie **goede, nieuwe keuzes aan te bieden**. Klassiek voorbeeld: filmvoorkeuren.

### Notatie

- Users: $u$; producten: $i, j, \dots$; ratings: $r$; de rating van user $u$ over film $i$: $r_{u,i}$

### Feedback

- **Expliciet**: likes, reviews, sterren, … → hoge kwaliteit, maar niet overal beschikbaar
- **Impliciet**: bezochte producten, aangeklikte films maar niet gespeeld, … → lage kwaliteit, maar in heel grote mate verkrijgbaar
- Dit is waarom je een kortingscode kunt krijgen/winnen door een review te schrijven!

### Ratingsmatrix

- Feedback wordt in een matrix R gestoken.
- Expliciete versie: typisch waarden tussen 0 en 1 — zo niet, eerst **normaliseren**.
- Impliciete versie: enkel waarden 0 en 1.

## Strategieën

### 1. Popular Based

- Niet gebaseerd op een individuele user, maar op algemene patronen ("most watched in Belgium this week").
- Dit is niet ML — kan met standaard SQL. Maar voor een nieuwe gebruiker zonder voorkennis is dit het meest zinvolle wat je kan doen.

### 2. Content Based

- Op basis van een bekeken film worden films voorgesteld met **gelijkaardige features** (zelfde regisseur, genre).
- Nadeel: één goede film van een genre → plots een hele hoop films van dat genre; iedereen die een film ziet, krijgt dezelfde suggesties.

### 3. Collaborative Filtering

- user-based en item-based — beide sterk gebaseerd op de ratingsmatrix.

## Content Based — implementatie

1. **Genres als aparte kolommen** uittrekken; verzamel alle genres (lowercase, zonder spaties — vermijdt *Fantasy* vs *fantasy*).
2. Genres zijn categorisch → **one-hot encoden** (per genre een 1/0-kolom via `np.where(...contains(genre)...)`).
3. **Similariteitsfunctie**: `cosine_similarity(item_genre_mat)` — diagonaal is 1, matrix is symmetrisch; cosine similarity berekent per rijenpaar het product gedeeld door de normen → getal tussen 0 en 1.
4. Gegeven een gezien film (rij): zoek de kolommen met de grootste waarde → de meest verwante films.

```python
from sklearn.metrics.pairwise import cosine_similarity
corr_mat = cosine_similarity(item_genre_mat)
```

### Similarity metrics

- **Pearson correlatie**: correlatie tussen 2 variabelen (zie Mathematical Foundations).
- **Cosine similarity**: cosinus van de hoek tussen vectoren → dicht bij 1 als de dingen analoge features hebben.
- **Time-decay latent Dirichlet Allocation** en andere geavanceerde opties.
- Wij gebruiken cosine similarity — één van de meest eenvoudige.

## Collaborative Filtering — memory based

1. Vind een groep vergelijkbare gebruikers (groepsgrootte willekeurig) op basis van de metriek.
2. Bereken het gemiddelde van de beoordeling van elk item binnen die groep.
3. Rangschik op aflopend gemiddelde en beveel aan wat de doelgebruiker nog nooit eerder gezien heeft.

### Implementatie

- Initialiseer een **ijle matrix** (veel nullen; bv. maar 0.35% is niet 0):

```python
from scipy.sparse import csr_matrix
mat = csr_matrix((data, (row, col)), shape=(NUM_USERS, NUM_ITEMS))
mat.eliminate_zeros()
```

  Enkel de niet-nul items worden aangeduid — véél efficiënter dan de hele matrix bijhouden.
- Rijen = alle gebruikers, kolommen = de items (anders dan bij content based).
- Transponeer (`mat.T`) — items moeten als kolom — en pas de similariteitsfunctie toe:

```python
item_corr_mat = cosine_similarity(mat.T)
```

- Het resultaat verschilt van content based: de kolommen zijn nu niet genres, maar alle mogelijke gebruikers.

### Schaalbaarheidsprobleem

- Bij een nieuwe gebruiker moet de hele matrix herberekend worden, voor alle gebruikers — **niet schaalbaar**; op den duur past zelfs een ijle matrix niet meer in memory.

## Collaborative Filtering — model based

- De grote matrices kleiner maken met **TruncatedSVD** (equivalent aan PCA, de dimensie-reductietechniek).

```python
from sklearn.decomposition import TruncatedSVD
epsilon = 1e-9        # tegen numerieke instabiliteit bij SVD, en om geen nul-data te hebben
n_latent_factors = 10 # aantal dimensies/items dat je wil overhouden
item_svd = TruncatedSVD(n_components=n_latent_factors)
item_features = item_svd.fit_transform(mat.transpose()) + epsilon
item_corr_mat = cosine_similarity(item_features)
```

- Ook dit bereikt uiteindelijk een limiet; in de oefeningen kijken we naar andere libraries (installeer alvast `scikit-surprise` en `scipy`).

## Nadelen van recommender systems

- Je kan al snel in een soort **tunnelvisie** terechtkomen (test het op YouTube) — een gevaarlijk voorbeeld van hoe goedbedoelde technologie fake news e.d. in de hand kan werken.

## Beste praktijken (Week 8)

- Normaliseer de ratingsmatrix naar [0,1] of gebruik impliciete 0/1-feedback.
- Gebruik **sparse matrices** (`csr_matrix`) — ratingsmatrices zijn 99%+ nullen.
- Zet genres/categorieën eerst naar lowercase/zonder spaties vóór one-hot encoding.
- Nieuwe gebruiker zonder historiek (cold start)? Val terug op popular-based aanbevelingen.
- Maak je bewust van de tunnelvisie/fake-news-risico's van aanbevelingsalgoritmes.

## Kernpunten

- Strategieën: popular based (geen ML), content based (features van items), collaborative filtering (ratingsmatrix).
- Content based: one-hot genres + cosine similarity; collaborative: user-item matrix transponeren + cosine similarity.
- Memory-based schaalt niet → model-based met TruncatedSVD (equivalent aan PCA).