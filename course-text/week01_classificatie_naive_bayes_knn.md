# Week 1 — Classificatie deel 1: Naive Bayes en KNN

> Notebooks: `classification/naive bayes/naive_bayes_oplossingen.ipynb`, `classification/knn/knn_oplossingen.ipynb`

## Herhaling ML-basisbegrippen

- **Classificatie** vs **clustering**; **supervised** (begeleid) vs **unsupervised** (onbegeleid).
- **Underfitting / overfitting**.
- **Train/test/validation data**, **cross-validatie**.

## Scikit-learn

De standaard ML-bibliotheek voor Python — gebouwd op NumPy & SciPy.

- **Bekendste ML-bibliotheek** voor Python.
- **Wat kan het?**
  - Classificatie (o.a. Naive Bayes, KNN, SVM, …)
  - Regressie
  - Clustering
  - Dimensionality reduction
  - Preprocessing & feature engineering
  - Model selectie & evaluatie (cross-validatie, metrics)
- **Open source** (BSD-licentie) — gratis, ook voor commercieel gebruik; ontwikkeld door INRIA (Frans onderzoeksinstituut) en de community; volledige broncode op GitHub; actieve community, uitgebreide documentatie.

### Consistent API: één patroon voor élk algoritme

Elk schattingsalgoritme ("estimator") in scikit-learn volgt dezelfde structuur:

```python
from sklearn.xxxx import SomeModel

model = SomeModel(hyperparameter=waarde)  # 1. Maak model
model.fit(X, y)  # 2. Leer uit data
model.predict(X_new)  # 3. Voorspel
model.score(X_test, y_test)  # 4. Evalueer
```

| Methode | Gebruik | Voorbeeld |
|---|---|---|
| `.fit(X, y)` | Train het model | `knn.fit(X_train, y_train)` |
| `.predict(X)` | Voorspel labels | `knn.predict(X_test)` |
| `.score(X, y)` | Bereken nauwkeurigheid | `knn.score(X_test, y_test)` |
| `.transform(X)` | Transformeer data (preprocessing) | `scaler.transform(X_test)` |
| `.fit_transform(X)` | Fit + transform in één stap | `scaler.fit_transform(X_train)` |

> **Opgelet:** `.fit()` en `.predict()` zijn **altijd** beschikbaar voor supervised modellen; `.transform()` enkel voor preprocessing / unsupervised. Eén patroon leren = **alle** algoritmes in deze cursus kunnen gebruiken!

## Naive Bayes

> **Naive Bayes — historische noot:** Thomas Bayes (1701–1761), Engelse predikant en wiskundige — zijn stelling werd **postuum** gepubliceerd in 1763 door zijn vriend Richard Price. Pierre-Simon Laplace (1749–1827) herontdekte en generaliseerde Bayes' werk → de **Bayes–Laplace-stelling**. Eerste ML-toepassing: M.E. Maron (1961) gebruikte Naive Bayes voor automatische indexering van documenten. In de jaren 1990–2000 werd Naive Bayes de **standaard voor spamfilters** (Sahami et al., 1998; Graham, 2002). De "naïeve" onafhankelijkheidsaanname is al sinds de jaren 1960 als *onrealistisch* gekend — toch blijft de classifier verrassend goed presteren.

### Conditionele (voorwaardelijke) kans

- $P(A|B)$: de kans op A als B al is gebeurd.
- Definitie: $P(A|B) = \dfrac{P(A \& B)}{P(B)}$
- Voorbeeld: kans op twee keer zes gooien = $1/36 = P(A\&B)$; kans op één keer zes = $1/6 = P(B)$; dus $P(A|B) = \frac{1/36}{1/6} = \frac{1}{6}$.

### Regel van Bayes

$$P(A|B) = \frac{P(B|A)\,P(A)}{P(B)}$$

- In ML-context: B = data/feiten, A = het classificatielabel dat we willen toekennen. We willen dus $P(\text{label} \mid \text{data})$ berekenen.

### Spamfilter voorbeeld

Gegeven: $P(\text{moneytransfer} \mid \text{spam}) = 0.4$, $P(\text{moneytransfer} \mid \neg\text{spam}) = 0.0001$, $P(\text{spam}) = 0.45$.

$$P(\text{spam} \mid \text{mt}) = \frac{P(\text{mt} \mid \text{spam})\,P(\text{spam})}{P(\text{mt}\mid\text{spam})P(\text{spam}) + P(\text{mt}\mid\neg\text{spam})P(\neg\text{spam})}$$

Numeriek ingevuld:

$$P(\text{spam} \mid \text{mt}) = \frac{0.4 \times 0.45}{0.4 \times 0.45 \;+\; 0.0001 \times 0.55} \approx 0.9997$$

**Besluit:** $P(\text{spam} \mid \text{mt}) \approx 99.97\%$ → de classifier kiest het label met de grootste kans → label = **spam**. 🚫

### Meerdere woorden screenen

Met meerdere tokens ($mo$=moneytransfer, $ug$=uganda, $ap$=AP Hogeschool):

$$P(\neg mo \& ug \& ap \mid \text{spam}) = P(\neg mo \mid \text{spam})\,P(ug \mid \text{spam})\,P(ap \mid \text{spam})\cdots$$

- Idem voor $\neg$spam; kies uiteindelijk het label met de grootste teller (noemer hoeft niet uitgerekend te worden).

### Waarom "naief"?

- De kansen van woorden samen en apart zijn niet onafhankelijk — ze beïnvloeden elkaar in werkelijkheid.
- In de praktijk werkt de classifier vrij goed, dus we negeren dit.

### Laplace-smoothing (uit notebook)

Als $P(x_i \mid y=k) = 0$ voor een label, wordt het hele product 0. Oplossing: tel een klein getal $\delta$ op:

$$P(x_i \mid y=k) = \frac{\delta + \#(x_i, y=k)}{\sum_j (\delta + \#(x_j, y=k))}$$

$\delta$ wordt bepaald in functie van de documentgrootte en kan met cross-validatie worden geschat.

### Implementatie (sklearn)

```python
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X, Y)
clf.predict(X[2:3])
```

Let op de `.fit`/`.predict`-structuur — dit komt overal terug.

## K-Nearest Neighbours (KNN)

> **KNN — historische noot:** Fix & Hodges (1951) beschreven als eersten de "nearest neighbor"-beslissingsregel — een **niet-parametrische** classifier, zonder aannames over de onderliggende verdeling. Cover & Hart (1967) bewezen dat de fout van 1-NN met voldoende data hoogstens **2× de Bayes-optimale fout** bedraagt. "Lazy learning" was al in de jaren 1950 gekend: geen echte trainingsfase, alle data wordt opgeslagen. Vanaf ~1990–2010 breed toegepast in patroonherkenning, beeldclassificatie en aanbevelingssystemen (collaborative filtering). KNN is één van de **oudste** en **simpelste** ML-algoritmes — maar nog steeds relevant!

### Basisidee

- Gelijkaardige dingen zijn "dicht bij elkaar".
- Algoritme:
  1. Kies K (aantal buren).
  2. Voor een nieuw punt: bereken de K dichtste datapunten.
  3. Het meest voorkomende label van die K punten is de voorspelling.

### Eigenschappen en aandachtspunten

- Supervised; zowel binary als multiclass.
- Werkt enkel als er een notie van "afstand" is:
  - enkel numerieke data → **normaliseren is verplicht!**
  - of zelf een metriek definiëren voor categorische data (bv. vogelvluchtafstand tussen steden).
- K moet op voorhand gekozen worden:
  - **Te kleine K** → gevoelig voor ruis (overfitting)
  - **Te grote K** → te veel middeling (underfitting)
  - **Oplossing:** kies K via **cross-validatie**, niet "op het gevoel".
- **Training** kost nauwelijks tijd (enkel data opslaan) ✅; **classificatie** is duur (alle afstanden berekenen) ❌.
- KNN is een **lazy learner** — er is geen echte trainingsfase.

- Visueel (k=5): vindt het algoritme 4×rood en 1×groen, dan is de voorspelling rood.

### Implementatie (sklearn)

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
```

## Beste praktijken (Week 1)

- **Normaliseer** je features vóór KNN — afstanden zijn anders vertekend.
- Kies K via cross-validatie, niet "op het gevoel".
- Gebruik `.fit`/`.predict` via scikit-learn in plaats van zelf te implementeren, tenzij de oefening het vraagt.
- Check dat categorische features een bruikbare metriek hebben of one-hot encoded worden.

## Kernpunten

- Naive Bayes: Bayes' regel + aanname van onafhankelijke features (Laplace-smoothing tegen nulkansen).
- KNN: lazy learner — geen training, trage predictie, afstandsgebaseerd.
- Beide zijn supervised classifiers; beide vereisen goede datavoorbereiding.