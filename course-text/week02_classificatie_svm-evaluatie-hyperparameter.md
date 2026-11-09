# Week 2 — Classificatie deel 2: Support Vector Machines

> Notebooks: `classification/svm/support_vector_machines.ipynb`, `classification/pipelines/Titanic_pipeline.ipynb`

## Herhaling KNN

- Voordeel: zeer snelle training.
- Nadeel: bij grote datasets trage classificatie, omdat er veel berekend moet worden voor een nieuw datapunt.

## Large margin classification

- Startprobleem: lineair separabele data. We zoeken een **lineaire** scheiding (=> heel snelle classificatie later).
- "Beste" optie: de scheiding tussen de twee uiterste cases zo **groot mogelijk** — *large margin classification*. Dit lukt helaas niet altijd.

```python
from sklearn.svm import SVC
svm_clf = SVC(kernel="linear", C=0.25)
svm_clf.fit(X, y)
```

```python
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    # Op het beslissingsvlak: w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]
    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    svs = svm_clf.support_vectors_
    plt.plot(x0, decision_boundary, "k-", linewidth=2, zorder=-2)
    plt.plot(x0, gutter_up, "k--", linewidth=2, zorder=-2)
    plt.plot(x0, gutter_down, "k--", linewidth=2, zorder=-2)
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#AAA', zorder=-1)
```

- De marge wordt bepaald door enkel de punten **op de rand** — de **support vectors**.
> Code ter **illustratie**, niet te kennen — mag je altijd gebruiken indien nodig.

## Soft-margin classification

- Bij niet-separabele data: compromis tussen zo breed mogelijke marge en zo weinig mogelijke "overtredingen" van die marge.
- Parameter **C**: hoe kleiner C, hoe meer fouten worden getolereerd.
  - Lage C → risico underfitting
  - Hoge C → risico overfitting

## Niet-lineaire data: kernels

- Voorbeeld: *moons*-dataset — niet separabel, maar er zit structuur in.
  1. Lineair SVM: geen goede fit.
  2. Lineair SVM + polynomiale features (tot 3e graad) toegevoegd als kolommen: beter.
  3. SVM met similariteitsfuncties (punten die "op mekaar lijken"): beter.
- **Doel van de kernel**: zorgen dat de dataset in een hogere dimensie wél separabel is. Concreet voor de moons-dataset: je wil minstens **één extra dimensie** zodat er nu wél een **lineaire scheiding** (een vlak) gemaakt kan worden. Eens de kernel gekozen, doet de classifier zelf al het andere werk: hij werkt dan alsof hij een lineaire classifier in hogere dimensie is.
- **Similariteitsfuncties**: we kijken naar punten die 'dicht bij mekaar liggen' (bv. de Gaussische RBF).
- **Kernel trick**: efficiënter omdat de kernel (een non-lineaire operator) niet uitgerekend wordt, maar enkel de **inproducten** tussen die features — dankzij geavanceerde wiskunde kan dit efficiënt gebeuren.

### Wanneer SVM gebruiken?

- **Niet-lineaire datasets** — in de praktijk is bijna niets "vanzelf" lineair.
- **Niet al te grote datasets**.
- Ze zijn onderhevig aan **scaling** — gebruik zeker een `StandardScaler`:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

svm_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1))
```

## Evaluatie van modellen

| | Regressie | Classificatie |
|---|---|---|
| Standaard | $R^2$ | Cross-validatie |
| Fouten | MAE / MSE / RMSE | Confusion matrix (zeker bij scheve labelverdeling), Precision, Recall, F1 |
| Aandachtspunt | geen specifieke | multilabel classificatie en labelverdeling |

- **Cross-validation** (K-fold): zie `scikit-learn.org/stable/modules/cross_validation.html`.
- **Confusion matrix**: toont per klasse echt vs voorspeld label; ziet meteen of een bepaalde klasse slechter voorspeld wordt. Ideaal: diagonaal bevat zoveel mogelijk data.
- **Precision** $= TP / (\text{totaal } P)$, **Recall** $= TP / (\text{totaal echt } P)$, **F1** = harmonisch gemiddelde ervan.

## Hyperparameter search en preprocessing

- **GridSearchCV / RandomizedSearchCV**: automatisch de beste waarde van parameters vinden (bv. C):

```python
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(svm_clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
grid_search.best_estimator_
```

- **Niet-numerieke data** (labels, categorieën — geen onderling verband): gebruik **One-Hot encoding** en vervang de categorieën door de geëncodeerde kolommen:

```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first')
encoded_categorical = enc.fit_transform(X_categorical)
```

- **Pipelines**: scalen, train-test split, model trainen, grid search, … kan allemaal in één `Pipeline`.

## Beste praktijken (Week 2)

- **Schaal je data (StandardScaler) vóór SVM** — SVM's zijn onderhevig aan scaling.
- Stem C af via (Grid/Randomized)SearchCV met cross-validatie; lage C = toleranter, hoge C = meer overfitting-risico.
- Gebruik Pipelines zodat preprocessing en model altijd dezelfde stappen doorlopen (voorkomt data leakage).
- Evalueer classificatie met confusion matrix + precision/recall/F1, niet enkel accuracy, zeker bij scheve labels.
- Encode categorische features met One-Hot encoding (met `handle_unknown='ignore'`).

## Kernpunten

- SVM zoekt de grootst mogelijke marge; soft margin (C) tolereert overtredingen.
- Kernels maken niet-lineaire data lineair separabel in hogere dimensie, efficiënt via de kernel trick.
- Evaluatie: R²/MAE/MSE/RMSE voor regressie; confusion matrix, precision/recall/F1, cross-validatie voor classificatie.
- One-Hot encoding + Pipelines zijn standaardgereedschap voor realistische datasets.