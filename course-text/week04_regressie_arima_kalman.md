# Week 4 — Regressie en Forecasting (ARIMA, Kalman Filters)

> Notebooks: `week04basis_regressie.ipynb`, `week04arima_uitgebreide_versie.ipynb`; voor Kalman: `kalman_oefening_oplossing.ipynb`

## Herhaling: vorige week ensembles

- **Bagging / Random Forest:** parallel trainen van bomen op random subsets → minder variantie.
- **Boosting:** sequentieel, elk model corrigeert de vorige (AdaBoost, Gradient Boosting, XGBoost).
- **Stacking:** een meta-model ('blender') leert de ensembleleden combineren.
- Tot nu toe waren al onze modellen **classifiers** van gelabelde, categorieke data. Vandaag: wat als de output **numeriek** is in plaats van categorisch?

## Regressie

### Dezelfde algoritmes als regressor

- De modellen die we al kennen hebben ook een **Regressor-variant** in sklearn:

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

knn = KNeighborsRegressor(n_neighbors=5)
svr = SVR(kernel='linear')
rf  = RandomForestRegressor(n_estimators=100, random_state=42)
# elk: .fit(X_train, y_train); predictions = model.predict(X_test)
```

### Regressie evalueren

```python
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
```

- **MAE** = gemiddelde absolute fout.
- **MSE** = gemiddelde gekwadrateerde fout; de meest gekende — **zo laag mogelijk** is het best.
- $R^2$ = de absolute standaard.

| | Classificatie (vorige weken) | Regressie (vandaag) |
|---|---|---|
| Output | label | getal |
| Evaluatie | cross-validatie, confusion matrix | MAE / MSE / R² |

> **Probeer eerst dezelfde algoritmes als *Regressor*-variant** voor je iets complexers neemt.

## Tijdsreeksen

- Data uit tijdsreeksen komt heel vaak voor in (industriële) processen: meetwaarden van machines, logs, opvolging van winst doorheen de tijd, …
- Voor tijdsreeksen zijn er **speciale technieken** die de tijdsdimensie een speciale status geven.
- Vaak willen we een model waarmee we **patronen kunnen vinden** en/of **forecasting** doen: datapunten in de toekomst voorspellen.

### Lineaire regressie

- Een heel simpel model om time series te modelleren is lineaire regressie; zoals de naam zegt geeft dit **enkel een lineair verband**.
- Beperking: in veel datasets geen goede fit — bv. sinuspatronen, kwadratische trends, seizoenspatronen → we hebben een **algemener regressiemodel** nodig: ARIMA.

## ARIMA

**A**uto**R**egressive **I**ntegrated **M**oving **A**verage — een algemener regressiemodel met 3 parameters:

| Onderdeel | Parameter | Betekenis |
|---|---|---|
| AutoRegression | $p$ | hoeveel vorige datapunten meenemen |
| Integration | $d$ | hoe vaak differentiëren tot de reeks stationair is |
| Moving Average | $q$ | hoeveel vorige errorwaardes meenemen |

$$ \mathrm{ARIMA}(p, d, q) $$

- **Voorwaarden van het model**: geen cyclische trends, weinig outliers, de variatie rond het gemiddelde blijft consistent in functie van de tijd.

### p — AutoRegression

- Voorspelling op basis van de **vorige outputs** van het model (*lags*):

$$ y_t = \beta_0 + \beta_1 y_{t-1} + \beta_2 y_{t-2} + \ldots + \beta_{p} y_{t-p} $$

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df.value, order=(p, 0, 0))  # enkel AR-term
model_fit = model.fit()
```

| | $p$ hoog | $p$ laag |
|---|---|---|
| Geschiedenis | veel vorige datapunten meenemen | minder lange geschiedenis |
| Reactie | weinig responsief voor plotse wijzigingen | reageert sneller op plotse wijzigingen |

### q — Moving Average

- Voorspelling op basis van de **errorwaardes** (residuals) van het verleden:

$$ y_t = \Phi_0 + \Phi_1 \epsilon_{t-1} + \Phi_2 \epsilon_{t-2} + \ldots + \Phi_q \epsilon_{t-q} $$

| | $q$ hoog | $q$ laag |
|---|---|---|
| Fouten | veel vorige foutwaardes meenemen | weinig vorige foutwaardes meenemen |

> ⚠️ Laat je niet vangen door de term 'moving average': een simpel lopend gemiddelde van cijfers (bv. een 7-daags gemiddelde) is **NIET** wat hier gebeurt!

### d — Integration (differencing)

- De vorige methodes werken enkel wanneer de tijdsreeks **stationair** blijft door de tijd — in de praktijk is dat zelden het geval.
- Oplossing: kijk naar de **verschillen tussen elke tijdsstap** en maak daar een nieuwe tijdsreeks mee:

$$ \Delta y_t = y_t - y_{t-1} $$

- Dit proces herhalen we (parameter $d$) tot de nieuwe reeks **stabiel** is.
- Het berekenen van $\Delta$ is de **numerieke versie van een afgeleide** nemen: een veelterm is na een aantal keer afleiden een constante; uit die filosofie herhalen we dit proces op tijdsreeksen tot die stabiel (ongeveer constant) zijn.

| | kenmerken |
|---|---|
| **Hoge $d$** | reeksen met een kwadratisch verband zijn na $d=2$ constant; voor reeksen die heel abrupt gedrag vertonen (hard slingeren) |
| **Lage $d$** | lineaire reeksen zijn na $d=1$ al stationair; de reeks wordt redelijk snel stationair |

- Als een parameter op 0 wordt gezet, doet dat element **niet mee** in het model — bv. een zuiver 'moving average'-trendlijn: $p=0$ en $d=0$.

## Hoe parameters bepalen?

- We moeten goede waarden vinden voor $p$, $d$ en $q$.

**Protocol:**
1. **ADF-test**: is differencing nodig?
2. **Grafieken bekijken**: differencing + ACF
3. **PACF-plot**: bepaal $p$
4. **ACF-plot**: bepaal $q$
5. → **ARIMA(p, d, q)**

### Stap 1: stationariteit — Adfuller-test

- Eerst met de **Adfuller-test** bepalen of je effectief differencing moet doen. **Null-hypothese** van de ADF-test: de time series is **niet-stationair**.

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(df.value.dropna())
print('ADF statistiek: %f' % result[0])
print('p-waarde: %f' % result[1])  # als p > 0.05 -> differencing toepassen
```

- Als $p > 0.05$: de reeks is vermoedelijk **niet-stationair** → differencing toepassen.

### Stap 2: d bepalen via grafieken

- Doe wat differencing en kijk hoe de plots en de autocorrelatie er intussen uitzien (de ACF wil je dicht bij 0):

```python
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Originele data
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.value); axes[0, 0].set_title('Originele data')
plot_acf(df.value, ax=axes[0, 1])

# 1st Orde Differencing
axes[1, 0].plot(df.value.diff()); axes[1, 0].set_title('1st Orde Differencing')
plot_acf(df.value.diff().dropna(), ax=axes[1, 1])

# 2nd Orde Differencing
axes[2, 0].plot(df.value.diff().diff()); axes[2, 0].set_title('2nd Orde Differencing')
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])
plt.show()
```

- **Interpreteer de plots**: gedraagt de data zich na 1 keer differentiëren al als een **stationaire reeks**, dan is verder differentiëren **niet nodig** → $d = 1$.
- Voorbeeld: synthetische random tijdsreeks (`np.random.seed(41)`, `np.cumsum(np.random.randn(100))`; zie notebook).

> 📌 **Conda-omgeving**: het installeren van sommige packages kan problemen geven — maak hiervoor best een nieuwe conda-omgeving (`conda config --add channels conda-forge` / `conda config --set channel_priority strict` / `conda install pmdarima`).

### Stap 3: p bepalen — Partial Autocorrelation (PACF)

- PACF geeft de correlatie aan tussen een **lag** en heel de reeks; op basis van de PACF beslis je **hoeveel lags** er nodig zijn in de AR.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0, 5))
plot_pacf(df.value.diff().dropna(), ax=axes[1])
plt.show()
```

- **Hoe lees je dit af?** Houd de lags bij die **buiten de lichtblauwe zone** komen. Komt er na de **eerste** bijna niets meer erboven, dan is **$p = 1$** een goede keuze.

### Stap 4: q bepalen — Autocorrelation (ACF)

- Net zoals we naar de PACF-plot kijken voor de AR-termen, kijken we naar de **ACF-plot** voor het aantal MA-termen **$q$**. De ACF toont hoeveel MA-termen er nodig zijn om eventueel resterende autocorrelatie **weg te werken** zodat er een stationaire reeks overblijft.

```python
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0, 1.2))
plot_acf(df.value.diff().dropna(), ax=axes[1])
plt.show()
```

- Ook hier kan je dus beginnen met **$q = 1$**.

## ARIMA-model maken en fitten

```python
from statsmodels.tsa.arima.model import ARIMA

# 1,1,1 ARIMA Model
model = ARIMA(df.value, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())
```

- **AIC** (Akaike Information Criterion): een getal dat voor een goed model **zo laag mogelijk** is. Twijfel je tussen verschillende versies → **check de AIC** en kies het model met de laagste waarde.
- In de summary vind je ook de uiteindelijke **coëfficiënten** van het model en de **p-waarde** van die coëfficiënt.
- **Residual errors**: kijk tot slot naar de uiteindelijke fouten — zijn deze normaal verdeeld (KDE = Kernel Density Estimation plot)?

```python
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
```

## Voorspellingen maken

```python
forecast_steps = 25
forecast = model_fit.forecast(steps=forecast_steps)

plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(df)), df['value'], label='Gegeven data')
plt.plot(np.arange(len(df), len(df) + forecast_steps), forecast, label='Forecast', color='red')
plt.title('ARIMA Forecast')
plt.xlabel('Tijd')
plt.ylabel('Waarde')
plt.legend()
plt.show()
```

- In deze forecast zitten nu nieuwe tijdstappen die de trend van de data volgen; je kunt nu **variëren** met de parameters van `order`.

## Voorbeeld: shampoo-dataset

- Realistische dataset: maandelijkse shampooverkoop, ingelezen met datum als **index**:

```python
def parser(x):
    return datetime.strptime('201'+x, '%Y-%m')

series = pd.read_csv('shampoo.csv', header=0, parse_dates=[0],
                     index_col=0, date_parser=parser)
series.rename(columns={'Sales':'value'}, inplace=True)
series.plot(); pyplot.show()
```

- Door goed naar de autocorrelatiegrafiek te kijken: op een lag van **12 maanden** is er geen correlatie; de eerste **5 lags** zijn significant (boven het stippellijntje) → AR-parameter 5:

```python
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit()
print(model_fit.summary())
```

- De density plot van de residuals lijkt een Gaussische curve, maar is niet gecentreerd rond 0 maar rond **21.93**.

### Walk-forward validatie

- Splits netjes in **train en test**, en doe een *walk-forward* evaluatie: voorspel elke teststap met het model op de geschiedenis, voeg daarna de echte waarde toe aan de geschiedenis en herhaal.

```python
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]

for t in range(len(test)):
    model = ARIMA(history, order=(1,2,2))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(test[t])   # echte waarde toevoegen en herhalen

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
```

### Finetuning

- **Box-Jenkins methode**:
  1. Gebruik **plots** om trends, seasonality en autoregressie-elementen terug te vinden: bepaal zo de lag en de mate van differencing.
  2. Fit je model met deze **zinvolle parameters**.
  3. Check: zit er op de plots van de residual error nog **structuur**? Dan herhaal je het proces.
- **Grid search**: probeer systematisch combinaties van $(p, d, q)$; voor elke combinatie fit je het model, voorspel je via walk-forward en bereken je de RMSE. Kies uiteindelijk de combinatie met de **laagste RMSE**:

```python
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
```

## SARIMA (seizoenspatronen)

- Zijn er seizoenspatronen, dan is de data niet helemaal stationair constant → de basisvoorwaarde van ARIMA is niet vervuld, maar er is een model dat een oplossing biedt: **Seasonal ARIMA**, kortweg **SARIMA**.
- Implementatie: `statsmodels.tsa.statespace.sarimax.SARIMAX`.
- Hier gaan we niet verder op in, maar het kan handig zijn voor **projecten later**.

## Kalman Filters

- Tweede techniek voor tijdsreeksen. Voorbeeldprobleem: een app die de **GPS-locatie van een vliegtuig** aangeeft en probeert te **voorspellen** waar het vliegtuig naartoe gaat.
- **Het probleem met sensors**: GPS bevat een **foutmarge**; daardoor ziet het ruwe traject van de data er "bumpy" uit en zit er af en toe een **afwijkende waarde** tussen → boordcomputer slaat op hol?
- **Werking (predict + update)**:
  - Begint met een **initiële waarde**.
  - **Predict-fase**: geeft een nieuwe **staat** en een **(on)zekerheid** over die staat.
  - **Update-fase** (wanneer een echte meting binnenkomt): kijkt of de staat "dicht genoeg" bij de meting ligt.
- Dit wordt in real life ook echt gebruikt in **boordnavigatie van vliegtuigen**: de filter vermijdt dat een boordcomputer op hol slaat als 1 sensor een afwijkende waarde meet.

> 📌 Zie ook de oefening: `kalman_oefening_oplossing.ipynb`

## Beste praktijken (Week 4)

- Evalueer regressie met MAE/MSE/R² en probeer dezelfde algoritmes (KNN, SVM, RF) eerst als *Regressor*-variant voor je iets complexers pakt.
- ARIMA-protocol: eerst **ADF-test** → differencing ($d$), dan **PACF** → $p$, dan **ACF** → $q$; vergelijk modellen via de **AIC** (lager = beter) en evalueer forecasting met **walk-forward + RMSE**.
- Verwar "moving average" in ARIMA niet met een simpel rollend gemiddelde.
- Seizoenspatronen in de data? Gebruik **SARIMA** in plaats van ARIMA.
- Ruisende sensordata? Een **Kalman filter** strijkt glad en voorspelt tegelijk (predict + update).

## Kernpunten

- Klassieke classifiers hebben ook een **regressor-variant**; evaluatie via MAE/MSE/R².
- ARIMA = AR($p$) + I($d$) + MA($q$); parameters via **ADF-test**, **PACF** en **ACF**.
- Walk-forward validatie + RMSE voor het correct evalueren van forecasting-modellen.
- **SARIMA** voor seizoensdata; **Kalman filter** voor ruisende metingen (predict + update).


