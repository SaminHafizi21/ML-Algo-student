# Week 9 — Genetische algoritmes

> Geen notebooks bij deze les — enkel slides.

## Introductie — Survival of the fittest

- Genetische algoritmes zijn algoritmes die, zoals in de biologie, langzaam kunnen veranderen; er zit dus een **kanscomponent** aan vast.
- Ze zijn krachtiger en efficiënter dan pure random zoekalgoritmes of brute-force zoekalgoritmes (zie AI Programming).
- Worden vaak gebruikt voor **optimalisatieproblemen**.

## Algemene werking

Het algoritme wordt elke stap aangepast:
1. Genereer een set van random items
2. Kies daaruit de beste
3. **Kruis** deze met elkaar → 'children'/'offspring' uit 'parents'
4. **Muteer** de nieuwe generatie offsprings
5. Herhaal tot aan termination

- Een oplossing voor het optimalisatieprobleem is dus een 'individu' in deze populatie.

## Initiële populatie

- Wordt willekeurig gekozen:
  - volledig willekeurig, of
  - — als er al kennis over het probleem is — volgens een heuristiek of 'educated guess'.

## Fitness van een individu

- Om in te schatten hoe goed een oplossing is, heb je een **fitness function** nodig; op basis daarvan selecteer je de beste individuen om verder mee te 'kweken'.
- Idealiter geeft deze functie aan hoe dicht je bij de ideale oplossing zit.
- Afweging: een **snel berekenbare** functie vs de **accuraatheid** van die functie — omdat we de exacte oplossing per definitie niet hebben.

## Selectie

- Twee aspecten afwegen: enerzijds de **beste individuen**, anderzijds **voldoende variatie** — anders lijken volgende generaties te veel op mekaar → niet robuust.
- Strategieën:
  - **Fitness proportionate**: kans op selectie is evenredig met fitness score
  - **Tournament selection**: in bracketed matchups random individuen vergelijken
  - **Rankings**: hoogste fitness-scores worden geselecteerd

## Crossover — volgende generatie

- Twee methodes:
  - **Point crossover** (zie prent)
  - **Gene-based crossover**: elke parent levert 50% van de oplossing

## Mutatie — volgende generatie

- Lichte wijziging van de nieuwe generatie → creëert **variatie**.
- De kans op mutatie kan vastgelegd worden in een parameter; als die hoog is, ga je eerder naar een randomized algoritme zoals simulated annealing.

## Termination

- Het algoritme eindigt wanneer een individu zó fit is (zódán dicht bij de optimale oplossing) dat we stoppen.
- Of: stop na een vast aantal generaties, ongeacht de fitheid.

## Beste praktijken (Week 9)

- Balanceer **exploratie (mutatie, variatie)** en **exploitatie (beste individuen selecteren)** — te weinig variatie maakt de populatie niet robuust.
- Kies een fitness function die snel te berekenen is maar toch accuraat genoeg.
- Een hoge mutatiekans maakt het algoritme randomized — gebruik deze parameter bewust.
- Bepaal op voorhand je termination-criterium (fitheid of maximaal aantal generaties).

## Kernpunten

- Genetische algoritmes: populatie → selectie → crossover → mutatie → herhaal.
- Fitness function stuurt de selectie; selection strategieën: fitness proportionate, tournament, rankings.
- Terminatie op fitheid of na vast aantal generaties.