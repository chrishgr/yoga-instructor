# coachNasarus — prosjektbeskrivelse

## Hva dette er

coachNasarus er en sanntids yoga-coach som bruker datasyn til å gjenkjenne yoga-positurer, måle hvor lenge du holder dem, estimere hvor mye du avviker fra en referansepositur, og gi lydfeedback som hjelper deg med å justere kroppen inn i riktig stilling uten å se på skjermen.

Appen er designet for å kjøre lokalt på helt vanlig maskinvare. En fem år gammel laptop uten dedikert GPU er mer enn nok. Tunge operasjoner som trening av klassifikatoren eller batch-prosessering av referansedata er bevisst flyttet ut av den interaktive pipelinen, slik at de kan kjøres én gang på et HPC-cluster (Fox) og resultatene lastes av lokalmaskinen.

## Motivasjon

Yoga-positurer er statiske eller saktebevegelige, noe som gjør dem spesielt godt egnet for sanntids pose-estimering på CPU. Et realistisk alternativ hadde vært å bruke et cloud-API, men verken MediaPipe, MoveNet eller tyngre modeller som ViTPose tilbys som pay-per-call API-er slik LLM-er gjør. Det er derfor naturlig å kjøre alt lokalt, med Fox som treningsbackend når det er nødvendig.

## Arkitektur

Pipelinen består av fem moduler som kommuniserer via enkle, rene dataklasser:

1. **`PoseClassifier`** — tar en frame, bruker en backend (MediaPipe i første omgang) til å hente ut 33 landmarks, og returnerer et positurnavn. Første versjon bruker regler basert på ledvinkler; senere kan `classify()` byttes ut med en kNN eller MLP uten at resten av pipelinen merker noe.

2. **`PoseTracker`** — konverterer en strøm av (potensielt støyende) per-frame-prediksjoner til stabile positur-intervaller med varighet. Bruker majoritetsvotering over et glidende vindu for å absorbere isolerte feilklassifiseringer. Intervaller kortere enn `min_hold_seconds` regnes som overganger og loggføres ikke.

3. **`DeviationEstimator`** — sammenligner brukerens ledvinkler med en referansemal og returnerer både et skalart gjennomsnittsavvik og en per-ledd-oppdeling. Vinkelbasert metrikk er valgt fordi den er invariant under translasjon, skala og speiling, og fordi resultatet kan mappes tilbake til noe brukeren forstår ("høyre kne er for bøyd"). Procrustes-distanse er reservert for senere.

4. **`AudioFeedback`** — mapper avviket til en kontinuerlig tone på en bakgrunnstråd, slik at hovedløkken aldri blokkeres av lyd. Mappingen er bevisst enkel: lavere avvik gir en høyere, renere tone. Kan senere utvides med flere stemmer eller taktile signaler.

5. **`main.py`** — hovedløkken som binder alt sammen. Leser frames fra en `VideoSource` (webkamera, videofil, eller bildemappe), kjører hver frame gjennom de fire modulene over, tegner overlay, og skriver ut en øktoppsummering ved avslutning.

Backend-laget er abstrahert bak `PoseBackend`-grensesnittet i `src/backends/`, slik at man kan bytte MediaPipe ut med MoveNet, Sapiens eller noe annet uten å røre resten av koden.

## Konfigurasjoner

Tre YAML-profiler dekker de viktigste bruksscenariene:

| Profil | Maskin | Formål |
|---|---|---|
| `local.yaml` | Vanlig laptop, CPU | Daglig yoga. `model_complexity=1`, regelbasert klassifikator. |
| `local_hq.yaml` | Kraftig desktop | Eksperimentering med `model_complexity=2` og kNN-klassifikator. |
| `fox_train.yaml` | Fox HPC (SLURM) | Trening og batch-bygging av maler. Ingen sanntidsinferens. |

Prinsippet er **tren tungt, kjør lett**: de eneste jobbene som trenger Fox er trening av klassifikatoren og batch-ekstraksjon av referansemaler fra store videodatasett. Begge er engangsjobber som produserer filer (`models/knn_classifier.pkl`, `templates/*.json`) som lokalmaskinen så laster.

## Inputkilder

Én av de viktigste designavgjørelsene er at `main.py` er agnostisk til hvor frames kommer fra. `VideoSource`-klassen støtter tre kilder bak ett felles grensesnitt:

- **Webkamera** (default) — live yoga-økt.
- **Videofil** — `--video data/sample_videos/x.mp4` lar deg teste appen mot opptak du allerede har. Uvurderlig under utvikling fordi du slipper å gjøre yoga foran kameraet hver gang du vil teste en endring.
- **Bildemappe** — batch-prosessering for templatebygging eller debugging.

Dette betyr at utviklingssyklusen blir: ta opp en kort video av deg selv i noen positurer, legg den i `data/sample_videos/`, og kjør `python main.py --video data/sample_videos/din_video.mp4`. Du får samme overlay og samme øktoppsummering som ved live kjøring.

## Implementeringsfaser

**Fase 1 — Fundament og pose-deteksjon.** MediaPipe Pose via `MediaPipeBackend`, regelbasert klassifikator for tadasana, vrikshasana og adho mukha svanasana. Minimal `main.py` som viser positurnavn på kameraframen.

**Fase 2 — Tidslogging.** `PoseTracker` med glidende vindu og øktshistorikk. Session summary skrives til JSON ved avslutning.

**Fase 3 — Avviksestimering.** `DeviationEstimator` med vinkelbasert metrikk. Templatebygging via `scripts/build_templates.py`. Per-ledd-feedback i overlay.

**Fase 4 — Lydfeedback.** `AudioFeedback` på separat tråd. Enkel avvik→frekvens-mapping.

**Fase 5 — Integrasjon.** Alle moduler kombinert i `main.py`. Session summary og valgfri JSON-logg.

**Senere.** Trent kNN/MLP-klassifikator via `scripts/train_classifier.py` kjørt på Fox. Flere positurer. Eventuelt bedre backends. Kanskje Procrustes som alternativ avvik-metrikk.

## Datastrømmer

```
 ┌─────────────┐      ┌────────────────┐      ┌─────────────────┐
 │ VideoSource │ ───▶ │ PoseClassifier │ ───▶ │   PoseTracker   │
 └─────────────┘      └────────┬───────┘      └────────┬────────┘
                               │                       │
                               ▼                       ▼
                      ┌────────────────┐      ┌─────────────────┐
                      │ DeviationEst.  │ ───▶ │  AudioFeedback  │
                      └────────────────┘      └─────────────────┘
                               │
                               ▼
                        overlay + log
```

Per frame: les frame → ekstraher landmarks → klassifiser → oppdater tracker → beregn avvik → oppdater lyd → tegn overlay → vis. Alt er sekvensielt i hovedtråden bortsett fra lyden, som kjører på en daemon-tråd og kun leser en enkelt delt verdi under en kort lås.

## Avhengigheter

Kjerneavhengighetene er `mediapipe`, `opencv-python`, `numpy`, `scipy`, `pygame` og `pyyaml`. Trening legger til `scikit-learn` (ikke listet i `requirements.txt` siden den kun brukes i offline-scriptet). Ingen PyTorch, ingen CUDA, ingen nettverkstrafikk etter første installasjon.
