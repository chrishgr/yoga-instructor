# coachNasarus

En sanntids yoga-coach som bruker kameraet (eller forhåndsopptatte videoer) til å gjenkjenne positurer, måle hvor lenge du holder dem, estimere avvik fra en referansepositur, og gi lydfeedback som hjelper deg å justere deg inn i riktig stilling.

## Hva appen gjør

1. **Pose-deteksjon** — 33 landmarks per frame via MediaPipe Pose (CPU, sanntid på vanlig laptop).
2. **Klassifisering** — identifiserer hvilken yoga-positur du står i (regelbasert på ledvinkler til å begynne med; byttbar til kNN/MLP senere).
3. **Tidslogging** — måler hvor lenge hver positur holdes, med hysterese så enkeltfeil ikke nullstiller tidtakeren.
4. **Avviksestimering** — sammenligner dine ledvinkler med en referansemal og gir et skalart avvik + per-ledd-oppdeling.
5. **Lydfeedback** — en kontinuerlig tone som endrer seg med avviket, slik at du kan justere deg inn uten å se på skjermen.

## Kjøremodi

Appen støtter tre inputkilder, valgt via konfigurasjon eller CLI:

- **Webkamera** (default) — live yoga-økt.
- **Videofil** — last opp en `.mp4`/`.mov` fra disk for å teste eller analysere opptak i etterkant.
- **Bildemappe** — batch-prosessering av stillbilder for å bygge referansemaler.

## Konfigurasjoner

Tre config-profiler dekker ulike maskinvaremiljøer:

| Profil | Maskin | Bruk |
|---|---|---|
| `local.yaml` | Vanlig laptop, CPU | Daglig yoga-bruk. MediaPipe `model_complexity=0` eller `1`. |
| `local_hq.yaml` | Kraftig desktop/GPU | Eksperimentering med `model_complexity=2` eller alternativ backend. |
| `fox_train.yaml` | Fox HPC-cluster (SLURM) | Trening av klassifikator og batch-bygging av maler. |

Prinsippet er **tren tungt, kjør lett**: Fox brukes kun for engangsjobber som treningsdata-prosessering. Selve yoga-appen kjører lokalt.

## Rask start

```bash
# 1. Installer
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Kjør med webkamera (default config)
python main.py

# 3. Kjør mot en videofil
python main.py --video data/sample_videos/min_okt.mp4

# 4. Kjør med annen config
python main.py --config config/local_hq.yaml
```

## Prosjektstruktur

```
coachNasarus/
├── config/                     # YAML-konfigurasjoner per miljø
│   ├── local.yaml
│   ├── local_hq.yaml
│   └── fox_train.yaml
├── src/
│   ├── pose_classifier.py      # Landmark-ekstraksjon + klassifisering
│   ├── pose_tracker.py         # Tidslogging med hysterese
│   ├── deviation_estimator.py  # Vinkelbasert avvik mot referansemaler
│   ├── audio_feedback.py       # Sanntids lydgenerering
│   ├── video_source.py         # Felles grensesnitt for kamera/fil
│   ├── utils.py                # Vinkelberegning og fellesfunksjoner
│   └── backends/
│       ├── base.py             # Abstrakt pose-backend
│       ├── mediapipe_backend.py
│       └── movenet_backend.py  # valgfritt alternativ
├── scripts/
│   ├── train_classifier.py     # Trener kNN/MLP på landmark-datasett
│   ├── train_classifier.slurm  # SLURM-jobb for Fox
│   └── build_templates.py      # Ekstraherer maler fra referansebilder/-video
├── models/                     # Trente klassifikator-filer (.pkl/.pt)
├── templates/                  # Referansemaler per positur (.json)
├── data/
│   └── sample_videos/          # Testvideoer
├── tests/                      # Enhetstester
├── docs/
│   └── project_description.md  # Utvidet prosjektbeskrivelse
├── main.py                     # Hovedløkke — binder alt sammen
└── requirements.txt
```

## Faser (implementeringsplan)

- **Fase 1** — Pose-deteksjon og regelbasert klassifisering (tadasana, vrikshasana, adho mukha svanasana). Kjørbar end-to-end mot webkamera.
- **Fase 2** — `PoseTracker` med glidende vindu og øktshistorikk som JSON.
- **Fase 3** — `DeviationEstimator` med vinkelbasert metrikk + per-ledd-tilbakemelding.
- **Fase 4** — `AudioFeedback` på separat tråd, mapping avvik → tone.
- **Fase 5** — Full integrasjon i `main.py` med overlay og session summary.
- **Senere** — kNN/MLP-klassifikator trent på Fox, flere backends, flere positurer.

Se `docs/project_description.md` for den fulle implementeringsplanen.

## Systemkrav

- Python 3.10+
- Webkamera (valgfritt hvis du kun bruker videofiler)
- ~500 MB diskplass for MediaPipe-modellene

Fungerer på Linux, macOS og Windows. Ingen GPU kreves.
