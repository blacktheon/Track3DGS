# Track02 t001: first regional training pass

Status: **six regions trained and exported; draft assembly, manual visual review required**. The 420 m boundary has additional near-ground coverage loss after trimming. The dataset is not a gap-free or production-quality acceptance result.

## Location and recipe

The local dataset is `data/routes/track02/r002/training/t001/`. Its `RESULTS.md` links all original, cleaned and trimmed PLYs and review images. Large training data is excluded from Git. The portable reconstruction package is `data/routes/track02/r002/packages/training-t001/`.

- Track02: 210.033 s, 7680×3840 video; 421 registered panoramas at 2 Hz, eight 1600×1600 crops per panorama.
- One reviewed global route; all crops of held-out panoramas excluded from pixel training.
- 30,000 Splatfacto iterations per region; one GPU job at a time.
- Native SH3 retained with identity local-to-package transforms; full originals preserved.
- Approximate scale from an assumed 10 km/h, not a metric survey. Usable core covers nominal station 20–563.33 m; the capture ends supply training context.

| Chunk | Core (nominal m) | Original | Cleaned | Trimmed |
|---|---:|---:|---:|---:|
| 1 / `cell_000` | 20.0–120.0 | 597,638 | 584,389 | 378,428 |
| 2 / `cell_001` | 120.0–220.0 | 612,278 | 599,189 | 437,360 |
| 3 / `cell_002` | 220.0–320.0 | 608,875 | 590,060 | 364,301 |
| 4 / `cell_003` | 320.0–420.0 | 620,153 | 608,436 | 396,691 |
| 5 / `cell_004` | 420.0–520.0 | 664,147 | 651,828 | 499,960 |
| 6 / `cell_005` | 520.0–563.3 | 818,153 | 796,425 | 394,073 |
| **Total** | | **3,921,244** | **3,830,327** | **2,470,813** |

## Evidence and limits

Continuous nearest-route ownership assigns nonoverlapping centre domains; the final endpoint is inclusive. No ambiguous core rows were reported. All five neighboring pairs were displayed in QuestSBTC with no manual positioning. Gaussian footprints extend beyond their centres, so ownership is not proof of seamless appearance.

Each region has held-out original/clean photographic reports, and every boundary has individual-context versus assembled-core reports. A one-run diagnostic (`training/t001/reports/run_lateral_diagnostics.py`) additionally rendered 20 synthetic poses, translated ±1.5 nominal metres at the recorded camera height. Eleven composites were inspected, including the largest additional low-alpha case at every boundary. These synthetic views have no photographic ground truth.

Road layout remains aligned in the inspected views, but near-road texture is elongated, vegetation is soft, and canopy floaters remain. Near-ground holes occur in the source models too. At the 420 m join, trimming enlarges the weakness: the largest diagnostic view contains 4,714 pixels with assembled alpha below 0.5 where both parent references exceed 0.9. Human review was requested; no numeric threshold grants acceptance. The final region has the lowest average held-out photographic score of the six.

## Unity and package

QuestSBTC uses the pinned `wu.yize.gsplat` 1.4.0 renderer and `Assets/Scenes/Track02_TrainingPreview.unity`. Open **Tools → Track3DGS → Training Model Preview**. Only two Gaussian slots exist; switching unloads both. Spark pairs use global sorting. Native PLY masters remain unchanged; translation-centred Unity caches are reproducible. The saved preview shows Chunks 4 and 5 near the flagged join.

The approximately 840 MB reconstruction package validates with **6 assets, 2,470,813 splats, state draft**. It includes exact source-row identities, route/camera metadata and all photographic/synthetic evidence. Validation checks hashes, attributes, covariance bounds and provenance. No absolute machine paths were found in package JSON. A two-region pilot package also passed validation after relocation.

Implementation verification: 113 Python CPU tests passed; the editor pair-sorting image regression matched a merged reference in forward and reverse views (mean error 0.000000); completed-model console checks reported no errors. The first editor attempt encountered a D3D12 device-hung crash before real model loading; its log is retained in the run reports. The editor was restarted and subsequent pair inspections completed. These checks do not establish standalone Quest performance.

The reviewed route SHA-256 remains `c5f9128e7d0de1124873110d8109d16dd106f5a9ebedb2474e050709c041273d`. The package producer records code commit `7f19166861a9655c9adc64093b8540fa76ee55b3`.

## Remaining work

Resolve visual acceptance and the 420 m ground issue before treating the route as a final surface. A changed ownership boundary or shared repair region with denser registered views can be evaluated against this preserved baseline. Stage1 visibility reduction, generic VR3DGS package ingestion, LOD generation and standalone Quest pressure tests are separate milestones. See the [Step 2 workflow](route-step2.md) for reproduction commands.
