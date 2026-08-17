# Engineering Demonstration Dashboard Report

Date: 2026-08-05  
Dashboard: `/engineering-demo`  
Disposition: **READY FOR INTERNAL PROJECT DEMONSTRATIONS**

## Purpose and boundary

The dashboard is a single same-origin Jinja2 page for internal engineering demonstrations.
It is not the customer frontend and contains no independent clinical inference or storage
logic. Bootstrap 5, focused CSS, and vanilla JavaScript render the page; no React build or
separate frontend server exists.

The page reuses existing REST operations:

- OCR upload, health, and model inventory;
- Medical NER inference, health, and model inventory;
- Entity Linking health;
- Relation Extraction health;
- Medical Embeddings inference, health, and model inventory.

The available browser flow is upload → OCR → Medical NER → Medical Embeddings. Embeddings
are requested only when their health operation reports ready. Entity Linking and Relation
Extraction remain health-visible but runtime-pending. No unavailable stage receives fake
output.

## Delivery-state presentation

| Module | Dashboard state |
|---|---|
| OCR | Health-driven; Production Ready only when the live provider is ready |
| Medical NER | Health-driven; Production Ready only when the live provider is healthy |
| Entity Linking | Architecture Complete / Runtime Pending |
| Relation Extraction | Architecture Complete / Runtime Pending |
| Medical Embeddings | Health-driven; Architecture Complete while runtime is pending |
| PostgreSQL | Planned |
| Redis | Planned |
| Celery | Planned |
| Qwen3 Simplification | Coming Next |
| Verification | Coming Next |
| Translation | Coming Next |
| TTS | Frozen until Qwen Simplification is complete |

Infrastructure and future-stage cards are sourced from the frozen roadmap state rather
than an API because those modules intentionally have no REST endpoints. Operational cards
consume existing endpoints directly.

## Verification evidence

| Gate | Result | Evidence |
|---|---|---|
| Ruff | PASS | `python -m ruff check app tests` |
| Full test suite | PASS | 72 passed, 1 skipped |
| Import | PASS | `app.main` imported successfully |
| Compilation | PASS | Application and tests byte-compiled |
| Swagger/OpenAPI | PASS | Schema generated with 21 public paths; `/docs` returned 200 |
| Dashboard | PASS | `/engineering-demo`, JS, and CSS returned 200 in lifecycle test |
| Required sections | PASS | All eleven sections asserted in the rendered HTML |
| Existing API reuse | PASS | Every JavaScript runtime endpoint exists in OpenAPI |
| JavaScript syntax | PASS | Bundled Node.js `--check` |
| No React | PASS | No React asset or dashboard reference |
| No database dependency | PASS | Page performs no database operation |
| No Qdrant | PASS | No dashboard Qdrant route or client behavior |

## Known limitations

- Real stage output depends on each provider's current health and approved local model
  availability. Pending stages remain visibly pending.
- Bootstrap is loaded from the same CDN convention used by the existing engineering
  consoles; a disconnected demonstration environment must provide network access to that
  asset or vendor Bootstrap separately under an approved dependency change.
- The page intentionally has no authentication because it is an internal demonstration
  surface. It must be protected by deployment/network controls and used only with
  synthetic or approved de-identified reports.
