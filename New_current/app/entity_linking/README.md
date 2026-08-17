# Entity Linking

Phase 6 links normalized production NER spans to canonical UMLS concepts through:

`route -> EntityLinkingService -> BaseEntityLinkingProvider -> SciSpacyUMLSProvider`

The adapter is local-only and fail-closed. Exact SciSpaCy, language-model, UMLS release,
artifact paths, and license information are governed by the Entity Linking block in the
repository `MODEL_MANIFEST.md` or its listed environment variables. Set
`ENTITY_LINKING_CONFIG__UMLS_LICENSE_ACCEPTED=true` only after the deployment is licensed.
The configured language-model and UMLS directories must already exist and be non-empty;
application startup is not an artifact acquisition mechanism.

Until the manifest is approved and artifacts are provisioned, the rest of the application
starts but `GET /api/v1/entity-linking/health` returns HTTP 503 with `NOT_READY`. No
heuristic or alternate terminology fallback is used.

Public operations are `POST /api/v1/entity-linking`,
`GET /api/v1/entity-linking/health`, and `GET /api/v1/entity-linking/models`. The internal
engineering console is `/entity-linking`. Inputs and outputs may contain protected health
information and must follow platform authorization, transport, logging, and retention
controls.
