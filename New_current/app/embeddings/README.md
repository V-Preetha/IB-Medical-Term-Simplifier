# Medical Embeddings

The Phase 8 Stage 1 boundary is:

`route -> MedicalEmbeddingService -> BaseEmbeddingProvider -> BioClinicalModernBERTProvider`

`POST /api/v1/embeddings` accepts a bounded batch of text records and returns vectors
directly. It does not persist them and does not import or call Qdrant. Text is never written
to normal logs or echoed in the response; callers correlate vectors through `input_id`.

The adapter loads only an approved local Transformers checkpoint, pools contextual token
states using `attention-mask-mean-v1`, and optionally L2-normalizes vectors. Exact model
identity, revision, license, cache path, device, batching, max length, and normalization
are environment-backed and exposed safely as reproducibility metadata.

The current manifest intentionally leaves repository ID, revision, license, and cache as
`PENDING_APPROVAL`. Until approved deployment values and matching local revision evidence
exist, `GET /api/v1/embeddings/health` returns HTTP 503 with `NOT_READY`. No experimental
or archived model is substituted.
