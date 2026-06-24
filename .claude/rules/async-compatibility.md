# Async Compatibility Rules

The SDK ships **two parallel client stacks** that must stay in lockstep. Any
change to one stack has to be reflected in the other, or the async API silently
drifts (or deadlocks).

| Concern | Sync | Async |
|---------|------|-------|
| HTTP client | `weaver/_http.py` · `APIClient` | `weaver/_async_http.py` · `AsyncAPIClient` |
| Operation handle | `OperationHandle` | `AsyncOperationHandle` (awaitable) |
| Service client | `service_client.py` · `ServiceClient` | `async_service_client.py` · `AsyncServiceClient` |
| Training client | `training_client.py` · `TrainingClient` | `async_training_client.py` · `AsyncTrainingClient` |
| Sampling client | `sampling_client.py` · `SamplingClient` | `async_sampling_client.py` · `AsyncSamplingClient` |

## 1. Single Source of Truth — no logic forks

Request payloads and response parsing are **pure** and MUST live in the shared
modules so both stacks build identical bytes:

- `weaver/_payloads.py` — payload builders, metadata validation, surrogate-grad helpers
- `weaver/_sampling_utils.py` — sample/logprobs bodies, result normalization

**Never inline payload/parse logic into a single stack.** If you add a request
field or change normalization, edit the shared helper and both stacks inherit it.

Pure status logic for handles lives in `operations._OperationHandleMixin`
(shared by both handles). HTTP helpers shared via `_http.py`:
`extract_model_id_from_path`, `raise_for_response`, `apply_request_span_attributes`,
`compute_retry_delay`, `_is_connection_error`.

## 2. When you add or change a client method

Do it in **both** the sync and async client, with aligned signatures:

- Same name, same args, same keyword-only `wait` flag.
- Sync returns `OperationHandle` (or result); async returns `AsyncOperationHandle`
  (or awaited result). The async method is `async def`; submit is `await`ed.
- Sync blocks via `handle.result()`; async via `return await handle.result() if wait else handle`.
- Export any new public class in `weaver/__init__.py.__all__`.

A PR that touches `ServiceClient`/`TrainingClient`/`SamplingClient` without the
matching `Async*` change (or vice versa) is incomplete.

## 3. Never block the event loop in async paths

The whole point of the async stack is yielding the loop. In any `async def` /
async module:

- **No `time.sleep`** — use `await asyncio.sleep`. (Polling backoff included.)
- **No sync `httpx.Client`** — use `httpx.AsyncClient` / awaited requests.
- **No blocking IO or heavy sync CPU** without a comment justifying it. Tokenizer
  loads are tolerated as one-time lazy work; resolve sources (e.g. base_model)
  with `await` *before* calling into the sync normalization helpers.
- Mirror any change to `APIClient` retry/trace/fork-safety logic in
  `AsyncAPIClient` (the loop bodies are intentionally parallel).

## 4. Tests

- Add async coverage alongside sync coverage. Mocked unit tests use
  `unittest.mock.AsyncMock` and drive coroutines with `asyncio.run` (no
  `pytest-asyncio` dependency).
- Liveness / no-deadlock tests (`tests/test_async_no_deadlock.py`) MUST:
  - hit a **real** local socket server under concurrency,
  - carry a hard `@pytest.mark.timeout(...)` **and** an inner `asyncio.wait_for`
    so a deadlock or accidental blocking call **fails** instead of hanging,
  - include a "ticker" probe asserting the loop stays responsive while awaiting.

## Quick Checklist (in addition to core-development.md)

- [ ] Shared logic changed in `_payloads.py` / `_sampling_utils.py`, not forked
- [ ] Sync **and** async client both updated with aligned signatures
- [ ] New public classes exported in `__init__.py`
- [ ] No `time.sleep` / sync `httpx` / blocking IO in async code paths
- [ ] `APIClient` ↔ `AsyncAPIClient` retry/trace/fork logic kept in sync
- [ ] Async tests added; liveness tests have timeout + ticker probe
