# CLAUDE.md

Voice companion server: LLM (streamed) -> sentence splitter -> CSM (Sesame) TTS -> WebSocket audio chunks to browser client. Single GPU, single model instance.

## Architecture

- `main.py` - FastAPI + WebSocket server (`/ws`). Orchestrates the whole turn.
- `generator.py` - `Generator.generate_stream()`: autoregressive frame-by-frame audio generation with a Mimi-style streaming decoder, yields decoded audio chunks.
- `llm_interface.py` - blocking `requests.iter_lines()` wrapper around the LLM's streaming HTTP API (`generate_response_stream`).
- `models.py` - backbone + decoder transformer defs (loaded/compiled in `model_worker`).
- `vad.py` - mic-side VAD for barge-in/turn-end detection (`AudioStreamProcessor`), unrelated to inter-sentence TTS latency.
- `config.py` - JSON persistence for `CompanionConfig` (UI-editable settings only; no latency tunables live here).
- `rag_system.py` - retrieval augmentation injected into the system prompt before LLM generation.

## Turn flow (`main.py:speak_streaming`, ~line 516)

1. LLM streams text deltas (`llm.generate_response_stream`, held under `llm_lock`).
2. `extract_complete_sentences` peels off finished sentences from the running buffer as deltas arrive.
3. Each sentence (optionally split further by `_split_long_sentence` for long clauses) is handed to `_generate_sentence_audio` (`main.py:397`), which:
   - computes a `max_audio_length_ms` cap from word count (`estimated_seconds * 1.6`, floor 1500ms),
   - pushes `(text, speaker, context, cap, temperature, topk)` onto `model_queue`,
   - **blocks** in a loop draining `model_result_queue` until the `None` EOS marker for *that sentence* arrives, forwarding each chunk to the client over the websocket as it's produced,
   - appends the finished sentence's audio to `turn_context` (so the next sentence's generation stays prosody-consistent with it).
4. The outer `for delta in ...` loop only resumes pulling more LLM tokens once step 3 fully returns.
5. `model_worker` (`main.py:724`) is the single consumer of `model_queue`; it owns the one loaded `Generator`/model instance and processes requests strictly one at a time to completion (`for chunk in generator.generate_stream(...): model_result_queue.put(chunk)`).

**Net effect: sentences are generated strictly sequentially — sentence N+1 is not even requested until sentence N's entire generation (all frames, not just first chunk) has drained.** There is no cross-sentence prefetch/pipelining, and only one GPU generation can be in flight at a time regardless (single model instance, single worker thread).

## Why inter-sentence gaps are audible

`generate_stream` (`generator.py:161`) buffers `buffer_size` frames before decoding and yielding the *first* chunk of any sentence. Historically this was meant to be tiered — a smaller `initial_*` for fast first-chunk, then a larger `normal_*` for throughput once the pump is primed (see the `first_chunk_delivered` toggle at `generator.py:333-338`) — but in the current code:

```python
initial_batch_size = 20
normal_batch_size = 20
initial_buffer_size = 20
normal_buffer_size = 20
```

`initial_*` == `normal_*`, so the tiering is a no-op: **every sentence**, not just the very first one of a turn, must generate a full 20-frame (1.6s-of-audio-worth) buffer before its first chunk is decoded and sent. This is what the `TTS time-to-first-chunk` log line measures (typically ~900-1400ms in practice), and it recurs identically per sentence.

Because sentence N+1 isn't dispatched to `model_queue` until sentence N is fully drained (not just until N starts playing on the client), and because generation frequently runs close to real-time (RTF often 0.5-1.0x, occasionally >1.0x under load — see `generator.py:386`), N+1's ~1-1.4s priming latency is often *not* fully hidden behind whatever's left of N's playback tail on the client. That uncovered remainder is the audible dead-air gap between sentences.

Truncation is a related but separate failure mode: `max_audio_length_ms` cap is derived from an estimated words-per-minute duration (`main.py:419-420`) that is frequently off by 2-3x sentence-to-sentence (confirmed in logs — 6.64s actual vs 13.41s estimated on one sentence, 22.56s actual vs 14.12s estimated, i.e. cap hit, on the very next one). A tight/wrong cap can truncate a natural sentence before its own EOS frame.

## Key tunables for latency work

- `generator.py:187-190` - `initial_batch_size` / `initial_buffer_size` (frames before first chunk of a sentence). Lowering these directly cuts per-sentence time-to-first-chunk.
- `generator.py:308` - buffer flush threshold (`len(frame_buffer) >= buffer_size`).
- `main.py:419-420` - `max_audio_length_ms` cap formula (estimated_seconds * 1.6, floor 1500ms) — the estimator is noisy; widening the multiplier or floor reduces truncation risk at the cost of allowing longer stray generations.
- `main.py:737` - `model_queue.get(timeout=0.1)` poll interval, adds up to 100ms between one sentence's EOS and the next being picked up.
- `speak_streaming`'s inner loop (`main.py:586-604`) is where sequential dispatch happens; any pipelining fix (prefetch sentence N+1's TTS while N is still playing) goes here.

## Notes

- `audio_queue` (`main.py:108`) and the `sounddevice` import are dead code — actual playback is client-side over the websocket (`audio_chunk` messages relayed by `process_message_queue`).
- `Real-time factor` (`generator.py:386-389`) is printed via `print()`, not `logger` — won't show up in `session_data/server.log`, only stdout/console.
- `speak_streaming`'s docstring claims "LLM decoding and TTS generation overlap" — this is aspirational, not what the code currently does (see turn flow above).
