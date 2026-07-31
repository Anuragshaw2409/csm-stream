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

1. A producer thread (`_llm_producer`, `main.py:572`) streams LLM text deltas (`llm.generate_response_stream`, a real OpenRouter HTTPS call, held under `llm_lock`) and pushes each complete sentence (`extract_complete_sentences`, optionally split further by `_split_long_sentence`) onto `sentence_queue`, finishing with a `None` sentinel. This runs concurrently with TTS generation so LLM network latency/jitter for sentence N+1's text is absorbed while sentence N's audio is still being generated, instead of blocking the LLM read behind a full TTS drain.
2. The main thread consumes `sentence_queue` and, for each sentence, calls `_generate_sentence_audio` (`main.py:397`), which:
   - computes a `max_audio_length_ms` cap from word count (`estimated_seconds * 1.6`, floor 1500ms),
   - pushes a request dict (`gen_id`, `text`, `speaker`, `context`, `max_ms`, `temperature`, `topk`, `cancel`) onto `model_queue`,
   - **blocks** in a loop draining `model_result_queue` until the `None` EOS marker for *that sentence* arrives, forwarding each chunk to the client over the websocket as it's produced,
   - appends the finished sentence's audio to `turn_context` (so the next sentence's generation stays prosody-consistent with it).
3. `model_worker` is the single consumer of `model_queue`; it owns the one loaded `Generator`/model instance and processes requests strictly one at a time to completion. Results go onto `model_result_queue` as `(gen_id, payload)` pairs, where payload is a chunk, `None` (EOS), or an `Exception`. The `gen_id` tag is what lets a consumer discard leftovers from a cancelled turn instead of desyncing on them.

## Barge-in / cancellation

Cancellation is **scoped per generation**, not global. Each turn gets a private `threading.Event` from `begin_turn(gen_id)`; `end_turn(gen_id)` retires it. Everything that wants to stop a turn goes through `request_interrupt(reason, gen_id=None)`, which refuses to fire if no turn is active or if `gen_id` names a turn that is no longer running.

This matters because interrupts arrive from three places with very different timing — VAD's `on_speech_start` (immediate), the client's `interrupt` websocket message (a few hundred ms later, tagged with the `gen_id` the user actually heard), and `process_user_input` when a new utterance lands mid-turn. A single global flag meant a late interrupt for turn N cancelled turn N+1 instead, so every barge-in silently killed its own reply.

Invariants worth preserving:
- `generate_stream`'s `cancel_event` is the *request's* event, carried in the model_queue dict — never a process-wide flag.
- The consumer in `_generate_sentence_audio` must always reach a terminator, and its wait is deadline-bounded: it holds `audio_gen_lock` for the whole turn, so hanging there wedges every future turn.
- `end_turn()` runs before `audio_gen_lock.release()`, so there is a window where no turn is "active" but the lock is still held. `speak_streaming` therefore acquires the lock *with a timeout* rather than `blocking=False` — dropping the input there means the user gets no answer at all.
- The mic path must not block the asyncio loop: `vad_processor.process_audio` runs in `vad_executor` (1 thread, order-preserving) and STT runs on its own thread, because the loop is what services the client's interrupt message.
- Client side (`static/chat.js`): `clearAudioPlayback()` must reset `nextPlayTime = 0` — it is a timestamp on the *old* turn's timeline, and leaving it set makes the next turn's first chunk get scheduled behind a tail that no longer exists.
- Client side (`static/chat.js`): `clearAudioPlayback()` must never close or replace the `AudioContext`. Mic capture lives on a separate `micContext`; closing the shared context killed `onaudioprocess` and the browser stopped sending audio entirely after the first barge-in.

**GPU generation is still strictly sequential across sentences** — sentence N+1's audio is not requested until sentence N's entire generation (all frames, not just first chunk) has drained, and only one GPU generation can be in flight at a time regardless (single model instance, single worker thread; confirmed no idle GPU gap exists between sentences — the worker picks up the next request within ~1ms of the previous one's EOS). The producer/consumer split above only removes the *non-GPU* overhead that used to sit on the same critical path (LLM read-ahead latency); it does not and cannot make two sentences generate concurrently on the GPU.

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

## Chunk splicing / audible clicks

Chunks are consecutive slices of one continuous waveform *within* a sentence (the Mimi decoder is stateful across `decode()` calls inside the `streaming(1)` context), but sentences are fully independent generations with no waveform continuity between them. Two rules follow, and breaking either produces clicking at chunk boundaries:

1. **The client must splice chunks on the AudioContext clock, not on `onended`.** `playAudioChunk` used to `await` the previous source's `onended` before creating and starting the next one, so the graph output sat at silence for at least a main-thread task (worse under GC/rAF load) in the middle of a continuous waveform — a step discontinuity on *every* boundary. `scheduleAudioChunk` (`static/chat.js`) now keeps a `nextPlayTime` cursor and calls `src.start(nextPlayTime)`, so each chunk begins exactly where the previous ended. The cursor only resyncs to `currentTime + SCHEDULE_LEAD` on a genuine underrun. Because the queue no longer advances via `onended`, "playback finished" is detected by `scheduleDrainCheck()` polling the cursor instead.
2. **The playback `AudioContext` must be opened at the stream's sample rate** (`ensurePlaybackContext`, 24 kHz = `SERVER_SAMPLE_RATE`). A default 44.1/48 kHz context resamples every chunk *in isolation* — no neighbouring samples to interpolate against, so both edges of every buffer are wrong, and at 44.1 kHz a chunk doesn't even span a whole number of output samples. That crackles regardless of how well the chunks are timed. This is why the context is created lazily on first user gesture rather than eagerly at default rate.

Sentence edges are the one real discontinuity, so `generate_stream` ramps them: `_apply_edge_fade` (`generator.py`) applies a 5ms raised cosine to the *first* chunk's head and the *last* chunk's tail only — never to interior boundaries, which would notch the waveform every `buffer_size` frames. To guarantee the last chunk is still in hand when the generation ends (it otherwise wouldn't be when the frame count lands on an exact multiple of `expected_frame_count`, or EOS falls on a batch boundary), `_split_carry` withholds the final 5ms of each yielded chunk and prepends it to the next. Nothing is dropped; it costs `EDGE_FADE_MS` of added latency. The fade-out also cleans up the hard cut left by `max_audio_length_ms` cap truncation and by the padding trim in the tail block.

## Key tunables for latency work

- `generator.py:187-190` - `initial_batch_size` / `initial_buffer_size` (frames before first chunk of a sentence). **Do not change these in isolation** — `self._stream_buffer_size` is hardcoded to `20` in three places (`generator.py:68, 655, 850`, re-pinned right after each `torch.compile` call) and the decoder pipeline (`model.decoder` compiled with `mode='reduce-overhead'`, i.e. CUDA graphs, at `generator.py:648/842`) is graph-captured against that fixed 20-frame chunk shape. Lowering `initial_buffer_size` alone (tried and reverted — see git history) feeds the compiled decode path a shape it was never captured for, forcing an unplanned recompile+cudagraph capture mid-request (~56s stall observed) and crashing `model_worker` with `captures_underway.empty() INTERNAL ASSERT FAILED` when warmup's `torch.cuda.empty_cache()` collides with the in-flight capture. Any change to buffer/batch sizing must also update all three `_stream_buffer_size = 20` sites and be warmed up through the same path `warmup_generator` (`generator.py:673`) exercises, or must avoid `reduce-overhead`/cudagraphs for the decoder entirely.
- `generator.py:308` - buffer flush threshold (`len(frame_buffer) >= buffer_size`).
- `main.py:419-420` - `max_audio_length_ms` cap formula (estimated_seconds * 1.6, floor 1500ms) — the estimator is noisy; widening the multiplier or floor reduces truncation risk at the cost of allowing longer stray generations.
- `main.py:737` - `model_queue.get(timeout=0.1)` poll interval, adds up to 100ms between one sentence's EOS and the next being picked up.
- `speak_streaming`'s inner loop (`main.py:586-604`) is where sequential dispatch happens; any pipelining fix (prefetch sentence N+1's TTS while N is still playing) goes here.

## Notes

- `audio_queue` (`main.py:108`) and the `sounddevice` import are dead code — actual playback is client-side over the websocket (`audio_chunk` messages relayed by `process_message_queue`).
- `Real-time factor` (`generator.py:386-389`) is printed via `print()`, not `logger` — won't show up in `session_data/server.log`, only stdout/console.
- `speak_streaming`'s docstring claims "LLM decoding and TTS generation overlap" — this is aspirational, not what the code currently does (see turn flow above).
