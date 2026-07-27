import asyncio
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  
os.environ["PYTORCH_DISABLE_CUDA_GRAPHS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import platform
import sqlite3
import time
import threading
import json
import queue
from fastapi.websockets import WebSocketState
import torch
import torchaudio
import sounddevice as sd
import numpy as np
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Optional
from generator import Segment, load_csm_1b_local
from llm_interface import LLMInterface
from rag_system import RAGSystem 
from vad import AudioStreamProcessor
from pydantic import BaseModel
import logging
from config import ConfigManager
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import re
speaking_start_time = 0.0          # set every time the AI begins a new turn
MIN_BARGE_LATENCY   = 0.9   
speaker_counters = {
    0: 0,  # AI
    1: 0   # User
}
current_generation_id = 1
pending_user_inputs = []
user_input_lock = threading.Lock()
audio_fade_duration = 0.3  # seconds for fade-out
last_interrupt_time = 0
interrupt_cooldown = 6.0  # seconds between allowed interrupts
audio_chunk_buffer = []  # Buffer to store the most recent audio chunks for fade-out
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
model_thread = None
model_queue = queue.Queue()
model_result_queue = queue.Queue()
model_thread_running = threading.Event()
llm_lock = threading.Lock()
audio_gen_lock = threading.Lock()
# Database
Base = declarative_base()
engine = create_engine("sqlite:///companion.db")
SessionLocal = sessionmaker(bind=engine)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    timestamp = Column(String)
    user_message = Column(Text)
    ai_message = Column(Text)
    audio_path = Column(String)

Base.metadata.create_all(bind=engine)

# Pydantic config schema
class CompanionConfig(BaseModel):
    system_prompt: str
    reference_audio_path: str
    reference_text: str
    reference_audio_path2: Optional[str] = None  # optional field
    reference_text2: Optional[str] = None  # optional field
    reference_audio_path3: Optional[str] = None  # optional field
    reference_text3: Optional[str] = None  # optional field
    model_path: str
    llm_path: str
    max_tokens: int = 8192
    voice_speaker_id: int = 0
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    embedding_model: str = "all-MiniLM-L6-v2"

# Global state
conversation_history = []
config = None
audio_queue = queue.Queue()
is_speaking = False
interrupt_flag = threading.Event()
# Set for the entire duration of an AI turn: from the moment the LLM starts
# streaming through the end of audio playback. Lets on_speech_start() detect
# a barge-in even before playback has begun (while CSM is still generating).
ai_turn_active = threading.Event()
generator = None
llm = None
rag = None
vad_processor = None
reference_segments = []
active_connections = []
message_queue = asyncio.Queue()

# Async event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
config_manager = ConfigManager()
model_id = "openai/whisper-tiny"
# Whisper
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
)            
whisper_model.to("cuda")
processor = AutoProcessor.from_pretrained(model_id)
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device='cuda',
)
# Background queue
async def process_message_queue():
    while True:
        message = await message_queue.get()
        for client in active_connections[:]:
            try:
                if client.client_state == WebSocketState.CONNECTED:
                    await client.send_json(message)
            except Exception as e:
                logger.error(f"Error in message queue for client: {e}")
                if client in active_connections:
                    active_connections.remove(client)
        message_queue.task_done()

def load_reference_segments(config_data: CompanionConfig):
    """Load multiple reference clips for voice‑cloning."""
    global reference_segments
    reference_segments = []
    
    # Load primary reference (required)
    if os.path.isfile(config_data.reference_audio_path):
        logger.info(f"Loading primary reference audio: {config_data.reference_audio_path}")
        wav, sr = torchaudio.load(config_data.reference_audio_path)
        wav = torchaudio.functional.resample(wav.squeeze(0),
                                         orig_freq=sr,
                                         new_freq=24_000)
        reference_segments.append(Segment(text=config_data.reference_text,
                                  speaker=config_data.voice_speaker_id,
                                  audio=wav))
    else:
        logger.warning(f"Primary reference audio '{config_data.reference_audio_path}' not found.")
    
    # Load second reference (optional)
    if config_data.reference_audio_path2 and os.path.isfile(config_data.reference_audio_path2):
        logger.info(f"Loading second reference audio: {config_data.reference_audio_path2}")
        wav, sr = torchaudio.load(config_data.reference_audio_path2)
        wav = torchaudio.functional.resample(wav.squeeze(0),
                                         orig_freq=sr,
                                         new_freq=24_000)
        reference_segments.append(Segment(text=config_data.reference_text2,
                                  speaker=config_data.voice_speaker_id,
                                  audio=wav))
    
    # Load third reference (optional)
    if config_data.reference_audio_path3 and os.path.isfile(config_data.reference_audio_path3):
        logger.info(f"Loading third reference audio: {config_data.reference_audio_path3}")
        wav, sr = torchaudio.load(config_data.reference_audio_path3)
        wav = torchaudio.functional.resample(wav.squeeze(0),
                                         orig_freq=sr,
                                         new_freq=24_000)
        reference_segments.append(Segment(text=config_data.reference_text3,
                                  speaker=config_data.voice_speaker_id,
                                  audio=wav))
    
    logger.info(f"Loaded {len(reference_segments)} reference audio segments.")

def transcribe_audio(audio_data, sample_rate):
    global whisper_model
    audio_np = np.array(audio_data).astype(np.float32)
    if sample_rate != 16000:
        try:
            audio_tensor = torch.tensor(audio_np).unsqueeze(0)
            audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sample_rate, new_freq=16000)
            audio_np = audio_tensor.squeeze(0).numpy()
        except: pass
    try:
        with torch.jit.optimized_execution(False):
            result = whisper_pipe(audio_np, generate_kwargs={"language": "english"}) 
            return result["text"]
    except:
        return "[Transcription error]"

def initialize_models(config_data: CompanionConfig):
    global generator, llm, rag, vad_processor, config
    config = config_data                         

    logger.info("Loading LLM …")
    llm = LLMInterface(config_data.llm_path,
                       config_data.max_tokens)

    logger.info("Loading RAG …")
    rag = RAGSystem("companion.db",
                    model_name=config_data.embedding_model)

    vad_model, vad_utils = torch.hub.load('snakers4/silero-vad',
                                          model='silero_vad',
                                          force_reload=False)
    vad_processor = AudioStreamProcessor(
        model=vad_model,
        utils=vad_utils,
        sample_rate=16_000,
        vad_threshold=config_data.vad_threshold,
        callbacks={"on_speech_start": on_speech_start,
                   "on_speech_end":   on_speech_end},
    )

    load_reference_segments(config_data)

    start_model_thread()

    logger.info("Compiling / warming‑up voice model …")
    t0 = time.time()

    # send a dummy request; max 0.5 s of audio, result discarded
    model_queue.put((
        "warm‑up.",                          # text
        config_data.voice_speaker_id,        # speaker
        [],                                  # no context
        500,                                 # max_ms
        0.7,                                 # temperature
        40,                                  # top‑k
    ))

    # block until worker signals EOS (None marker)
    while True:
        r = model_result_queue.get()
        if r is None:
            break

    logger.info(f"Voice model ready in {time.time() - t0:.1f}s")


def on_speech_start():
    global interrupt_flag

    should_interrupt = False
    if ai_turn_active.is_set():
        time_since_turn_start = time.time() - speaking_start_time if speaking_start_time > 0 else 999
        # Guard against the mic picking up the AI's own voice right as a turn
        # starts and immediately self-interrupting.
        if time_since_turn_start > MIN_BARGE_LATENCY and not interrupt_flag.is_set():
            logger.info("User started speaking during an active AI turn - aborting generation/playback")
            interrupt_flag.set()
            should_interrupt = True

    asyncio.run_coroutine_threadsafe(
        message_queue.put(
            {
                "type":   "vad_status",
                "status": "speech_started",
                "should_interrupt": should_interrupt,
            }
        ),
        loop,
    )

def on_speech_end(audio_data, sample_rate):
    try:
        logger.info("Transcription starting")
        user_text = transcribe_audio(audio_data, sample_rate)
        logger.info(f"Transcription completed: '{user_text}'")

        session_id = "default"
        speaker_id = 1
        index = speaker_counters[speaker_id]
        user_audio_path = f"audio/user/{session_id}_user_{index}.wav"
        os.makedirs(os.path.dirname(user_audio_path), exist_ok=True)

        audio_tensor = torch.tensor(audio_data).unsqueeze(0)
        save_audio_and_trim(user_audio_path, session_id, speaker_id, audio_tensor.squeeze(0), sample_rate)
        add_segment(user_text, speaker_id, audio_tensor.squeeze(0))

        logger.info(f"User audio saved and segment appended: {user_audio_path}")

        speaker_counters[speaker_id] += 1

        # Send transcription to clients
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "transcription", "text": user_text}),
            loop
        )
        
        threading.Thread(target=lambda: process_user_input(user_text, session_id), daemon=True).start()
    except Exception as e:
        logger.error(f"VAD callback failed: {e}")

def process_pending_inputs():
    """Process only the latest user input after an interruption"""
    global pending_user_inputs, is_speaking, interrupt_flag
    time.sleep(0.2)
    is_speaking = False
    interrupt_flag.clear()
    
    with user_input_lock:
        if not pending_user_inputs:
            logger.info("No pending user inputs to process")
            return
        
        # Only take the most recent input and ignore others
        latest_input = pending_user_inputs[-1]
        logger.info(f"Processing only latest input: '{latest_input[0]}'")
        
        # Clear all pending inputs
        pending_user_inputs = []
        
        # Process only the latest input
        user_text, session_id = latest_input
        process_user_input(user_text, session_id)

_SENTENCE_SPLIT_RE = re.compile(r'(.*?[.!?])(\s+|$)', re.DOTALL)

def extract_complete_sentences(buffer_text):
    """Split off complete sentences from a growing text buffer.

    Returns (list_of_complete_sentences, remainder_still_incomplete).
    """
    sentences = []
    end = 0
    for m in _SENTENCE_SPLIT_RE.finditer(buffer_text):
        sentence = m.group(1).strip()
        if sentence:
            sentences.append(sentence)
        end = m.end()
    return sentences, buffer_text[end:]


def _generate_sentence_audio(sentence_text, turn_context, playback_state, gen_id):
    """Run CSM on a single completed LLM sentence, streaming each resulting
    audio chunk to the client immediately as it's generated (true streaming
    playback - no waiting for the rest of the response). Also appends the
    generated audio to turn_context so the next sentence in this turn stays
    prosody-consistent with this one.

    playback_state is a dict shared across every sentence in this turn:
        {"all_chunks": [...], "chunk_counter": int, "gen_id": int}

    Cancellation is cooperative: generator.generate_stream() itself checks
    interrupt_flag (passed as cancel_event) and stops within about one frame,
    so there's no need to kill/restart the model worker thread here - we just
    stop forwarding chunks and let the worker's (now-fast) EOS marker arrive.

    Returns False if generation was aborted due to a barge-in, True otherwise.
    """
    sentence_text = preprocess_text_for_tts(sentence_text.lower())
    if not sentence_text:
        return True

    words = sentence_text.split()
    estimated_seconds = len(words) / (100 / 60)  # ~100 wpm
    max_audio_length_ms = max(int(estimated_seconds * 1000), 500)

    model_queue.put((
        sentence_text,
        config.voice_speaker_id,
        list(turn_context),
        max_audio_length_ms,
        0.8,  # temperature
        50    # topk
    ))

    sentence_chunks = []
    aborted = False
    while True:
        try:
            result = model_result_queue.get(timeout=0.5)
        except queue.Empty:
            if interrupt_flag.is_set():
                aborted = True
            continue

        if result is None:
            break
        if isinstance(result, Exception):
            logger.error(f"Generation {gen_id} - sentence generation error: {result}")
            break

        if interrupt_flag.is_set():
            aborted = True
            continue  # keep draining until EOS, but stop forwarding chunks

        chunk_array = result.cpu().numpy().astype(np.float32)
        sentence_chunks.append(chunk_array)

        playback_state["chunk_counter"] += 1
        chunk_num = playback_state["chunk_counter"]
        audio_queue.put(chunk_array)

        if chunk_num == 1:
            asyncio.run_coroutine_threadsafe(
                message_queue.put({"type": "audio_status", "status": "first_chunk", "gen_id": gen_id}),
                loop
            )

        asyncio.run_coroutine_threadsafe(
            message_queue.put({
                "type": "audio_chunk",
                "audio": chunk_array.tolist(),
                "sample_rate": generator.sample_rate,
                "gen_id": gen_id,
                "chunk_num": chunk_num
            }),
            loop
        )

    if aborted:
        return False

    playback_state["all_chunks"].extend(sentence_chunks)

    if sentence_chunks:
        sentence_audio = torch.cat([torch.from_numpy(c) for c in sentence_chunks])
        turn_context.append(Segment(text=sentence_text, speaker=config.voice_speaker_id, audio=sentence_audio))

    return True


def speak_streaming(user_text, session_id="default"):
    """Stream the LLM response and feed completed sentences to CSM as they
    arrive, so LLM decoding and TTS generation overlap instead of running
    fully sequentially. Each sentence's audio is sent to the client the
    moment CSM produces it - playback naturally begins as soon as the first
    chunk of the first sentence is ready, which is already after the user's
    turn has ended (generation only starts once STT has finished transcribing
    the completed utterance). If the user starts speaking again at any point
    - before the first chunk or mid-playback - everything in flight is
    aborted and the pending input is processed fresh once this turn unwinds.
    """
    global config, is_speaking, pending_user_inputs, interrupt_flag, current_generation_id, speaking_start_time

    if not audio_gen_lock.acquire(blocking=False):
        logger.warning("Generation already in progress, ignoring new input")
        return

    current_generation_id += 1
    this_id = current_generation_id

    interrupt_flag.clear()
    ai_turn_active.set()
    speaking_start_time = time.time()

    aborted = False
    turn_context = list(reference_segments)
    full_response_text = ""
    output_file = ""
    playback_state = {"all_chunks": [], "chunk_counter": 0, "gen_id": this_id}

    try:
        start_model_thread()

        context = "\n".join([f"User: {msg['user']}\nAI: {msg['ai']}" for msg in conversation_history[-5:]])
        rag_context = rag.query(user_text)
        system_prompt = config.system_prompt
        if rag_context:
            system_prompt += f"\n\nRelevant context:\n{rag_context}"

        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "status", "message": "Thinking..."}),
            loop
        )

        sentence_buffer = ""
        with llm_lock:
            for delta in llm.generate_response_stream(system_prompt, user_text, context):
                if interrupt_flag.is_set():
                    aborted = True
                    break

                sentence_buffer += delta
                full_response_text += delta

                # is_speaking flips True the moment the first sentence starts
                # generating audio, inside _generate_sentence_audio below.
                complete_sentences, sentence_buffer = extract_complete_sentences(sentence_buffer)
                for sentence in complete_sentences:
                    is_speaking = True
                    if not _generate_sentence_audio(sentence, turn_context, playback_state, this_id):
                        aborted = True
                        break
                if aborted:
                    break

            if not aborted and sentence_buffer.strip():
                is_speaking = True
                if not _generate_sentence_audio(sentence_buffer.strip(), turn_context, playback_state, this_id):
                    aborted = True

        full_response_text = full_response_text.strip()

        if aborted:
            logger.info(f"Generation {this_id} - aborted (user started speaking)")
            asyncio.run_coroutine_threadsafe(
                message_queue.put({"type": "audio_status", "status": "interrupted", "gen_id": this_id}),
                loop
            )
            return

        if not full_response_text:
            return

        # Save to history/DB and notify the client of the full text once known.
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        conversation_history.append({
            "timestamp": timestamp,
            "user": user_text,
            "ai": full_response_text
        })

        try:
            db = SessionLocal()
            conv = Conversation(
                session_id=session_id,
                timestamp=timestamp,
                user_message=user_text,
                ai_message=full_response_text,
                audio_path=""
            )
            db.add(conv)
            db.commit()
            index = speaker_counters[0]
            output_file = f"audio/ai/{session_id}_response_{index}.wav"
            speaker_counters[0] += 1
            conv.audio_path = output_file
            db.commit()
            db.close()
        except Exception as e:
            logger.error(f"Database error: {e}")

        threading.Thread(target=lambda: rag.add_conversation(user_text, full_response_text), daemon=True).start()

        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "response", "text": full_response_text}),
            loop
        )

        if playback_state["all_chunks"]:
            try:
                complete_audio = torch.cat([torch.from_numpy(c) for c in playback_state["all_chunks"]])
                if output_file:
                    save_audio_and_trim(output_file, session_id, config.voice_speaker_id, complete_audio, generator.sample_rate)
                add_segment(full_response_text.lower(), config.voice_speaker_id, complete_audio)

                total_audio_seconds = complete_audio.size(0) / generator.sample_rate
                logger.info(f"Generation {this_id} - played {total_audio_seconds:.2f}s of audio")
            except Exception as e:
                logger.error(f"Generation {this_id} - error saving complete audio: {e}")

    except Exception as e:
        import traceback
        logger.error(f"Generation {this_id} - unexpected error: {e}\n{traceback.format_exc()}")
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "error", "message": "Failed to generate response"}),
            loop
        )
    finally:
        is_speaking = False
        ai_turn_active.clear()
        audio_queue.put(None)

        try:
            asyncio.run_coroutine_threadsafe(
                message_queue.put({"type": "audio_status", "status": "complete", "gen_id": this_id}),
                loop
            )
        except Exception as e:
            logger.error(f"Generation {this_id} - failed to send completion status: {e}")

        with user_input_lock:
            if pending_user_inputs:
                logger.info(f"Generation {this_id} - processing pending input queued during this turn")
                process_pending_inputs()

        audio_gen_lock.release()


def process_user_input(user_text, session_id="default"):
    global pending_user_inputs, interrupt_flag

    if not user_text or user_text.strip() == "":
        logger.warning("Empty user input received, ignoring")
        return

    if ai_turn_active.is_set():
        logger.info(f"AI turn in progress, replacing pending input with: '{user_text}'")

        with user_input_lock:
            pending_user_inputs = [(user_text, session_id)]

        if not interrupt_flag.is_set():
            logger.info("Interrupting current generation/speech for new input")
            interrupt_flag.set()
            asyncio.run_coroutine_threadsafe(
                message_queue.put({"type": "audio_status", "status": "interrupted"}),
                loop
            )
        return

    logger.info(f"Processing user input: '{user_text}'")
    threading.Thread(target=speak_streaming, args=(user_text, session_id), daemon=True).start()

def model_worker(cfg: CompanionConfig):
    global generator, model_thread_running

    logger.info("Model worker thread started")

    if generator is None:
        torch._inductor.config.triton.cudagraphs = False  # Disable cudagraphs
        torch._inductor.config.fx_graph_cache = False  # Disable graph caching
        logger.info("Loading voice model inside worker thread …")
        generator = load_csm_1b_local(cfg.model_path, "cuda")
        logger.info("Voice model ready (compiled with cudagraphs)")

    while model_thread_running.is_set():
        try:
            request = model_queue.get(timeout=0.1)
            if request is None:
                break

            text, speaker_id, context, max_ms, temperature, topk = request

            for chunk in generator.generate_stream(
                    text=text,
                    speaker=speaker_id,
                    context=context,
                    max_audio_length_ms=max_ms,
                    cancel_event=interrupt_flag,
                    temperature=temperature,
                    topk=topk):
                model_result_queue.put(chunk)

                if not model_thread_running.is_set():
                    break

            model_result_queue.put(None) # EOS marker

        except queue.Empty:
            continue
        except Exception as e:
            import traceback
            logger.error(f"Error in model worker: {e}\n{traceback.format_exc()}")
            model_result_queue.put(Exception(f"Generation error: {e}"))

    logger.info("Model worker thread exiting")

def start_model_thread():
    global model_thread, model_thread_running

    if model_thread is not None and model_thread.is_alive():
        return                        

    model_thread_running.set()
    model_thread = threading.Thread(target=model_worker,
                                    args=(config,),
                                    daemon=True,
                                    name="model_worker")
    model_thread.start()
    logger.info("Started dedicated model worker thread")

def send_to_all_clients(message: dict):
    """Send a message to all connected WebSocket clients"""
    for client in active_connections[:]:
        try:
            if client.client_state == WebSocketState.CONNECTED:
                asyncio.run_coroutine_threadsafe(client.send_json(message), loop)
                logger.info(f"Sent message to client: {message}")
            else:
                logger.warning("Detected non-connected client; removing from active_connections")
                active_connections.remove(client)
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
            if client in active_connections:
                active_connections.remove(client)

saved_audio_paths = {
    "default": {
        0: [],  # AI
        1: []   # User
    }
}
MAX_AUDIO_FILES = 8

def save_audio_and_trim(path, session_id, speaker_id, tensor, sample_rate):
    """
    Save audio file and trim old audio files for both AI and user to maintain storage limits.
    
    Args:
        path: Path to save the audio file
        session_id: Conversation session ID
        speaker_id: 0 for AI, 1 for user
        tensor: Audio tensor to save
        sample_rate: Audio sample rate
    """
    torchaudio.save(path, tensor.unsqueeze(0), sample_rate)
    
    saved_audio_paths.setdefault(session_id, {}).setdefault(speaker_id, []).append(path)
    
    paths = saved_audio_paths[session_id][speaker_id]
    while len(paths) > MAX_AUDIO_FILES:
        old_path = paths.pop(0)
        if os.path.exists(old_path):
            os.remove(old_path)
            logger.info(f"Removed old audio file: {old_path}")
    
    other_speaker_id = 1 if speaker_id == 0 else 0
    if other_speaker_id in saved_audio_paths[session_id]:
        other_paths = saved_audio_paths[session_id][other_speaker_id]
        while len(other_paths) > MAX_AUDIO_FILES:
            old_path = other_paths.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)
                logger.info(f"Removed old audio file from other speaker: {old_path}")

# CSM's audio context window is only for voice continuity, not conversation
# memory (that's handled separately by the RAG system + LLM history). Keep it
# small so the model never approaches its fixed 2048-token position limit,
# no matter how long the overall conversation runs.
MAX_DYNAMIC_VOICE_TURNS = 2  # each turn = 1 user segment + 1 AI segment
MAX_SEGMENTS = MAX_DYNAMIC_VOICE_TURNS * 2

def add_segment(text, speaker_id, audio_tensor):
    """
    Add a new segment and ensure the total context stays within token limits.
    This version correctly separates protected and dynamic segments, performs trimming
    on the dynamic list, and rebuilds the global context list at the end.

    Args:
        text: Text content of the segment
        speaker_id: ID of the speaker (0 for AI, 1 for user)
        audio_tensor: Audio data as a tensor
    """
    global reference_segments, generator, config

    # Determine the number of protected, initial reference segments based on what was actually loaded.
    num_protected_segments = 0
    if config.reference_audio_path and os.path.exists(config.reference_audio_path):
        num_protected_segments += 1
    if config.reference_audio_path2 and os.path.exists(config.reference_audio_path2):
        num_protected_segments += 1
    if config.reference_audio_path3 and os.path.exists(config.reference_audio_path3):
        num_protected_segments += 1

    # Separate protected from dynamic segments from the current global state
    protected_segments = reference_segments[:num_protected_segments]
    dynamic_segments = reference_segments[num_protected_segments:]

    # Add the new segment to the dynamic list
    new_segment = Segment(text=text, speaker=speaker_id, audio=audio_tensor)
    dynamic_segments.append(new_segment)

    # Keep only the last MAX_DYNAMIC_VOICE_TURNS conversational turns of audio
    # context, regardless of how many protected reference clips are configured.
    if len(dynamic_segments) > MAX_SEGMENTS:
        dynamic_segments = dynamic_segments[-MAX_SEGMENTS:]

    # Then, check and trim by token count if necessary.
    # This loop will trim the oldest dynamic segments until the token count is acceptable.
    if hasattr(generator, '_text_tokenizer'):
        while dynamic_segments:
            # Tentatively combine for token calculation
            temp_full_list = protected_segments + dynamic_segments
            total_tokens = 0

            # Calculate total tokens for the current combination
            for segment in temp_full_list:
                tokens = generator._text_tokenizer.encode(f"[{segment.speaker}]{segment.text}")
                total_tokens += len(tokens)
                if segment.audio is not None:
                    # Approximate frame count to token conversion
                    audio_frames = segment.audio.size(0) // 6094
                    total_tokens += audio_frames

            # Model's hard position limit is 2048; leave headroom for the new
            # text segment plus the frames about to be generated.
            if total_tokens <= 1024:
                break

            # Otherwise, remove the oldest dynamic segment and re-check in the next loop iteration.
            dynamic_segments.pop(0)

    else:
        # Fallback if tokenizer is not available
        logger.warning("Unable to access tokenizer - falling back to word-based estimation for context trimming")

        def estimate_tokens(segment):
            words = segment.text.split()
            punctuation = sum(1 for char in segment.text if char in ".,!?;:\"'()[]{}")
            text_tokens = len(words) + punctuation
            audio_tokens = 0
            if segment.audio is not None:
                audio_frames = segment.audio.size(0) // 6094
                audio_tokens = audio_frames
            return text_tokens + audio_tokens

        while dynamic_segments:
            total_estimated_tokens = sum(estimate_tokens(s) for s in protected_segments) + \
                                     sum(estimate_tokens(s) for s in dynamic_segments)
            if total_estimated_tokens <= 1024:
                break
            dynamic_segments.pop(0)

    # Finally, overwrite the global variable with the new, correctly-trimmed list.
    # This is the single source of truth for the update.
    reference_segments = protected_segments + dynamic_segments

    # Log the final state for debugging
    logger.info(f"Context updated. Segments: {len(reference_segments)} total " +
                f"({len(protected_segments)} protected, {len(dynamic_segments)} dynamic).")

def preprocess_text_for_tts(text):
    """
    Removes all punctuation except periods, commas, exclamation points, and question marks
    from the input text to create cleaner speech output while preserving intonation.
    Args:
    text (str): Input text with potential punctuation
    Returns:
    str: Cleaned text with only allowed punctuation
    """
    # Define a regex pattern that matches all punctuation except periods, commas, exclamation points, and question marks
    # This includes: ; : " '  ~ @ # $ % ^ & * ( ) _ - + = [ ] { } \ | / < >
    pattern = r'[^\w\s.,!?\']'
    # Replace matched punctuation with empty string
    cleaned_text = re.sub(pattern, '', text)
    # normalize multiple spaces to single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # ensure there's a space after punctuation for better speech pacing
    cleaned_text = re.sub(r'([.,!?])(\S)', r'\1 \2', cleaned_text)
    return cleaned_text.strip()

def handle_interrupt(websocket):
    global is_speaking, last_interrupt_time, interrupt_flag, model_thread_running, speaking_start_time
    
    # Log the current state
    logger.info(f"Interrupt requested. Current state: is_speaking={is_speaking}")
    
    current_time = time.time()
    time_since_speech_start = current_time - speaking_start_time if speaking_start_time > 0 else 999
    time_since_last_interrupt = current_time - last_interrupt_time
    
    # Only apply cooldown for established speech, not for new speech
    if time_since_last_interrupt < interrupt_cooldown and time_since_speech_start > 3.0:
        logger.info(f"Ignoring interrupt: too soon after previous interrupt ({time_since_last_interrupt:.1f}s < {interrupt_cooldown}s)")
        # Let the client know we're not interrupting
        asyncio.run_coroutine_threadsafe(
           websocket.send_json({
               "type": "audio_status",
               "status": "interrupt_acknowledged",
               "success": False,
               "reason": "cooldown"
           }),
           loop
        )
        return False
    
    # Update the last interrupt time
    last_interrupt_time = current_time
    
    # We should interrupt if we're speaking, generating (even pre-playback), or model generation is in progress
    if is_speaking or ai_turn_active.is_set() or not model_result_queue.empty():
        logger.info("Interruption processing: we are speaking or generating")
        
        interrupt_flag.set()
        
        # Notify clients
        asyncio.run_coroutine_threadsafe(
            message_queue.put({"type": "audio_status", "status": "interrupted"}),
            loop
        )
        
        asyncio.run_coroutine_threadsafe(
           websocket.send_json({
               "type": "audio_status",
               "status": "interrupt_acknowledged"
           }),
           loop
        )
        
        # Clear the audio queue to stop additional audio from being processed
        try:
            # Drain the existing queue
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Add end signal
            audio_queue.put(None)
            logger.info("Audio queue cleared")
        except Exception as e:
            logger.error(f"Error clearing audio queue: {e}")

        # NOTE: deliberately NOT calling vad_processor.reset() here. This
        # interrupt commonly fires *while* VAD is actively collecting the
        # user's new utterance (the one that triggered the barge-in) -
        # resetting it mid-collection would wipe that in-progress audio
        # buffer and truncate the very speech we're trying to respond to.
        # VAD's own is_collecting/silent_chunk_count state machine already
        # handles turn boundaries correctly without an external reset.

        # generator.generate_stream() checks interrupt_flag itself (passed as
        # cancel_event) and stops within about one frame, so the model worker
        # doesn't need to be killed/restarted here - it'll pick up the next
        # queued request on its own once the current one exits.

        return True
    
    logger.info("No active speech to interrupt")
    return False

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global is_speaking, audio_queue
    
    await websocket.accept()
    active_connections.append(websocket)
    
    saved = config_manager.load_config()
    if saved:
        await websocket.send_json({"type": "saved_config", "config": saved})
        
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "config":
                # Config handling
                try:
                    config_data = data["config"]
                    
                    logger.info(f"Received config data keys: {config_data.keys()}")

                    for key in ["reference_audio_path", "reference_audio_path2", "reference_audio_path3",
                               "reference_text", "reference_text2", "reference_text3"]:
                        if key in config_data:
                            logger.info(f"Config includes {key}: {config_data[key]}")
                        else:
                            logger.warning(f"Config missing {key}")
                    
                    conf = CompanionConfig(**config_data)
                    
                    saved = config_manager.save_config(config_data)
                    
                    if saved:
                        initialize_models(conf)
                        await websocket.send_json({"type": "status", "message": "Models initialized and configuration saved"})
                    else:
                        await websocket.send_json({"type": "error", "message": "Failed to save configuration"})
                        
                except Exception as e:
                    logger.error(f"Error processing config: {str(e)}")
                    await websocket.send_json({"type": "error", "message": f"Configuration error: {str(e)}"})
                
                
            elif data["type"] == "request_saved_config":
                saved = config_manager.load_config()
                await websocket.send_json({"type": "saved_config", "config": saved})
            
            elif data["type"] == "text_message":
                user_text   = data["text"]
                session_id  = data.get("session_id", "default")
                logger.info(f"TEXT-MSG from client: {user_text!r}")

                # If the model is already talking, queue the request but
                if is_speaking:
                    with user_input_lock:
                        if len(pending_user_inputs) >= 3:
                            pending_user_inputs = pending_user_inputs[-2:]
                        pending_user_inputs.append((user_text, session_id))
                    await websocket.send_json(
                        {"type":"status","message":"Queued – I’ll answer in a moment"})
                    continue                         

                await message_queue.put({"type":"transcription","text":user_text})
                threading.Thread(
                    target=lambda: process_user_input(user_text, session_id),
                    daemon=True).start()
                
            elif data["type"] == "audio":
                audio_data = np.asarray(data["audio"], dtype=np.float32)
                sample_rate = data["sample_rate"]

                if sample_rate != 16000:
                    audio_tensor = torch.tensor(audio_data).unsqueeze(0)
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor, orig_freq=sample_rate, new_freq=16000
                    )
                    audio_data  = audio_tensor.squeeze(0).numpy()
                    sample_rate = 16000

                if config and config.vad_enabled:
                    vad_processor.process_audio(audio_data)  
                else:
                    text = transcribe_audio(audio_data, sample_rate)
                    await websocket.send_json({"type": "transcription", "text": text})
                    await message_queue.put({"type": "transcription", "text": text})

                    if is_speaking:
                        with user_input_lock:
                            pending_user_inputs.append((text, "default"))
                    else:
                        process_user_input(text)

                        
            elif data["type"] == "interrupt":
                logger.info("Explicit interrupt request received")
                
                # Always acknowledge receipt of interrupt request
                await websocket.send_json({
                    "type": "audio_status", 
                    "status": "interrupt_acknowledged"
                })
                
                # Then try to handle the actual interrupt
                success = handle_interrupt(websocket)
                
                # If successful, allow a brief delay for clearing everything
                if success:
                    await asyncio.sleep(0.3)  # Short delay to allow complete clearing
                    
                    # Force process pending inputs after interrupt
                    with user_input_lock:
                        if pending_user_inputs:
                            user_text, session_id = pending_user_inputs.pop(0)
                            pending_user_inputs.clear()  # Clear any backup to avoid multiple responses
                            
                            # Process in a new thread to avoid blocking
                            threading.Thread(
                                target=lambda: process_user_input(user_text, session_id),
                                daemon=True
                            ).start()
                
                # Send final status update about the interrupt
                await websocket.send_json({
                    "type": "audio_status", 
                    "status": "interrupted",
                    "success": success
                })
                
            elif data["type"] == "mute":
                await websocket.send_json({"type": "mute_status", "muted": data["muted"]})
                if not data["muted"] and config and config.vad_enabled:
                    vad_processor.reset()
                    
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    return templates.TemplateResponse(request, "setup.html")

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html")

@app.on_event("startup")
async def startup_event():
    os.makedirs("static", exist_ok=True)
    os.makedirs("audio/user", exist_ok=True)
    os.makedirs("audio/ai", exist_ok=True)
    os.makedirs("embeddings_cache", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    with open("templates/index.html", "w") as f:
        f.write("""<meta http-equiv="refresh" content="0; url=/setup" />""")
    try:
        torch.hub.load('snakers4/silero-vad', model='silero_vad', force_reload=False)
    except: pass
    asyncio.create_task(process_message_queue())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Server shutting down...")

class ConversationUpdate(BaseModel):
    user_message: str
    ai_message: str

@app.get("/api/conversations")
async def get_conversations(request: Request):
    conn = sqlite3.connect("companion.db")
    cur = conn.cursor()
    cur.execute("SELECT id, user_message, ai_message FROM conversations ORDER BY id DESC")
    data = [{"id": row[0], "user_message": row[1], "ai_message": row[2]} for row in cur.fetchall()]
    conn.close()
    return JSONResponse(content=data)

@app.put("/api/conversations/{conv_id}")
async def update_conversation(conv_id: int, update: ConversationUpdate):
    conn = sqlite3.connect("companion.db")
    cur = conn.cursor()
    cur.execute("UPDATE conversations SET user_message=?, ai_message=? WHERE id=?",
                (update.user_message, update.ai_message, conv_id))
    conn.commit()
    conn.close()
    return JSONResponse(content={"status": "updated", "id": conv_id})

@app.delete("/api/conversations")
async def delete_all_conversations():
    try:
        conn = sqlite3.connect("companion.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM conversations")
        conn.commit()
        conn.close()
        return {"status": "all deleted"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: int):
    try:
        conn = sqlite3.connect("companion.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()
        conn.close()
        return JSONResponse(content={"status": "deleted", "id": conv_id})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/crud", response_class=HTMLResponse)
async def crud_ui(request: Request):
    return templates.TemplateResponse(request, "crud.html")

if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=lambda: asyncio.run(loop.run_forever()), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
