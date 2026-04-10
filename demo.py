import threading
import queue
import time
import sys
import re
import os
import io
import wave
import time
import sounddevice as sd
import torch
import numpy as np
from transformers import AutoModel, AutoProcessor, MoonshineStreamingForConditionalGeneration, AutoModelForSpeechSeq2Seq

# ==========================================
# IMPORT AI LIBRARIES
# ==========================================
import huggingface_hub as hf_hub
from transformers import AutoTokenizer
import openvino_genai as ov_genai
from piper.voice import PiperVoice

# ==========================================
# 0. PATH CONFIGURATION AND PARAMETERS
# ==========================================
PATH_PIPER = "C:/Users/danil/PycharmProjects/QwenKittenKhadas/en_US-lessac-medium.onnx"
SAMPLE_RATE_MIC = 16000
CHUNK_SIZE = 4000

# Thread-safe queues
queue_audio_in = queue.Queue()  # Mic -> STT
queue_stt_to_llm = queue.Queue()  # STT -> LLM
queue_llm_to_tts = queue.Queue()  # LLM -> TTS
queue_audio_out = queue.Queue()  # TTS -> Speakers

stop_event = threading.Event()
ai_is_speaking = threading.Event()
llm_is_generating = threading.Event()
shutdown_event = threading.Event()


# ==========================================
# 1. MODEL CLASSES (WRAPPERS)
# ==========================================
class STTModuleMoonshine:
    def __init__(self, model_id="UsefulSensors/moonshine-tiny"):
        print(f"\n🎤 [Init] Initializing Standard Moonshine ({model_id})...")

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True)

        self.device = "cpu"
        self.torch_dtype = torch.float32

        self.model.to(self.device).to(self.torch_dtype)
        self.model.eval()

        # Buffer to accumulate audio and counter for silence seconds
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_chunks = 0
        self.is_speaking = False
        print(f"✅ Standard Moonshine ready on {self.device}!")

    def process_audio(self, audio_chunk, sample_rate=16000):
        volume = np.sqrt(np.mean(audio_chunk ** 2))
        NOISE_THRESHOLD = 0.005

        # --- DEBUG LOG (Visual) ---
        # Print the current volume. If you see numbers like 0.001 you are silent.
        # If you see numbers like 0.020 you are speaking.
        # print(f"🔈 [DEBUG VAD] Current volume: {volume:.5f} | is_speaking: {self.is_speaking}")
        # ------------------------------

        if volume > NOISE_THRESHOLD:

            self.is_speaking = True
            self.silence_chunks = 0
            # Record the audio
            self.audio_buffer = np.concatenate((self.audio_buffer, audio_chunk.flatten()))

        else:
            # LOW VOLUME: Silence
            if self.is_speaking:
                # If we were recording, count this as a "pause" between words
                self.silence_chunks += 1
                self.audio_buffer = np.concatenate((self.audio_buffer, audio_chunk.flatten()))

                # If the pause lasts more than 1.5 seconds (6 chunks) -> sentence finished
                if self.silence_chunks > 6:

                    if len(self.audio_buffer) > sample_rate * 0.5:
                        inputs = self.processor(self.audio_buffer, sampling_rate=sample_rate, return_tensors="pt")
                        inputs = inputs.to(self.device, self.torch_dtype)

                        # --- STT STOPWATCH ---
                        t0_stt = time.perf_counter()

                        with torch.no_grad():
                            generated_ids = self.model.generate(**inputs)
                            testo_generato = self.processor.decode(generated_ids[0], skip_special_tokens=True)

                        t1_stt = time.perf_counter()
                        print(
                            f"\n⏱️ [BENCHMARK] 🎤 Moonshine STT: Inference completed in {t1_stt - t0_stt:.3f} seconds")
                        # ----------------------

                        self.reset_buffer()
                        return testo_generato, True
                    else:
                        print("🗑️ [DEBUG VAD] Audio too short, discarded.")
                        self.reset_buffer()
                        return "", False
            else:
                # If we were not recording and it's silent, ignore the chunk
                pass

        return "", False

    def reset_buffer(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_chunks = 0
        self.is_speaking = False


class LLMModuleQwenNPU:
    def __init__(self, model_id="OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov", device="NPU"):
        print(f"\n🧠 [Init] Qwen 2.5 on {device}...")
        self.local_model_path = "Qwen2.5-1.5B-Instruct-int4-ov"
        hf_hub.snapshot_download(model_id, local_dir=self.local_model_path)
        self.pipe = ov_genai.LLMPipeline(self.local_model_path, device)

        # Initialize persistent memory
        self.memory = PersistentMemory("Knowledge.txt")
        self.short_term_memory = []
        self.max_short_term = 3
        print(f"✅ Qwen ready. Persistent memory loaded: {len(self.memory.facts)} known facts.")

    def extract_key_facts(self, user_text, assistant_text):
        """Extracts personal facts only from the user's text to avoid hallucinations."""

        if len(user_text.strip()) < 8:
            return

        extraction_prompt = (
            "<|im_start|>system\n"
            "You are a personal data extractor. Your task is to read the user's sentence and determine if it reveals personal details (for example name, family, preferences, location, profession).\n"
            "RULES:\n"
            "- If it is a personal fact, write it in the third person (e.g., 'The user loves pizza').\n"
            "- If the user asks a question, gives a command, or talks about general facts (history, geography, AI, science, technology), you must reply EXACTLY with the word 'NONE'.\n"
            "Write down just the really short, concise and precise information.\n"
            "EXAMPLES:\n"
            "User: 'My name is Lorenzo' -> The user's name is Lorenzo\n"
            "User: 'What is the capital of France?' -> NONE\n"
            "User: 'I ate sushi today and I liked it' -> The user loves sushi\n"
            "User: 'Explain how an engine works' -> NONE<|im_end|>\n"
            f"<|im_start|>user\nUser: '{user_text}'<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # Generate the fact
        config = self.pipe.get_generation_config()
        config.max_new_tokens = 30
        res = self.pipe.generate(extraction_prompt, config)
        fact = str(res).strip()

        if "NONE" not in fact.upper() and len(fact) > 2:
            if self.memory.add_fact(fact):
                print(f"💾 [MEMORY] New personal fact saved: {fact}")
            else:
                print(f"♻️ [MEMORY] Fact already known, discarded: {fact}")

    def generate_stream(self, text):
        # Retrieve facts from the txt file
        facts_block = self.memory.get_context_string()

        # System prompt with persistent facts
        prompt = f"<|im_start|>system\nYour name is Micro. You're a short, concise and precise assistant, specifically designed to help people that can only hear your answers. Keep in mind this important infos about the user:{facts_block}<|im_end|>\n"

        # Add short-term history
        for msg in self.short_term_memory:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

        prompt += f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

        config = self.pipe.get_generation_config()
        config.max_new_tokens = 200

        token_queue = queue.Queue()

        # Native OpenVINO Callback. Triggers on each new token
        def streamer_callback(subword: str) -> bool:
            token_queue.put(subword)
            return False  # Returns False to NOT interrupt generation

        # Start the blocking NPU engine in an isolated mini-thread
        def background_generation():
            self.pipe.generate(prompt, config, streamer=streamer_callback)
            token_queue.put(None)

        gen_thread = threading.Thread(target=background_generation)
        gen_thread.start()

        full_response = ""
        t0_llm = time.perf_counter()
        first_token = True

        while True:
            token = token_queue.get()
            if token is None:
                break

            if first_token:
                t1_llm = time.perf_counter()
                print(f"\n⚡ [BENCHMARK] 🧠 Qwen TTFT: First word in {t1_llm - t0_llm:.3f} seconds!")
                first_token = False

            full_response += token
            yield token

        # Update Hybrid Memories at the end of streaming
        self.short_term_memory.append({"role": "user", "content": text})
        self.short_term_memory.append({"role": "assistant", "content": full_response})

        self.extract_key_facts(text, full_response)

        if len(self.short_term_memory) > self.max_short_term * 2:
            self.short_term_memory = self.short_term_memory[2:]
            print("✂️ [DEBUG MEMORY] Short-term memory full: sliding forward completed.")


class TTSModulePiper:
    def __init__(self, model_path):
        self.model_path = os.path.abspath(model_path)
        print(f"🔊 [Init] Piper TTS from: {self.model_path}")
        try:
            self.voice = PiperVoice.load(self.model_path)
            print("✅ Piper ready!")
        except Exception as e:
            print(f"❌ Piper Error: {e}")

    def synthesize_to_memory(self, text):
        clean_text = re.sub(r'[*#_~`"]', '', text)
        clean_text = re.sub(r'\n+', '. ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # If the string does not contain AT LEAST one letter or number (e.g., it's just a ":)" or a "-"), ignore it
        if not any(char.isalnum() for char in clean_text):
            return None, None

        try:
            t0_tts = time.perf_counter()
            memory_file = io.BytesIO()
            with wave.open(memory_file, 'wb') as wav_file:
                self.voice.synthesize_wav(clean_text, wav_file)

            memory_file.seek(0)

            if memory_file.getbuffer().nbytes == 0:
                return None, None

            with wave.open(memory_file, 'rb') as wav_reader:
                raw_audio = wav_reader.readframes(wav_reader.getnframes())
                sample_rate = wav_reader.getframerate()

            audio_np = np.frombuffer(raw_audio, dtype=np.int16)

            t1_tts = time.perf_counter()
            print(
                f"⏱️ [BENCHMARK] 🔊 Piper TTS: Audio of '{clean_text[:15]}...' generated in {t1_tts - t0_tts:.3f} seconds")

            return audio_np, sample_rate

        except Exception as e:
            # Silence the specific missing channels error
            if "channels not specified" in str(e):
                return None, None

            print(f"❌ TTS Synthesis Error: {e}")
            return None, None


class PersistentMemory:
    def __init__(self, file_path="user_facts.txt"):
        self.file_path = file_path
        self.facts = self._load_facts()

    def _load_facts(self):
        if not os.path.exists(self.file_path):
            return []
        with open(self.file_path, "r", encoding="utf-8") as f:
            # Load rows ignoring the empty ones
            return [line.strip() for line in f.readlines() if line.strip()]

    def add_fact(self, new_fact):
        # Basic redundancy check: avoid exact or very similar duplicates
        for existing_fact in self.facts:
            if new_fact.lower() in existing_fact.lower() or existing_fact.lower() in new_fact.lower():
                return False

        self.facts.append(new_fact)
        self._save_to_disk()
        return True

    def _save_to_disk(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            for fact in self.facts:
                f.write(f"{fact}\n")

    def get_context_string(self):
        if not self.facts:
            return ""
        # We return only the last 15 facts to avoid clogging the NPU
        recent_facts = self.facts[-15:]
        return "Previous knowledge about the user: " + " | ".join(recent_facts) + "."


# ==========================================
# 2. PIPELINE THREAD DEFINITION
# ==========================================

def thread_microphone():
    """Listens to the microphone and puts data into the queue continuously."""

    def callback(indata, frames, time, status):
        queue_audio_in.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE_MIC, channels=1, dtype='float32', blocksize=CHUNK_SIZE,
                        callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)


def thread_stt_moonshine(stt_model):
    print("🎤 [STT] Moonshine listening...")
    was_ai_speaking = False

    while not stop_event.is_set():
        try:
            audio_chunk = queue_audio_in.get(timeout=1)

            # Transition check: Did the AI just finish speaking at this moment?
            if was_ai_speaking and not ai_is_speaking.is_set():
                stt_model.reset_buffer()  # Throw away all useless audio

            was_ai_speaking = ai_is_speaking.is_set()

            if ai_is_speaking.is_set():
                stt_model.reset_buffer()
                continue

            testo_generato, is_final = stt_model.process_audio(audio_chunk, SAMPLE_RATE_MIC)

            if is_final and testo_generato.strip() != "":
                testo_pulito = testo_generato.strip()

                # anti-hallucination filter
                if testo_pulito.lower() in ["thank you.", "thank you", "subtitles by amara.org",
                                            "thanks for watching."] or "♪" in testo_pulito:
                    stt_model.reset_buffer()
                    continue

                print(f"\033[94mYou:\033[0m {testo_pulito}")
                queue_stt_to_llm.put(testo_pulito)

        except queue.Empty:
            continue


def thread_llm_qwen(llm_model):
    """Manages smooth screen printing and groups full sentences for TTS."""
    while not stop_event.is_set():
        try:
            user_text = queue_stt_to_llm.get(timeout=1)

            text_lower = user_text.lower()
            if ("micro" in text_lower or "mikro" in text_lower) and "bye" in text_lower:
                shutdown_event.set()

            ai_is_speaking.set()
            llm_is_generating.set()

            buffer_tts = ""
            sys.stdout.write("\033[92mAI:\033[0m ")
            sys.stdout.flush()

            for token_text in llm_model.generate_stream(user_text):
                sys.stdout.write(token_text)
                sys.stdout.flush()

                buffer_tts += token_text

                # divide only on strong punctuation
                if any(p in token_text for p in [".", "!", "?"]):
                    chunk_to_speak = buffer_tts.strip()

                    if len(chunk_to_speak) > 10:
                        queue_llm_to_tts.put(chunk_to_speak)
                        buffer_tts = ""

            if len(buffer_tts.strip()) > 0:
                queue_llm_to_tts.put(buffer_tts.strip())

            print()
            llm_is_generating.clear()

        except queue.Empty:
            continue


def thread_tts_piper(tts_model):
    """Generates audio silently in the background."""
    while not stop_event.is_set():
        try:
            text_chunk = queue_llm_to_tts.get(timeout=1)
            audio_data, sample_rate = tts_model.synthesize_to_memory(text_chunk)

            if audio_data is not None:
                queue_audio_out.put((audio_data, sample_rate))

        except queue.Empty:
            continue


def thread_audio_player():
    """Plays the audio sequentially and handles clean turn-taking."""
    while not stop_event.is_set():
        try:
            audio_data, sample_rate = queue_audio_out.get(timeout=1)

            sd.play(audio_data, samplerate=sample_rate)
            sd.wait()

            if queue_llm_to_tts.empty() and queue_audio_out.empty() and not llm_is_generating.is_set():

                if shutdown_event.is_set():
                    print("\n🛑 [SYSTEM] Shutting down Micro. Goodbye!")
                    stop_event.set()
                    break

                while not queue_audio_in.empty():
                    queue_audio_in.get()

                ai_is_speaking.clear()

                print("\n" + "-" * 40)
                print("🎤 [STT] Moonshine listening...")

        except queue.Empty:
            continue


# ==========================================
# 3. STARTUP
# ==========================================
if __name__ == "__main__":
    print("=== INITIALIZING EDGE AI PIPELINE ===")

    # Initialize AI models
    stt = STTModuleMoonshine()
    llm = LLMModuleQwenNPU()
    tts = TTSModulePiper(PATH_PIPER)

    print("\n=== STARTING DEMO OFFLINE ===")

    # Create threads
    t_mic = threading.Thread(target=thread_microphone)
    t_stt = threading.Thread(target=thread_stt_moonshine, args=(stt,))
    t_llm = threading.Thread(target=thread_llm_qwen, args=(llm,))
    t_tts = threading.Thread(target=thread_tts_piper, args=(tts,))
    t_play = threading.Thread(target=thread_audio_player)

    t_mic.start()
    t_stt.start()
    t_llm.start()
    t_tts.start()
    t_play.start()

    # --- MICRO'S INITIAL GREETING ---
    ai_is_speaking.set()
    saluto = "Hi! I'm Micro, how can I assist you today?"
    print(f"\n\033[92mAI:\033[0m {saluto}")
    queue_llm_to_tts.put(saluto)
    # -----------------------------------

    # --- LIFECYCLE AND HYBRID SHUTDOWN ---
    try:
        while not stop_event.is_set():
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 [SYSTEM] Manual interruption (Ctrl+C). Emergency shutdown...")
        stop_event.set()

    # --- CLEAN SHUTDOWN ---
    print("⏳ [SYSTEM] Disconnecting microphone and speakers...")

    t_mic.join()
    t_stt.join()
    t_llm.join()
    t_tts.join()
    t_play.join()

    print("=== PROGRAM TERMINATED SUCCESSFULLY ===")
    sys.exit(0)
