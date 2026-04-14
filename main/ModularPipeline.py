import os
import re
import time
import random
import psutil
import librosa
import pandas as pd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.io.wavfile
from pocket_tts import TTSModel
import openvino_genai as ov_genai
import huggingface_hub as hf_hub
from pydub import AudioSegment
from llama_cpp import Llama
from pywhispercpp.model import Model
import sentencepiece
import sphn
from dataclasses import dataclass
import jiwer
from jiwer import wer
from datasets import load_dataset
from huggingface_hub import snapshot_download
from huggingface_hub import hf_hub_download
import openvino as ov
import urllib.request
import imageio_ffmpeg
from soprano import SopranoTTS

# Model imports
from kittentts import KittenTTS
from supertonic import TTS
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, pipeline
from optimum.intel import OVModelForSpeechSeq2Seq, OVModelForCausalLM
import subprocess
from piper import PiperVoice
import wave
import torch
import torch._dynamo

torch._dynamo.config.disable = True
import onnxruntime as ort
from kokoro_onnx import Kokoro
from moshi.models import loaders, MimiModel, LMModel, LMGen
from pathlib import Path
import json
from kokoro import KPipeline
from kokoro.model import KModel
import traceback
import io
import sys
from datasets import Audio
import moonshine_voice
import outetts
from outetts import (Interface, ModelConfig, Backend, Models, LlamaCppQuantization, GenerationConfig, GenerationType, SamplerConfig)
from transformers import MoonshineForConditionalGeneration, AutoProcessor


# ==========================================
# PATHS AND PARAMETERS CONFIGURATION
# ==========================================
REFERENCE_TEXT = ("The rapid evolution of artificial intelligence is fundamentally transforming modern computing architectures. By shifting operations from centralized cloud servers to specialized edge devices, developers can significantly reduce inference latency and protect user privacy. Deploying highly optimized neural networks on local hardware requires efficient unified memory allocation and strict power management. Nevertheless, overcoming the limitations of thermal throttling and memory bandwidth remains a critical challenge for the next generation of embedded systems. What do you think about that")
PATH_WHISPER_SMALL = "C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/Models/whisper-small-ov-gpu"
PATH_QWEN3_TTS_OV = "C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/Models/qwen3-tts-0.6b-ov"
PATH_QWEN3_ASR_OV = "C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/Models/Qwen3-ASR-0.6B-OV"
PATH_WHISPER_TINY = "C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/Models/whisper-tiny-ov"
PATH_WHISPER_SMALL_DISTILL = "C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/Models/whisper-small-distill"
PATH_WHISPER_BASE = "C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/Models/whisper-base-ov-gpu"
PATH_WHISPER_MEDIUM = "C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/Models/whisper-medium-ov"
PATH_WHISPER_V3_TURBO = "C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/Models/whisper-v3-turbo-ov"
PATH_QWEN = "C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/Models/qwen-0.6b-ov-gpu"
PATH_INPUT = "C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/QwenKittenWhisperInput/EdgAIFoundationsTest.m4a"
FUN_ASR_PATH_OV = "C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/Models/Fun-ASR-Nano-2512-ov"
#REFERENCE_TEXT = ("Hi model how are you is everything fine")

PATH_TTS = "C:/Users/danil/PycharmProjects/QwenKittenKhadas/en_US-lessac-medium.onnx"
PATH_KOKORO_OV = "C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/Models/Kokoro-82M-OpenVino"

# Base folders for dynamic outputs
BASE_DIR_OUTPUT = "C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/QwenKittenWhisperOutput"
BASE_DIR_CSV = "C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/CSV_Benchmark"
BASE_DIR_CHARTS = "C:/Users/danil/Desktop/Lorenzo/QwenKittenWhisper/Performance"

os.makedirs(BASE_DIR_OUTPUT, exist_ok=True)
os.makedirs(BASE_DIR_CSV, exist_ok=True)
os.makedirs(BASE_DIR_CHARTS, exist_ok=True)


# ==========================================
# CLASS DEFINITIONS (MODULES)
# ==========================================

def smart_split_for_tts(text):
    text = text.replace("**", "").replace("*", "").replace("#", "").replace("`", "")
    chunks = re.split(r'(?<=[.,!?;:\n])\s+', text)
    return [c.strip() for c in chunks if len(c.strip()) > 1]


def clean_text_for_tts(text):
    text = text.replace("**", "").replace("*", "")
    text = text.replace("#", "")
    text = text.replace("`", "")
    text = re.sub(r'\n+', ' ', text)
    return text.strip()


class OfficialLiquidONNX:
    """Handles the Official Liquid LFM ONNX engine for processing audio using subprocess."""
    def __init__(self, model_path="Models/liquid-lfm2.5-audio-onnx", engine_dir="onnx-export"):
        self.model_path = os.path.abspath(model_path)
        self.engine_dir = os.path.abspath(engine_dir)
        print(f"🎙️ Initializing Official Liquid Engine from: {self.engine_dir}")

    def process_audio(self, input_path, output_wav_path):
        import os
        from pydub import AudioSegment
        import subprocess
        import time
        import psutil

        print(f"🎙️ Preparing audio for Liquid...")
        wav_input_path = input_path
        if not input_path.lower().endswith('.wav'):
            print(f"🔄 Forced conversion to .wav (16kHz, Mono)...")
            audio = AudioSegment.from_file(input_path)
            wav_input_path = os.path.abspath("temp_liquid_input.wav")
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_input_path, format="wav")
            print(f"✅ Audio converted: {wav_input_path}")

        comando = [
            "uv", "run", "lfm2-audio-infer",
            self.model_path,
            "--mode", "interleaved",
            "--audio", wav_input_path,
            "--output", os.path.abspath(output_wav_path),
            "--precision", "fp16"
        ]

        print(f"🚀 Launching Liquid inference (Hardware Monitoring active)...")

        t0 = time.perf_counter()

        process = subprocess.Popen(
            comando,
            cwd=self.engine_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        cpu_run = []
        ram_run = []

        while process.poll() is None:
            try:
                cpu_run.append(psutil.cpu_percent(interval=0.2))
                ram_run.append(psutil.virtual_memory().percent)
            except:
                pass

        stdout, stderr = process.communicate()
        t1 = time.perf_counter()

        if process.returncode != 0:
            print("❌ Error during Liquid inference:")
            print(stderr)
            return None, 0, 0, 0

        latenza_totale = t1 - t0
        avg_cpu = sum(cpu_run) / len(cpu_run) if cpu_run else psutil.cpu_percent()
        max_ram = max(ram_run) if ram_run else psutil.virtual_memory().percent

        print("\n🔍 --- OPENVINO HARDWARE DIAGNOSTICS ---")
        hardware_found = False
        log_completo = stdout + "\n" + stderr

        for line in log_completo.split('\n'):
            if any(keyword in line for keyword in
                   ["OpenVINOExecutionProvider", "Hardware Acceleration", "device_type", "CPUExecutionProvider"]):
                print(f"   ⚙️ {line.strip()}")
                hardware_found = True

        if not hardware_found:
            print("   ⚠️ Warning: No hardware acceleration log detected.")
            print("   The model is likely running on standard CPU (Fallback).")
        print("-----------------------------------------")

        return output_wav_path, latenza_totale, avg_cpu, max_ram


def test_liquid_latency(liquid_module, input_audio_array, sample_rate=16000):
    duration_in = len(input_audio_array) / sample_rate
    t0 = time.time()
    output_audio_array = liquid_module.process_audio(input_audio_array)
    t1 = time.time()
    total_time = t1 - t0
    duration_out = len(output_audio_array) / sample_rate
    rtf_e2e = total_time / duration_out
    print(f"⏱️ Response Latency (Time-to-Audio): {total_time:.3f} s")
    print(f"⚡ End-to-End RTF: {rtf_e2e:.3f}x")
    return total_time, rtf_e2e


@dataclass
class InferenceState:
    """Manages the state and streaming components for Moshi/Mimi models."""
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    device: str
    frame_size: int
    batch_size: int

    def __init__(self, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor, lm: LMModel,
                 batch_size: int, device: str):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.lm_gen = LMGen(lm, temp=0, temp_text=0, use_sampling=False)
        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.batch_size = batch_size
        self.mimi.streaming_forever(batch_size)
        self.lm_gen.streaming_forever(batch_size)

    def run(self, in_pcms: torch.Tensor):
        ntokens = 0
        first_frame = True
        chunks = [c for c in in_pcms.split(self.frame_size, dim=2) if c.shape[-1] == self.frame_size]
        all_text = []

        for chunk in chunks:
            codes = self.mimi.encode(chunk.to(self.device))
            if first_frame:
                tokens = self.lm_gen.step(codes)
                first_frame = False
            tokens = self.lm_gen.step(codes)
            if tokens is None:
                continue

            one_text = tokens[0, 0].cpu()
            if one_text.item() not in [0, 3]:
                text = self.text_tokenizer.id_to_piece(one_text.item())
                text = text.replace("▁", " ")
                all_text.append(text)
            ntokens += 1

        return "".join(all_text)


class STTModuleFunASRNanoOpenVINO:
    """Fun-ASR-Nano model optimized for Intel hardware via OpenVINO."""

    def __init__(self, model_path= FUN_ASR_PATH_OV, device="CPU"):
        print(f"🎙️ Initializing STT (Fun-ASR-Nano OpenVINO) on {device}...")
        self.is_loaded = False

        helper_url = "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/funasr-nano/ov_funasr_helper.py"
        helper_path = "ov_funasr_helper.py"

        if not os.path.exists(helper_path):
            print(f"📥 Downloading {helper_path}...")
            try:
                urllib.request.urlretrieve(helper_url, helper_path)
            except Exception as e:
                print(f"❌ Error downloading {helper_path}: {e}")
                return

        if not os.path.exists(model_path):
            print(f"❌ Error: Folder not found in {model_path}.")
            print("👉 Run the Colab notebook first to export the model to OpenVINO!")
            return

        try:
            from ov_funasr_helper import OVFunASRNano

            print(f"🧠 Loading OpenVINO weights on {device.upper()}...")

            llm_ov_config = {"PERFORMANCE_HINT": "LATENCY",
                             "INFERENCE_PRECISION_HINT": "f32"}

            self.model = OVFunASRNano(
                Path(model_path),
                device=device.upper(),
                llm_ov_config=llm_ov_config
            )

            try:
                if hasattr(self.model, 'llm') and hasattr(self.model.llm, 'generation_config'):
                    self.model.llm.generation_config.do_sample = False
            except:
                pass

            self.is_loaded = True
            print("✅ Fun-ASR-Nano OpenVINO ready!")

        except Exception as e:
            print(f"❌ Critical error loading Fun-ASR OpenVINO: {e}")
            traceback.print_exc()

    def transcribe(self, audio_path):
        if not self.is_loaded:
            return "Error: Fun-ASR OpenVINO model not loaded."

        print(f"🔄 Transcribing with Fun-ASR-Nano (OpenVINO)...")
        try:
            res = self.model.inference(data_in=[audio_path])

            if isinstance(res, tuple) and len(res) > 0:
                out_list = res[0]
                if isinstance(out_list, list) and len(out_list) > 0 and 'text' in out_list[0]:
                    return out_list[0]['text'].strip()

            elif isinstance(res, list) and len(res) > 0:
                if isinstance(res[0], list) and len(res[0]) > 0 and 'text' in res[0][0]:
                    return res[0][0]['text'].strip()
                elif 'text' in res[0]:
                    return res[0]['text'].strip()
                elif isinstance(res[0], dict) and 'text' in res[0]:
                    return res[0]['text'].strip()

            return ""

        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            return ""

class STTModuleKyutai:
    """STT module using Kyutai's 1B parameter Moshi model architecture."""
    def __init__(self, model_repo="kyutai/stt-1b-en_fr"):
        self.device = "cpu"
        print(f"🎙️ Loading Kyutai STT (1B params) on {self.device.upper()}...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(model_repo)
        self.mimi = checkpoint_info.get_mimi(device=self.device)
        self.mimi.eval()
        self.text_tokenizer = checkpoint_info.get_text_tokenizer()
        self.lm = checkpoint_info.get_moshi(device=self.device)
        self.lm.eval()
        self.stt_config = checkpoint_info.stt_config
        from moshi.models import LMGen
        self.lm_gen = LMGen(self.lm, temp=0, temp_text=0, use_sampling=False)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    def transcribe(self, audio_path):
        import librosa
        import torch
        print(f"🔄 Processing audio...")
        try:
            audio_data, _ = librosa.load(audio_path, sr=24000, mono=True)
            in_pcms = torch.from_numpy(audio_data).unsqueeze(0).to(device=self.device)
            pad_left = int(self.stt_config.get("audio_silence_prefix_seconds", 0.0) * 24000)
            pad_right = int((self.stt_config.get("audio_delay_seconds", 0.0) + 1.0) * 24000)
            in_pcms = torch.nn.functional.pad(in_pcms, (pad_left, pad_right), mode="constant")
            in_pcms = in_pcms[None, 0:1]

            chunks = [c for c in in_pcms.split(self.frame_size, dim=2) if c.shape[-1] == self.frame_size]

            all_token_ids = []
            first_frame = True

            for chunk in chunks:
                codes = self.mimi.encode(chunk)
                if first_frame:
                    self.lm_gen.step(codes)
                    first_frame = False

                tokens = self.lm_gen.step(codes)
                if tokens is not None:
                    token_id = tokens[0, 0].item()
                    if token_id not in [0, 3]:
                        all_token_ids.append(token_id)

            final_text = self.text_tokenizer.decode(all_token_ids)

            return final_text.strip()

        except Exception as e:
            print(f"❌ Error during Kyutai transcription: {e}")
            return ""


class STTModuleFunASRNano:
    """High-speed STT module utilizing the FunASR Nano model."""
    def __init__(self, model_path="FunAudioLLM/Fun-ASR-Nano-2512", device="CPU"):
        print(f"🎙️ Initializing STT (Fun-ASR-Nano 2512) on {device}...")
        self.is_loaded = False

        try:
            from funasr import AutoModel

            use_gpu = device.upper() in ["GPU", "CUDA"]
            target_device = "cuda:0" if use_gpu else "cpu"

            self.model = AutoModel(
                model=model_path,
                device=target_device,
                disable_update=True,
                trust_remote_code = True
            )
            self.is_loaded = True
            print("✅ Fun-ASR-Nano loaded and ready to listen!")

        except ImportError:
            print("❌ Error: 'funasr' library not found. Install it from GitHub.")
        except Exception as e:
            print(f"❌ Critical error during Fun-ASR initialization:")
            traceback.print_exc()

    def transcribe(self, audio_path):
        if not self.is_loaded:
            return "Error: Fun-ASR model not loaded."

        print(f"🔄 Transcribing with Fun-ASR...")
        try:
            res = self.model.generate(input=audio_path, batch_size_s=300)

            if isinstance(res, list) and len(res) > 0 and 'text' in res[0]:
                transcribed_text = res[0]['text'].strip()
                print(f"   -> {transcribed_text[:50]}...")
                return transcribed_text
            else:
                print("   ⚠️ Fun-ASR did not detect any text.")
                return ""

        except Exception as e:
            print(f"❌ Error during Fun-ASR transcription:")
            traceback.print_exc()
            return ""

class STTModuleQwen3ASROpenVINO:
    """OpenVINO implementation of Qwen3 ASR for Intel hardware acceleration."""
    def __init__(self, model_path="Models/qwen3-asr-0.6b-ov", device="CPU"):
        print(f"🎙️ Initializing STT (Qwen3-ASR 0.6B OpenVINO) on {device}...")
        self.is_loaded = False

        helper_url = "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/qwen3-asr/qwen_3_asr_helper.py"
        helper_path = "qwen_3_asr_helper.py"

        if not os.path.exists(helper_path):
            print("📥 Downloading qwen_3_asr_helper.py module...")
            try:
                urllib.request.urlretrieve(helper_url, helper_path)
            except Exception as e:
                print(f"❌ Error downloading helper: {e}")
                return

        if not os.path.exists(model_path):
            print(f"❌ Error: Folder not found in {model_path}.")
            return

        try:
            from qwen_3_asr_helper import OVQwen3ASRModel

            print(f"🧠 Loading OpenVINO weights into memory on {device.upper()}...")
            self.model = OVQwen3ASRModel(
                model_dir=Path(model_path),
                device=device.upper(),
                max_inference_batch_size=1
            )
            self.is_loaded = True
            print("✅ Qwen3-ASR OpenVINO loaded and ready to use!")

        except Exception as e:
            print(f"❌ Critical error loading Qwen3-ASR: {e}")
            traceback.print_exc()

    def transcribe(self, audio_path):
        if not self.is_loaded:
            return "Error: Qwen3-ASR model not loaded."

        if not audio_path.lower().endswith('.wav'):
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            temp_wav_path = "temp_audio_converted.wav"
            audio.export(temp_wav_path, format="wav")
            audio_path = temp_wav_path

        print(f"🔄 Transcribing with Qwen3-ASR (OpenVINO)...")
        try:
            results = self.model.transcribe(audio=audio_path, language=None)

            transcribed_text = results[0].text.strip()
            detected_lang = results[0].language

            print(f"   [Detected Language: {detected_lang}] -> {transcribed_text[:50]}...")
            return transcribed_text

        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            return ""

class STTModuleMoonshine:
    """Lightweight ASR using Moonshine models for low latency."""
    def __init__(self, device="cpu", model_name="UsefulSensors/moonshine-tiny"):
        print(f"👂 Initializing Moonshine ASR ({model_name}) on {device}...")

        if MoonshineForConditionalGeneration is None:
            print("❌ Error: installed 'transformers' version does not support Moonshine.")
            self.is_loaded = False
            return

        self.is_loaded = False
        self.device = "cuda" if device.upper() == "GPU" and torch.cuda.is_available() else "cpu"

        try:
            print(f"   📥 Downloading/Loading weights for {model_name}...")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = MoonshineForConditionalGeneration.from_pretrained(model_name).to(self.device)

            self.is_loaded = True
            print(f"✅ Moonshine ({model_name}) loaded successfully on {self.device}!")
        except Exception as e:
            print(f"❌ Error loading Moonshine:")
            traceback.print_exc()

    def transcribe(self, audio_path):
        if not self.is_loaded:
            return ""

        try:
            print("🔄 Transcribing with Moonshine...")
            audio_array, sr = librosa.load(audio_path, sr=16000)

            inputs = self.processor(audio_array, return_tensors="pt", sampling_rate=16000)
            inputs = inputs.to(self.device)

            generated_ids = self.model.generate(**inputs)
            extracted_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)

            return extracted_text.strip()

        except Exception as e:
            print(f"❌ Error during Moonshine transcription: {e}")
            return ""


class STTModuleWhisperMedium:
    """Whisper Medium model wrapper utilizing OpenVINO backend for efficient inference."""
    def __init__(self, local_model_path="Models/whisper-medium-ov", device="CPU"):
        print(f"🎙️ Initializing STT (Whisper Medium OpenVINO) on {device}...")
        if not os.path.exists(local_model_path):
            print("📥 Downloading and converting from PyTorch to OpenVINO (first time only)...")
            model_id = "openai/whisper-medium"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, device=device)
            os.makedirs(local_model_path, exist_ok=True)
            self.model.save_pretrained(local_model_path)
            self.processor.save_pretrained(local_model_path)
            print(f"✅ Medium OpenVINO model saved in: {local_model_path}")
        else:
            print(f"⚡ Loading local model from: {local_model_path}")
            self.processor = AutoProcessor.from_pretrained(local_model_path)
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(local_model_path, device=device)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            device="cpu"
        )

    def transcribe(self, audio_path):
        if not audio_path.lower().endswith('.wav'):
            audio = AudioSegment.from_file(audio_path)
            temp_wav_path = "temp_audio_converted.wav"
            audio.export(temp_wav_path, format="wav")
            audio_path = temp_wav_path

        print("🔄 Processing audio Whisper Medium (with chunking)...")
        result = self.pipe(audio_path, generate_kwargs={"task": "transcribe", "language": "en"})
        return result["text"].strip()


class STTModuleWhisperV3Turbo:
    """Latest generation Whisper V3 Turbo wrapped with OpenVINO export."""
    def __init__(self, local_model_path="Models/whisper-v3-turbo-ov", device="CPU"):
        print(f"🎙️ Initializing STT (Whisper V3 Turbo OpenVINO) on {device}...")
        if not os.path.exists(local_model_path):
            print("📥 Downloading and converting from PyTorch to OpenVINO (first time only)...")
            model_id = "openai/whisper-large-v3-turbo"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, device=device)
            os.makedirs(local_model_path, exist_ok=True)
            self.model.save_pretrained(local_model_path)
            self.processor.save_pretrained(local_model_path)
            print(f"✅ V3 Turbo OpenVINO model saved in: {local_model_path}")
        else:
            print(f"⚡ Loading local model from: {local_model_path}")
            self.processor = AutoProcessor.from_pretrained(local_model_path)
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(local_model_path, device=device)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            device="cpu"
        )

    def transcribe(self, audio_path):
        if not audio_path.lower().endswith('.wav'):
            audio = AudioSegment.from_file(audio_path)
            temp_wav_path = "temp_audio_converted.wav"
            audio.export(temp_wav_path, format="wav")
            audio_path = temp_wav_path

        print("🔄 Processing audio Whisper V3 Turbo (with chunking)...")
        result = self.pipe(audio_path, generate_kwargs={"task": "transcribe", "language": "en"})
        return result["text"].strip()

class STTModuleWhisperTiny:
    """Lightweight Whisper Tiny deployment via OpenVINO for fast CPU/GPU inference."""
    def __init__(self, local_model_path="Models/whisper-tiny-ov", device="CPU"):
        print(f"🎙️ Initializing STT (Whisper Tiny OpenVINO) on {device}...")
        if not os.path.exists(local_model_path):
            print("📥 Downloading and converting from PyTorch to OpenVINO (first time only)...")
            model_id = "openai/whisper-tiny"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, device=device)
            os.makedirs(local_model_path, exist_ok=True)
            self.model.save_pretrained(local_model_path)
            self.processor.save_pretrained(local_model_path)
            print(f"✅ Tiny OpenVINO model saved in: {local_model_path}")
        else:
            print(f"⚡ Loading local model from: {local_model_path}")
            self.processor = AutoProcessor.from_pretrained(local_model_path)
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(local_model_path, device=device)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            device="cpu"
        )

    def transcribe(self, audio_path):
        if not audio_path.lower().endswith('.wav'):
            audio = AudioSegment.from_file(audio_path)
            temp_wav_path = "temp_audio_converted.wav"
            audio.export(temp_wav_path, format="wav")
            audio_path = temp_wav_path
        print("🔄 Processing audio Whisper Tiny (with chunking)...")
        result = self.pipe(audio_path, generate_kwargs={"task": "transcribe", "language": "en"})
        return result["text"].strip()


class STTModuleDistilWhisperOpenVINO:
    """Knowledge-distilled version of Whisper designed for high-speed edge deployments."""
    def __init__(self, local_model_path="Models/distil-whisper-small-en-ov", device="CPU"):
        print(f"🎙️ Initializing STT (Distil-Whisper OpenVINO) on {device}...")

        if not os.path.exists(local_model_path):
            print("📥 Downloading and converting Distil-Whisper to OpenVINO...")
            model_id = "distil-whisper/distil-small.en"

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, device=device)

            os.makedirs(local_model_path, exist_ok=True)
            self.model.save_pretrained(local_model_path)
            self.processor.save_pretrained(local_model_path)
            print(f"✅ Distil-Whisper OpenVINO model saved in: {local_model_path}")
        else:
            print(f"⚡ Loading local model from: {local_model_path}")
            self.processor = AutoProcessor.from_pretrained(local_model_path)
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(local_model_path, device=device)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=15,
            device="cpu"
        )

    def transcribe(self, audio_path):
        if not audio_path.lower().endswith('.wav'):
            audio = AudioSegment.from_file(audio_path)
            temp_wav_path = "temp_audio_converted.wav"
            audio.export(temp_wav_path, format="wav")
            audio_path = temp_wav_path

        print("🔄 Processing audio Distil-Whisper (with 15s chunking)...")

        result = self.pipe(audio_path)

        return result["text"].strip()

class STTModuleWhisperBase:
    """Standard Whisper Base architecture exported for OpenVINO inference."""
    def __init__(self, local_model_path="Models/whisper-base-ov", device="CPU"):
        print(f"🎙️ Initializing STT (Whisper Base OpenVINO) on {device}...")
        if not os.path.exists(local_model_path):
            print("📥 Downloading and converting from PyTorch to OpenVINO...")
            model_id = "openai/whisper-base"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, device=device)
            os.makedirs(local_model_path, exist_ok=True)
            self.model.save_pretrained(local_model_path)
            self.processor.save_pretrained(local_model_path)
        else:
            print(f"⚡ Loading local model from: {local_model_path}")
            self.processor = AutoProcessor.from_pretrained(local_model_path)
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(local_model_path, device=device)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            device="cpu"
        )

    def transcribe(self, audio_path):
        if not audio_path.lower().endswith('.wav'):
            audio = AudioSegment.from_file(audio_path)
            temp_wav_path = "temp_audio_converted.wav"
            audio.export(temp_wav_path, format="wav")
            audio_path = temp_wav_path
        print("🔄 Processing audio (with chunking for long files)...")
        result = self.pipe(audio_path, generate_kwargs={"task": "transcribe", "language": "en"})
        return result["text"].strip()


class STTModuleDistilWhisperLargeOpenVINO:
    """Multilingual high-accuracy distilled Whisper Large model."""
    def __init__(self, local_model_path="Models/distil-whisper-large-v3-5-ov", device="CPU"):
        print(f"🎙️ Initializing STT (Distil-Whisper Large v3.5 OpenVINO) on {device}...")

        if not os.path.exists(local_model_path):
            print("📥 Downloading and converting Distil-Whisper Large to OpenVINO (this will take a while)...")
            model_id = "distil-whisper/distil-large-v3.5"

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, device=device)

            os.makedirs(local_model_path, exist_ok=True)
            self.model.save_pretrained(local_model_path)
            self.processor.save_pretrained(local_model_path)
            print(f"✅ Distil-Whisper Large OpenVINO model saved in: {local_model_path}")
        else:
            print(f"⚡ Loading local model from: {local_model_path}")
            self.processor = AutoProcessor.from_pretrained(local_model_path)
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(local_model_path, device=device)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            device="cpu"
        )

    def transcribe(self, audio_path):
        if not audio_path.lower().endswith('.wav'):
            audio = AudioSegment.from_file(audio_path)
            temp_wav_path = "temp_audio_converted.wav"
            audio.export(temp_wav_path, format="wav")
            audio_path = temp_wav_path

        print("🔄 Processing audio Distil-Whisper Large v3.5...")

        result = self.pipe(audio_path, generate_kwargs={"task": "transcribe", "language": "en"})

        return result["text"].strip()

class STTModuleWhisperSmall:
    """Standard Whisper Small model mapped for hybrid CPU/GPU architectures."""
    def __init__(self, model_path, device="AUTO:GPU,CPU"):
        print(f"🎙️ Initializing STT (Whisper Small OpenVINO) on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = OVModelForSpeechSeq2Seq.from_pretrained(model_path, device=device)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            device="cpu"
        )

    def transcribe(self, audio_path):
        if not audio_path.lower().endswith('.wav'):
            audio = AudioSegment.from_file(audio_path)
            temp_wav_path = "temp_audio_converted.wav"
            audio.export(temp_wav_path, format="wav")
            audio_path = temp_wav_path
        print("🔄 Processing audio Whisper Small (with chunking)...")
        result = self.pipe(audio_path, generate_kwargs={"task": "transcribe", "language": "en"})
        return result["text"].strip()


class LLMModuleSmolLM2:
    """Lightweight text generation module utilizing SmolLM2 360M parameter model."""
    def __init__(self, model_id="HuggingFaceTB/SmolLM2-360M-Instruct", device="cpu"):
        self.device = device
        print(f"🧠 Initializing LLM (SmolLM2 360M) on {self.device.upper()}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()

    def generate(self, prompt, out_txt_path):
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        t_start = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
        t_end = time.perf_counter()

        input_length = inputs["input_ids"].shape[1]
        output_ids = generated_ids[0][input_length:]
        generated_tokens = len(output_ids)
        generation_time = t_end - t_start
        tps = generated_tokens / generation_time if generation_time > 0 else 0

        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"🚀 SmolLM2 Speed: {tps:.2f} tok/s | Time: {generation_time:.2f}s | Tokens: {generated_tokens}")
        return content, generated_tokens


class LLMModuleGemma:
    """Gemma 3 text generation module for hardware accelerated environments."""
    def __init__(self, model_id="google/gemma-3-270m-it", device="gpu"):
        self.device = device
        hf_token = "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"
        print(f"🧠 Initializing LLM (Gemma 3 270M) on {self.device.upper()}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            token=hf_token
        ).to(self.device)
        self.model.eval()

    def generate(self, prompt, out_txt_path):
        text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        end_of_turn_id = self.tokenizer.encode("<end_of_turn>", add_special_tokens=False)
        terminators = [self.tokenizer.eos_token_id]
        if end_of_turn_id:
            terminators.append(end_of_turn_id[0])

        t_start = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
        t_end = time.perf_counter()

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        generated_tokens = len(output_ids)
        generation_time = t_end - t_start
        tps = generated_tokens / generation_time if generation_time > 0 else 0

        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"🚀 Gemma 3 Speed: {tps:.2f} tok/s | Time: {generation_time:.2f}s | Tokens: {generated_tokens}")
        return content, generated_tokens


class LLMModuleQwenNPU:
    """Official Qwen 2.5 architecture implementation specifically routed to NPU compute."""
    def __init__(self, model_id="OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov", device="NPU"):
        print(f"\n🧠 Initializing NATIVE LLM (Qwen 2.5 Instruct Official) on {device}...")
        self.local_model_path = "Qwen2.5-1.5B-Instruct-int4-ov"
        print("📥 Download/Verify model from official repository...")
        hf_hub.snapshot_download(model_id, local_dir=self.local_model_path)
        print(f"⏳ Initializing LLMPipeline on {device}...")
        self.pipe = ov_genai.LLMPipeline(self.local_model_path, device)
        print("🧮 Loading exact Tokenizer for precise metrics...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)

        print("✅ Qwen 2.5 ready on the NPU!")

    def generate(self, text, output_path=None):
        print(f"🧠 Qwen 2.5 (NPU) is generating...")
        t0 = time.perf_counter()
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful, brief, and concise AI assistant.<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        response = self.pipe.generate(prompt, max_new_tokens=150)
        t1 = time.perf_counter()

        response_str = str(response)
        if "<|im_start|>assistant" in response_str:
            response_clean = response_str.split("<|im_start|>assistant")[-1]
        else:
            response_clean = response_str
        response_clean = response_clean.replace("<|im_end|>", "").strip()

        latency = t1 - t0
        num_tokens = len(self.tokenizer.encode(response_clean))

        print(f"🤖 Response: {response_clean[:50]}...")
        print(f"🧮 Exact Tokens generated: {num_tokens}")

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response_clean)

        return response_clean, num_tokens


class LLMModuleDeepSeekNPU:
    """DeepSeek R1 Distill execution wrapped with OpenVINO GenAI for NPU offloading."""
    def __init__(self, model_id="OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov", device="NPU"):
        import huggingface_hub as hf_hub
        import openvino_genai as ov_genai
        print(f"\n🧠 Initializing NATIVE LLM (Official Repo Recipe) on {device}...")
        self.local_model_path = "DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov"
        print("📥 Download/Verify model from official repository...")
        hf_hub.snapshot_download(model_id, local_dir=self.local_model_path)
        print(f"⏳ Initializing LLMPipeline on {device}...")
        self.pipe = ov_genai.LLMPipeline(self.local_model_path, device)
        print("✅ DeepSeek ready on the NPU!")

    def generate(self, text, output_path=None):
        print(f"🧠 DeepSeek (NPU) is generating...")
        t0 = time.perf_counter()
        response = self.pipe.generate(text, max_length=5000)
        t1 = time.perf_counter()
        response_clean = str(response).strip()
        if "</think>" in response_clean:
            response_clean = response_clean.split("</think>")[-1].strip()
        response_clean = response_clean.replace("<think>", "").strip()
        if not response_clean:
            response_clean = "Sorry, I was thinking too long and lost my train of thought."

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response_clean)

        num_tokens = len(response_clean.split())
        print(f"🤖 Response: {response_clean}")
        return response_clean, num_tokens


class LLMModuleQwen:
    """Baseline Qwen text generation model operating via OpenVINO backend tensors."""
    def __init__(self, model_path, device="NPU"):
        print(f"🧠 Initializing LLM (Qwen) on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
        self.model = OVModelForCausalLM.from_pretrained(model_path, device=device)
        self.device = self.model.device

    def generate(self, prompt, out_txt_path):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        generated_tokens = len(generated_ids[0]) - len(model_inputs.input_ids[0])

        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(content)

        return content, generated_tokens


class TTSModuleKittenNano:
    """Microscopic footprint TTS engine primarily suitable for CPU-only inference."""
    def __init__(self, model_type="kitten"):
        self.model_type = model_type
        if self.model_type == "kitten":
            print("🔊 Initializing TTS (KittenTTS) on CPU...")
            self.model = KittenTTS("KittenML/kitten-tts-nano-0.2")

    def synthesize(self, text, out_wav_path):
        if self.model_type == "kitten":
            try:
                clean_text = text.replace("*", "").replace("#", "").strip()
                full_audio = self.model.generate(clean_text, voice='expr-voice-2-f', speed = 1.2)

                sf.write(out_wav_path, full_audio, 24000)
                return True

            except Exception as e:
                print(f"Error on KittenTTS synthesis: {e}")
                return False


class OVKModel(KModel):
    """Custom OpenVINO integration class enforcing NPU Static Shapes for Kokoro TTS."""
    def __init__(self, model_dir, device):
        super().__init__()
        self._device_name = device.upper()
        with open(Path(model_dir) / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        self.vocab = config["vocab"]
        self.context_length = config.get("plbert", {}).get("max_position_embeddings", 512)
        core = ov.Core()
        ov_model = core.read_model(Path(model_dir) / "openvino_model.xml")
        ov_config = {}

        if "NPU" in self._device_name:
            print("🧱 Forcing Static Shapes for the NPU...")
            ov_model.reshape({"input_ids": [1, 512], "ref_s": [1, 256], "speed": [1]})
            ov_config = {"NPU_USE_NPUW": "YES", "NPUW_DEVICES": "NPU,CPU", "NPUW_KOKORO": "YES"}
        elif "GPU" in self._device_name:
            ov_config = {"INFERENCE_PRECISION_HINT": ov.Type.f32}

        self.model = core.compile_model(ov_model, self._device_name, ov_config)

    @property
    def device(self):
        return torch.device("cpu")

    def forward_with_tokens(self, input_ids, ref_s, speed=1):
        text_len = input_ids.shape[-1]
        if "NPU" in self._device_name and text_len < self.context_length:
            input_ids = torch.nn.functional.pad(input_ids, (0, self.context_length - text_len), value=16)

        outputs = self.model([input_ids, ref_s, torch.tensor(speed)])
        audio = torch.from_numpy(outputs[0]).squeeze()
        pred_dur = torch.from_numpy(outputs[1]).squeeze()

        if "NPU" in self._device_name and text_len < self.context_length:
            total_frames = torch.sum(pred_dur).item()
            real_frames = torch.sum(pred_dur[:text_len]).item()
            if total_frames > 0:
                hop_length = len(audio) / total_frames
                valid_audio_samples = int(real_frames * hop_length)
                audio = audio[:valid_audio_samples]
            pred_dur = pred_dur[:text_len]

        return audio, pred_dur


class TTSModuleKokoroOV:
    """Pipeline wrapper for executing Kokoro TTS across OpenVINO-compatible hardware."""
    def __init__(self, model_dir=PATH_KOKORO_OV, device="NPU", voice="af_heart"):
        print(f"\n🔊 Initializing Kokoro TTS (OpenVINO) on {device}...")
        try:
            ov_model = OVKModel(model_dir=model_dir, device=device)
            self.pipeline = KPipeline(lang_code="a", model=ov_model, repo_id="hexgrad/Kokoro-82M")
            self.voice = voice
            print(f"✅ Kokoro OV ready on {device}!")
        except Exception as e:
            print(f"❌ Error initializing TTS: {e}")

    def synthesize(self, text, out_wav_path):
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            return False
        print(f"🔊 Kokoro OV is generating audio ({self.pipeline.model._device_name})...")
        t_start = time.perf_counter()
        clean_text = re.sub(r'[*#_~`]', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        if not clean_text:
            return False

        try:
            generator = self.pipeline(clean_text, voice=self.voice, speed=1.0, split_pattern=r'\n+')
            audio_pieces = []
            for _, _, audio in generator:
                if audio is not None:
                    audio_pieces.append(audio)
                    audio_pieces.append(np.zeros(int(24000 * 0.1), dtype=np.float32))

            if audio_pieces:
                full_audio = np.concatenate(audio_pieces)
                sf.write(out_wav_path, full_audio, 24000)
                t_end = time.perf_counter()
                print(f"✅ Audio generated in {t_end - t_start:.2f}s (Clean tails!)")
                return True
            return False
        except Exception as e:
            print(f"❌ Synthesis error: {e}")
            return False


class TTSModuleQwen3OpenVINO:
    """Robust Qwen3 Text-to-Speech deployment leveraging Intel hardware extensions."""
    def __init__(self, model_path="Models/qwen3-tts-0.6b-ov", device="CPU", speaker="Aiden"):
        print(f"🎙️ Initializing TTS (Qwen3-TTS 0.6B OV) on {device}...")
        self.is_loaded = False
        self.speaker = speaker

        helper_url = "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/qwen3-tts/qwen_3_tts_helper.py"
        helper_path = "qwen_3_tts_helper.py"
        if not os.path.exists(helper_path):
            try:
                urllib.request.urlretrieve(helper_url, helper_path)
            except Exception as e:
                print(f"❌ Error downloading helper: {e}")
                return

        if not os.path.exists(model_path):
            print(f"❌ Error: Folder {model_path} not found.")
            print("👉 Run the Colab conversion first!")
            return

        try:
            from qwen_3_tts_helper import OVQwen3TTSModel

            print(f"🧠 Loading weights on {device.upper()}...")
            self.model = OVQwen3TTSModel.from_pretrained(
                model_dir=Path(model_path),
                device=device.upper()
            )
            self.is_loaded = True
            print("✅ Qwen3-TTS OpenVINO loaded and ready to speak!")

        except Exception as e:
            print(f"❌ Loading error: {e}")
            traceback.print_exc()

    def synthesize(self, text, output_path):
        if not self.is_loaded:
            return False

        print("🔄 Voice synthesis with Qwen3-TTS in progress...")
        try:
            result = self.model.generate_custom_voice(
                text=text,
                language="English",
                speaker=self.speaker
            )

            if isinstance(result, tuple):
                audio_data = result[0]
                sr = result[1]
            else:
                audio_data = result
                sr = 24000

            import numpy as np

            if hasattr(audio_data, 'cpu'):
                audio_data = audio_data.cpu().numpy()
            elif hasattr(audio_data, 'data'):
                audio_data = np.array(audio_data.data)

            audio_array = np.array(audio_data).squeeze()

            import soundfile as sf
            sf.write(output_path, audio_array, sr)

            print("✅ Audio saved successfully!")
            return True

        except Exception as e:
            print(f"❌ Synthesis error: {e}")
            import traceback
            traceback.print_exc()
            return False

class TTSModulePocket:
    """Local, extremely low latency TTS utilizing Kyutai's audio representation structure."""
    def __init__(self, device="cpu", voice="alba"):
        print(f"🔊 Initializing Pocket-TTS (Kyutai) on {device}...")

        if TTSModel is None:
            print("❌ Error: 'pocket-tts' library not found. Run 'pip install pocket-tts'.")
            self.is_loaded = False
            return

        self.is_loaded = False
        try:
            self.model = TTSModel.load_model()

            print(f"   Loading voice '{voice}' into memory...")
            self.voice_state = self.model.get_state_for_audio_prompt(voice)

            self.is_loaded = True
            print("✅ Pocket-TTS loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Pocket-TTS: {e}")

    def synthesize(self, text, output_path):
        if not self.is_loaded:
            print("❌ Cannot synthesize: Pocket-TTS is not initialized.")
            return False

        try:
            audio_tensor = self.model.generate_audio(self.voice_state, text)

            audio_np = audio_tensor.numpy()
            scipy.io.wavfile.write(output_path, self.model.sample_rate, audio_np)

            return True
        except Exception as e:
            print(f"❌ Error during Pocket-TTS synthesis: {e}")
            return False


class TTSModuleOuteTTS_1B:
    """Text-to-Speech generation via LLM behavior utilizing the Llama.cpp hardware backend."""
    def __init__(self, device="cpu"):
        print(f"🔊 Initializing OuteTTS 1.0 (1B) via Llama CPP...")

        if Interface is None:
            print("❌ Error: 'outetts' library not found.")
            self.is_loaded = False
            return

        self.is_loaded = False

        try:
            print("   🧠 Configuring Llama 1B architecture (Q4_K_M Quantization)...")

            config = ModelConfig.auto_config(
                model=Models.VERSION_1_0_SIZE_1B,
                backend=Backend.LLAMACPP,
                quantization=LlamaCppQuantization.Q4_K_M
            )

            self.interface = Interface(config=config)

            print("   🗣️ Loading default voice profile...")
            self.speaker = self.interface.load_default_speaker("en-female-1-neutral")

            self.is_loaded = True
            print(f"✅ OuteTTS 1.0 loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading OuteTTS 1.0: {e}")
            traceback.print_exc()

    def synthesize(self, text, output_path):
        if not self.is_loaded:
            return False

        try:
            print(f"🔄 Llama synthesis in progress: '{text[:30]}...'")

            output = self.interface.generate(
                config=GenerationConfig(
                    text=text,
                    generation_type=GenerationType.CHUNKED,
                    speaker=self.speaker,
                    sampler_config=SamplerConfig(
                        temperature=0.4
                    )
                )
            )

            print("   💾 Saving audio file...")
            import soundfile as sf

            audio_array = output.audio.cpu().numpy().squeeze()

            sf.write(output_path, audio_array, output.sr)

            print("✅ Audio saved successfully!")
            return True

        except Exception as e:
            print(f"❌ Error during OuteTTS 1.0 synthesis: {e}")
            return False

class TTSModuleVoxCPM:
    """Diffusion-based acoustic model that hallucinates natural voices and prosody contextually."""
    def __init__(self, model_path="openbmb/VoxCPM-0.5B", device="CPU"):
        print(f"🎙️ Initializing TTS (VoxCPM-0.5B Diffusion) on {device}...")
        self.is_loaded = False

        import torch
        import traceback

        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        try:
            from voxcpm import VoxCPM
            print("   📥 Loading VoxCPM model...")

            self.model = VoxCPM.from_pretrained(model_path)
            self.is_loaded = True
            print("✅ VoxCPM loaded and ready (Contextual Improvisation Mode)!")

        except ImportError:
            print("❌ Error: voxcpm library not found. Run: pip install voxcpm")
        except Exception as e:
            print(f"❌ Critical error during VoxCPM initialization: {e}")
            traceback.print_exc()

    def synthesize(self, text, output_path):
        if not self.is_loaded:
            return False

        print(f"🔄 Voice synthesis with VoxCPM (Diffusion) in progress...")
        print("   ⚠️ Warning: Diffusion models in pure PyTorch can be very slow...")
        try:
            wav = self.model.generate(
                text=text,
                prompt_wav_path=None,
                prompt_text=None,
                cfg_value=2.0,
                inference_timesteps=10,
                normalize=True,
                denoise=False
            )

            audio_array = np.array(wav).squeeze()

            sf.write(output_path, audio_array, 16000)
            print("✅ Audio saved successfully!")
            return True

        except Exception as e:
            print(f"❌ Error during VoxCPM synthesis:")
            traceback.print_exc()
            return False

class TTSModuleKittenMini:
    """Scalable and flexible TTS handler executing small payload models rapidly."""
    def __init__(self, device="cpu", voice="Jasper"):
        print(f"🔊 Initializing Kitten-TTS Mini 0.8 on {device}...")

        if KittenTTS is None:
            print("❌ Error: 'kittentts' library not found.")
            self.is_loaded = False
            return

        self.is_loaded = False
        self.voice = voice
        try:
            self.model = KittenTTS("kitten-tts-mini-0.8")

            self.is_loaded = True
            print(f"✅ Kitten-TTS loaded successfully (Voice: {self.voice})!")
        except Exception as e:
            print(f"❌ Error loading Kitten-TTS: {e}")

    def synthesize(self, text, output_path):
        if not self.is_loaded:
            return False
        try:
            audio_array = self.model.generate(text, voice=self.voice)
            sf.write(output_path, audio_array, 24000)
            return True
        except Exception as e:
            print(f"❌ Error during Kitten-TTS synthesis: {e}")
            return False

class TTSModuleSoprano:
    """Streamlined low parameter acoustic generation engine targeted at local compute nodes."""
    def __init__(self, device="cpu"):
        print(f"🔊 Initializing Soprano-1.1-80M on {device}...")
        safe_device = "cpu" if device.upper() not in ["CUDA", "MPS"] else device.lower()

        try:
            self.model = SopranoTTS(
                backend='auto',
                device= 'auto',
                cache_size_mb=100,
                decoder_batch_size=1
            )
        except Exception as e:
            print(f"❌ Error loading Soprano: {e}")

    def synthesize(self, text, output_path):
        try:
            self.model.infer(text, output_path)
            return True
        except Exception as e:
            print(f"❌ Error during Soprano synthesis: {e}")
            return False


class TTSModuleSupertonic:
    """ONNX Runtime accelerated voice model supporting rapid native stylization features."""
    def __init__(self, device="cpu", voice_name="M1"):
        print(f"🔊 Initializing Supertonic-2 on {device}...")

        if TTS is None:
            print("❌ Error: 'supertonic' library not found. Run 'pip install supertonic'.")
            self.is_loaded = False
            return

        self.is_loaded = False
        try:
            self.model = TTS(auto_download=True)

            print(f"   Loading voice style '{voice_name}'...")
            self.voice_style = self.model.get_voice_style(voice_name=voice_name)

            self.is_loaded = True
            print("✅ Supertonic-2 loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Supertonic: {e}")

    def synthesize(self, text, output_path):
        if not self.is_loaded:
            print("❌ Cannot synthesize: Supertonic is not initialized.")
            return False

        try:
            audio_data, duration = self.model.synthesize(text, voice_style=self.voice_style, lang="en")

            self.model.save_audio(audio_data, output_path)

            return True
        except Exception as e:
            print(f"❌ Error during Supertonic synthesis: {e}")
            return False

class TTSModuleKokoro:
    """Standard CPU execution backend for Kokoro 82M bridging Python chunk logic."""
    def __init__(self, voice='af_heart'):
        print(f"🔊 Initializing Kokoro TTS (Voice: {voice}, 82M params) on CPU...")
        try:
            self.pipeline = KPipeline(lang_code='a')
            self.voice = voice
        except Exception as e:
            print(f"❌ Failed to load Kokoro TTS: {e}")

    def synthesize(self, text, out_wav_path):
        clean_text = re.sub(r'[*#_~`]', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        if not clean_text:
            return False
        try:
            generator = self.pipeline(clean_text, voice=self.voice, speed=1.1, split_pattern=r'\n+')
            audio_pieces = []
            trim_samples = int(24000 * 0.15)
            for _, _, audio in generator:
                if audio is not None and len(audio) > trim_samples:
                    clean_chunk = audio[:-trim_samples]
                    audio_pieces.append(clean_chunk)
                    audio_pieces.append(np.zeros(int(24000 * 0.1), dtype=np.float32))

            if audio_pieces:
                full_audio = np.concatenate(audio_pieces)
                sf.write(out_wav_path, full_audio, 24000)
                return True
            return False
        except Exception as e:
            print(f"❌ Kokoro Synthesis Error: {e}")
            return False


class TTSModulePiper:
    """Legacy VITS-based acoustic backend utilizing ONNX execution graphs directly."""
    def __init__(self, model_path):
        self.model_path = os.path.abspath(model_path)
        print(f"🔊 Initializing Piper TTS (Python API) - Voice: Lessac (English)")
        if not os.path.exists(self.model_path):
            print(f"❌ CRITICAL ERROR: .onnx model not found at {self.model_path}")
        try:
            self.voice = PiperVoice.load(self.model_path)
        except Exception as e:
            print(f"❌ Failed to load Piper model: {e}")

    def synthesize(self, text, out_wav_path):
        clean_text = re.sub(r'[*#_~`]', '', text)
        clean_text = re.sub(r'\n+', '. ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        if not clean_text:
            return False
        try:
            with wave.open(out_wav_path, "wb") as wav_file:
                self.voice.synthesize_wav(clean_text, wav_file)
            return True
        except Exception as e:
            print(f"❌ Piper Synthesis Error: {e}")
            return False


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


# ==========================================
# 🚀 INTERACTIVE CLI & MODEL REGISTRIES
# ==========================================

AVAILABLE_STT = {
    "1": {"name": "Whisper Tiny OV", "class": STTModuleWhisperTiny, "kwargs": {"local_model_path": PATH_WHISPER_TINY},
          "devices": ["CPU", "GPU"], "pass_device": True},
    "2": {"name": "Whisper Base OV", "class": STTModuleWhisperBase, "kwargs": {"local_model_path": PATH_WHISPER_BASE},
          "devices": ["CPU", "GPU"], "pass_device": True},
    "3": {"name": "Whisper Small OV", "class": STTModuleWhisperSmall, "kwargs": {"model_path": PATH_WHISPER_SMALL},
          "devices": ["CPU", "GPU"], "pass_device": True},
    "4": {"name": "Kyutai STT", "class": STTModuleKyutai, "kwargs": {}, "devices": ["CPU"],
          "pass_device": False},
    "5": {"name": "Moonshine Tiny (27M)", "class": STTModuleMoonshine, "kwargs": {"model_name": "UsefulSensors/moonshine-tiny"},
          "devices": ["CPU"], "pass_device": True},
    "6": {"name": "Moonshine Base (61M)", "class": STTModuleMoonshine, "kwargs": {"model_name": "UsefulSensors/moonshine-base"},
          "devices": ["CPU"], "pass_device": True},
    "7": {"name": "Whisper Medium (OpenVINO)", "class": STTModuleWhisperMedium, "kwargs": {"local_model_path": PATH_WHISPER_MEDIUM}, "devices": ["CPU", "GPU"],
          "pass_device": True},
    "8": {"name": "Whisper V3 Turbo (OpenVINO)", "class": STTModuleWhisperV3Turbo, "kwargs": {"local_model_path": PATH_WHISPER_V3_TURBO},
          "devices": ["CPU", "GPU"], "pass_device": True},
    "9": {"name": "Qwen3-ASR 0.6B OV", "class": STTModuleQwen3ASROpenVINO, "kwargs": {"model_path": PATH_QWEN3_ASR_OV},
          "devices": ["CPU", "GPU"], "pass_device": True},
    "10": {"name": "Fun-ASR-Nano", "class": STTModuleFunASRNano, "kwargs": {"model_path": "FunAudioLLM/Fun-ASR-Nano-2512"},
           "devices": ["CPU"], "pass_device": True},
    "11": {"name": "Distil-Whisper Small OpenVINO", "class": STTModuleDistilWhisperOpenVINO, "kwargs": {"local_model_path": PATH_WHISPER_SMALL_DISTILL},
           "devices": ["CPU", "GPU"], "pass_device": True},
    "12": {"name": "Distil-Whisper Large v3.5 OpenVINO", "class": STTModuleDistilWhisperLargeOpenVINO, "kwargs": {"local_model_path": "Models/distil-whisper-large-v3-5-ov"},
           "devices": ["CPU", "GPU"], "pass_device": True},
    "13": {"name": "Fun-ASR-Nano OpenVINO", "class": STTModuleFunASRNanoOpenVINO, "kwargs": {"model_path": FUN_ASR_PATH_OV},
           "devices": ["CPU", "GPU"], "pass_device": True}
}

AVAILABLE_LLM = {
    "1": {"name": "Qwen 2.5 1.5B Instruct", "class": LLMModuleQwenNPU,
          "kwargs": {"model_id": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"}, "devices": ["CPU", "GPU", "NPU"],
          "pass_device": True},
    "2": {"name": "DeepSeek R1 1.5B Distill", "class": LLMModuleDeepSeekNPU,
          "kwargs": {"model_id": "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov"}, "devices": ["CPU", "GPU", "NPU"],
          "pass_device": True},
    "3": {"name": "Qwen OV", "class": LLMModuleQwen, "kwargs": {"model_path": PATH_QWEN},
          "devices": ["CPU", "GPU", "NPU"], "pass_device": True},
    "4": {"name": "SmolLM2 360M", "class": LLMModuleSmolLM2, "kwargs": {}, "devices": ["CPU"], "pass_device": True},
    "5": {"name": "Gemma 3 270M", "class": LLMModuleGemma, "kwargs": {}, "devices": ["CPU"], "pass_device": True}
}

AVAILABLE_TTS = {
    "1": {"name": "Kokoro OV", "class": TTSModuleKokoroOV, "kwargs": {"model_dir": PATH_KOKORO_OV, "voice": "af_heart"},
          "devices": ["CPU", "GPU", "NPU"], "pass_device": True},
    "2": {"name": "Kokoro Native", "class": TTSModuleKokoro, "kwargs": {"voice": "af_heart"}, "devices": ["CPU"],
          "pass_device": False},
    "3": {"name": "Piper", "class": TTSModulePiper, "kwargs": {"model_path": PATH_TTS}, "devices": ["CPU"],
          "pass_device": False},
    "4": {"name": "KittenTTS Nano", "class": TTSModuleKittenNano, "kwargs": {"model_type": "kitten"}, "devices": ["CPU"],
          "pass_device": False},
    "5": {"name": "Soprano 1.1 80M", "class": TTSModuleSoprano, "kwargs": {}, "devices": ["CPU"],
          "pass_device": True},
    "6": {"name": "Pocket-TTS (Kyutai)", "class": TTSModulePocket, "kwargs": {"voice": "alba"}, "devices": ["CPU"],
          "pass_device": False},
    "7": {"name": "Supertonic-2 (ONNX)", "class": TTSModuleSupertonic, "kwargs": {"voice_name": "M1"}, "devices": ["CPU"],
          "pass_device": False},
    "8": {"name": "Kitten-TTS Mini 0.8 (ONNX)", "class": TTSModuleKittenMini, "kwargs": {"voice": "Jasper"}, "devices": ["CPU"],
          "pass_device": False},
    "9": {"name": "OuteTTS 0.1 (350M LLaMa)", "class": TTSModuleOuteTTS_1B, "kwargs": {}, "devices": ["CPU"],
          "pass_device": True},
    "10": {"name": "Qwen3-TTS 0.6B OV", "class": TTSModuleQwen3OpenVINO, "kwargs": {"model_path": PATH_QWEN3_TTS_OV}, "devices": ["CPU", "GPU"],
          "pass_device": True},
    "11": {"name": "VoxCPM 0.5B (Diffusion)", "class": TTSModuleVoxCPM, "kwargs": {"model_path": "openbmb/VoxCPM-0.5B"}, "devices": ["CPU"],
          "pass_device": True}
}


def select_model_and_device(title, options):
    print(f"\n--- {title} ---")
    for key, val in options.items():
        devices_str = ", ".join(val["devices"])
        print(f"[{key}] {val['name']} (Supported hardware: {devices_str})")

    choice = input("Select an option: ")
    while choice not in options:
        choice = input("Invalid choice. Try again: ")

    selected = options[choice]
    devices = selected["devices"]
    chosen_device = devices[0]

    if len(devices) > 1:
        print(f"\nOn which hardware do you want to run {selected['name']}?")
        for i, dev in enumerate(devices):
            print(f"[{i + 1}] {dev}")
        dev_choice = input("Select device (e.g. 1): ")
        while not dev_choice.isdigit() or int(dev_choice) < 1 or int(dev_choice) > len(devices):
            dev_choice = input("Invalid choice. Try again: ")
        chosen_device = devices[int(dev_choice) - 1]
    else:
        print(f"\n⚙️  {selected['name']} will be executed on: {chosen_device} (Forced hardware)")

    return selected, chosen_device


def instantiate_model(model_info, device):
    kwargs = model_info["kwargs"].copy()
    if model_info["pass_device"]:
        kwargs["device"] = device
    return model_info["class"](**kwargs)


# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
if __name__ == "__main__":
    print("=" * 65)
    print("🧠 KHADAS MIND AI BENCHMARKING FRAMEWORK")
    print("=" * 65)

    print("\n[AVAILABLE MODES]")
    print("[1] Full Pipeline (STT -> LLM -> TTS)")
    print("[2] STT WER Evaluation (LibriSpeech Streaming Dataset)")
    print("[3] Liquid ONNX Engine Benchmark")
    print("[4] TTS RTF Evaluation (LibriSpeech Streaming Dataset)")
    print("[5] LLM Tk/s & Semantic Similarity evaluation (SQuAD dataset)")

    mode_choice = input("Choose the mode (1/2/3/4/5): ")

    num_runs = int(input("\nHow many iterations (RUNS) do you want to execute?: "))

    if mode_choice == "3":
        # ---------------------------------------------------------
        # MODE 3: LIQUID
        # ---------------------------------------------------------
        print("\n" + "=" * 65)
        print("🚀 STARTING BENCHMARK: LIQUID FOUNDATION (OFFICIAL ONNX ENGINE)")
        print("=" * 65)

        liquid_engine = OfficialLiquidONNX(model_path="Models/liquid-lfm2.5-audio-onnx", engine_dir="onnx-export")
        PATH_LIQUID_OUT = f"{BASE_DIR_OUTPUT}/Liquid_Response_Def.wav"

        durata_audio_input = librosa.get_duration(path=PATH_INPUT)
        times_total, rtf_e2e_list, cpu_history, ram_history = [], [], [], []

        for i in range(num_runs):
            print(f"\n--- Liquid Run {i + 1}/{num_runs} ---")
            out_file, total_time, avg_cpu, max_ram = liquid_engine.process_audio(PATH_INPUT, PATH_LIQUID_OUT)
            if out_file:
                durata_audio_output = librosa.get_duration(path=out_file)
                rtf_e2e = total_time / durata_audio_output
                times_total.append(total_time)
                rtf_e2e_list.append(rtf_e2e)
                cpu_history.append(avg_cpu)
                ram_history.append(max_ram)

        # Plot Liquid
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        run_labels = [f"Run {i + 1}" for i in range(len(times_total))]
        axs[0].bar(run_labels, times_total, color='#1f77b4')
        axs[0].set_title('Liquid LFM: Response Latency')
        axs[1].plot(run_labels, cpu_history, marker='s', label='Avg CPU (%)')
        axs[1].plot(run_labels, ram_history, marker='^', label='Peak RAM (%)', linestyle='--')
        axs[1].legend()
        axs[2].plot(run_labels, rtf_e2e_list, marker='s', color='orange', label='E2E RTF')
        axs[2].legend()
        plt.tight_layout()
        plt.savefig(f"{BASE_DIR_CHARTS}/LiquidMetrics_{num_runs}runs.png", dpi=300)
        plt.show()


    elif mode_choice == "2":
        # ---------------------------------------------------------
        # MODE 2: WER EVALUATION (Streaming Dataset)
        # ---------------------------------------------------------

        print("\n" + "=" * 65)
        print("📊 STARTING WER EVALUATION MODE")
        print("=" * 65)

        selected_stt, stt_device = select_model_and_device("SELECT THE STT MODEL TO TEST", AVAILABLE_STT)
        stt_name = selected_stt['name'].replace(" ", "")
        stt_module = instantiate_model(selected_stt, stt_device)
        print("\n📡 Connecting to LibriSpeech dataset (Streaming)...")

        try:
            dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
            dataset = dataset.cast_column("audio", Audio(decode=False))
            sample_pool = list(dataset.take(num_runs + 50))

        except Exception as e:
            print(f"❌ Cannot download dataset. Check your connection: {e}")
            exit()

        stt_wer_list, stt_rtf_list = [], []
        temp_wav = "temp_wer_eval.wav"

        for i in range(num_runs):

            print(f"\n--- WER Evaluation Run {i + 1}/{num_runs} ---")

            random_sample = random.choice(sample_pool)
            audio_bytes = random_sample["audio"]["bytes"]
            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            testo_reale = clean_text(random_sample["text"])

            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)

            if sr != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

            sf.write(temp_wav, audio_array, 16000, subtype = 'PCM_16')
            durata_audio = len(audio_array) / 16000

            t0 = time.perf_counter()
            testo_predetto = stt_module.transcribe(temp_wav)
            t1 = time.perf_counter()
            testo_predetto_clean = clean_text(testo_predetto)
            current_wer = wer(testo_reale, testo_predetto_clean)
            current_rtf = (t1 - t0) / durata_audio
            stt_wer_list.append(current_wer)
            stt_rtf_list.append(current_rtf)

            print(f"📝 Actual: {testo_reale}")
            print(f"🤖 Predicted: {testo_predetto_clean}")
            print(f"📊 WER: {current_wer:.3f} | RTF: {current_rtf:.3f}x")

        # Statistical Calculations
        actual_runs = len(stt_wer_list)
        avg_wer = np.mean(stt_wer_list)
        var_wer = np.var(stt_wer_list)

        avg_rtf = np.mean(stt_rtf_list)
        var_rtf = np.var(stt_rtf_list)

        print("\n" + "=" * 65)
        print(f"📊 FINAL REPORT ({selected_stt['name']} on {stt_device})")
        print("-" * 65)
        print(f"📈 WER -> Mean: {avg_wer:.3f} | Variance: {var_wer:.6f}")
        print(f"⚡ RTF -> Mean: {avg_rtf:.3f}x | Variance: {var_rtf:.6f}")
        print("=" * 65)

        # Chart Generation
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        if actual_runs <= 15:
            run_labels = [f"Run {i + 1}" for i in range(actual_runs)]
        else:
            run_labels = [str(i + 1) for i in range(actual_runs)]

        # --- Chart 1: WER ---
        axs[0].plot(run_labels, stt_wer_list, marker='^', color='green', linewidth=2, label='WER per Run')
        # Legend with Mean and Variance
        label_wer = f'Mean: {avg_wer:.3f} | Var: {var_wer:.6f}'
        axs[0].axhline(y=avg_wer, color='red', linestyle='--', linewidth=2, label=label_wer)

        axs[0].set_title(f'{selected_stt["name"]} ({stt_device}) - Accuracy Analysis')
        axs[0].set_ylabel('WER')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].set_ylim(bottom=0)
        axs[0].legend(loc='upper right', fontsize='small', framealpha=0.9)

        # --- Chart 2: RTF ---
        axs[1].plot(run_labels, stt_rtf_list, marker='o', color='blue', linewidth=2, label='RTF per Run')
        # Legend with Mean and Variance
        label_rtf = f'Mean: {avg_rtf:.3f}x | Var: {var_rtf:.6f}'
        axs[1].axhline(y=avg_rtf, color='red', linestyle='--', linewidth=2, label=label_rtf)
        axs[1].axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.4, label='Real-Time Limit')

        axs[1].set_title(f'{selected_stt["name"]} ({stt_device}) - Efficiency Analysis')
        axs[1].set_ylabel('RTF')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend(loc='upper right', fontsize='small', framealpha=0.9)

        # X-axis tick decimation logic (for 50/100 runs)
        for ax in axs:
            if actual_runs > 20:
                step = max(1, actual_runs // 10)
                ax.set_xticks(range(0, actual_runs, step))
                ax.set_xticklabels([run_labels[i] for i in range(0, actual_runs, step)])

        plt.tight_layout()
        save_path = f"{BASE_DIR_CHARTS}/WER_Eval_{stt_name}_{stt_device}_{actual_runs}runs.png"
        plt.savefig(save_path, dpi=300)
        print(f"\n✅ Benchmark completed. Chart saved: {save_path}")
        plt.show()


    elif mode_choice == "4":
        # ---------------------------------------------------------
        # MODE 4: TTS RTF EVALUATION (Streaming Dataset)
        # ---------------------------------------------------------

        print("\n" + "=" * 65)
        print(" STARTING TTS RTF EVALUATION MODE")
        print("=" * 65)

        selected_tts, tts_device = select_model_and_device("SELECT THE TTS MODEL TO TEST", AVAILABLE_TTS)
        tts_name = selected_tts['name'].replace(" ", "")
        tts_module = instantiate_model(selected_tts, tts_device)
        print("\n Connecting to LibriSpeech dataset (Streaming)...")

        try:
            dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
            sample_pool = list(dataset.take(num_runs + 50))

        except Exception as e:
            print(f" Cannot download dataset. Check your connection: {e}")
            exit()

        tts_rtf_list, tts_time_list = [], []
        temp_wav = "temp_rtf_eval.wav"

        for i in range(num_runs):

            print(f"\n--- TTS Evaluation Run {i + 1}/{num_runs} ---")

            random_sample = random.choice(sample_pool)
            testo_input = clean_text(random_sample["text"])

            print(f" Text: {testo_input}")

            t0 = time.perf_counter()

            tts_module.synthesize(testo_input, temp_wav)

            t1 = time.perf_counter()

            try:
                audio_array, sr = sf.read(temp_wav)
                durata_audio = len(audio_array) / sr
            except Exception as e:
                print(f" Errore nella lettura dell'audio generato: {e}")
                continue

            gen_time = t1 - t0
            current_rtf = gen_time / durata_audio

            tts_time_list.append(gen_time)
            tts_rtf_list.append(current_rtf)

            print(f" Audio Length: {durata_audio:.2f}s | Gen Time: {gen_time:.2f}s | RTF: {current_rtf:.3f}x")

        # Statistical Calculations
        actual_runs = len(tts_rtf_list)
        avg_rtf = np.mean(tts_rtf_list)
        var_rtf = np.var(tts_rtf_list)

        avg_time = np.mean(tts_time_list)

        print("\n" + "=" * 65)
        print(f" FINAL REPORT ({selected_tts['name']} on {tts_device})")
        print("-" * 65)
        print(f" Generation Time -> Mean: {avg_time:.3f}s")
        print(f" RTF -> Mean: {avg_rtf:.3f}x | Variance: {var_rtf:.6f}")
        print("=" * 65)

        # Chart Generation
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        if actual_runs <= 15:
            run_labels = [f"Run {i + 1}" for i in range(actual_runs)]
        else:
            run_labels = [str(i + 1) for i in range(actual_runs)]

        # --- Chart 1: Generation Time (Al posto del WER) ---
        axs[0].plot(run_labels, tts_time_list, marker='s', color='orange', linewidth=2, label='Gen Time (s)')
        label_time = f'Mean Time: {avg_time:.2f}s'
        axs[0].axhline(y=avg_time, color='red', linestyle='--', linewidth=2, label=label_time)

        axs[0].set_title(f'{selected_tts["name"]} ({tts_device}) - Synthesis Time Analysis')
        axs[0].set_ylabel('Seconds')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].set_ylim(bottom=0)
        axs[0].legend(loc='upper right', fontsize='small', framealpha=0.9)

        # --- Chart 2: RTF ---
        axs[1].plot(run_labels, tts_rtf_list, marker='o', color='blue', linewidth=2, label='RTF per Run')
        label_rtf = f'Mean: {avg_rtf:.3f}x | Var: {var_rtf:.6f}'
        axs[1].axhline(y=avg_rtf, color='red', linestyle='--', linewidth=2, label=label_rtf)
        axs[1].axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.4, label='Real-Time Limit')

        axs[1].set_title(f'{selected_tts["name"]} ({tts_device}) - Efficiency Analysis (RTF)')
        axs[1].set_ylabel('RTF')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].set_ylim(bottom=0)
        axs[1].legend(loc='upper right', fontsize='small', framealpha=0.9)

        # X-axis tick decimation logic
        for ax in axs:
            if actual_runs > 20:
                step = max(1, actual_runs // 10)
                ax.set_xticks(range(0, actual_runs, step))
                ax.set_xticklabels([run_labels[i] for i in range(0, actual_runs, step)])

        plt.tight_layout()
        save_path = f"{BASE_DIR_CHARTS}/TTS_RTF_Eval_{tts_name}_{tts_device}_{actual_runs}runs.png"
        plt.savefig(save_path, dpi=300)
        print(f"\n Benchmark completed. Chart saved: {save_path}")
        plt.show()

    elif mode_choice == "5":
        # ---------------------------------------------------------
        # MODE 5: LLM EVALUATION (Tk/s & Semantic Similarity)
        # ---------------------------------------------------------
        from sentence_transformers import SentenceTransformer, util

        print("\n" + "=" * 65)
        print(" STARTING LLM EVALUATION MODE (SPEED & SEMANTICS)")
        print("=" * 65)

        selected_llm, llm_device = select_model_and_device("SELECT THE LLM MODEL TO TEST", AVAILABLE_LLM)
        llm_name = selected_llm['name'].replace(" ", "")

        llm_module = instantiate_model(selected_llm, llm_device)

        print("\n🧠 Loading lightweight Embedding Model for Semantic Evaluation...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        print(" 📚 Connecting to SQuAD dataset (Streaming)...")
        try:
            dataset = load_dataset("squad", split="validation", streaming=True)
            sample_pool = list(dataset.take(num_runs + 50))
        except Exception as e:
            print(f" Cannot download dataset. Check your connection: {e}")
            exit()

        llm_tks_list, llm_sim_list = [], []

        for i in range(num_runs):
            print(f"\n--- LLM Evaluation Run {i + 1}/{num_runs} ---")

            random_sample = random.choice(sample_pool)
            context = random_sample["context"]
            question = random_sample["question"]

            reference_answer = random_sample["answers"]["text"][0]

            prompt = (
                f"Based on the following context, answer the question in a complete, "
                f"detailed sentence.\n\nContext: {context}\nQuestion: {question}\nAnswer:"
            )

            t0 = time.perf_counter()

            generated_text, exact_tokens = llm_module.generate(prompt)

            t1 = time.perf_counter()

            gen_time = t1 - t0
            tks = exact_tokens / gen_time if gen_time > 0 else 0

            expected_lower = reference_answer.lower().strip()
            generated_lower = generated_text.lower().strip()

            if expected_lower in generated_lower:
                accuracy_score = 1.0
                print(f" 🟢 Match Type: Exact Substring")
            else:
                embeddings = embedder.encode([reference_answer, generated_text])
                cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
                accuracy_score = max(0.0, min(1.0, cosine_sim))
                print(f" 🟡 Match Type: Semantic (Score: {accuracy_score:.3f})")

            llm_tks_list.append(tks)
            llm_sim_list.append(accuracy_score)

            print(f" ❓ Question: {question}")
            print(f" 🎯 Expected: {reference_answer}")
            print(f" 🤖 Generated: {generated_text.strip()[:100]}...")
            print(f" ⚡ Tk/s: {tks:.2f} | 🧠 Final Score: {accuracy_score:.3f}")

        # Statistical Calculations
        actual_runs = len(llm_tks_list)
        avg_tks = np.mean(llm_tks_list)
        var_tks = np.var(llm_tks_list)
        avg_sim = np.mean(llm_sim_list)
        var_sim = np.var(llm_sim_list)

        print("\n" + "=" * 65)
        print(f" FINAL REPORT ({selected_llm['name']} on {llm_device})")
        print("-" * 65)
        print(f" Speed (Tk/s) -> Mean: {avg_tks:.2f} | Variance: {var_tks:.4f}")
        print(f" Semantics   -> Mean: {avg_sim:.3f} | Variance: {var_sim:.4f}")
        print("=" * 65)

        # Chart Generation
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        run_labels = [str(i + 1) for i in range(actual_runs)] if actual_runs > 15 else [f"Run {i + 1}" for i in
                                                                                        range(actual_runs)]

        # --- Chart 1: Tokens per Second ---
        axs[0].plot(run_labels, llm_tks_list, marker='D', color='purple', linewidth=2, label='Tk/s per Run')
        label_tks = f'Mean: {avg_tks:.2f} Tk/s | Var: {var_tks:.4f}'
        axs[0].axhline(y=avg_tks, color='red', linestyle='--', linewidth=2, label=label_tks)
        axs[0].set_title(f'{selected_llm["name"]} ({llm_device}) - Speed Analysis')
        axs[0].set_ylabel('Tokens / Second')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].set_ylim(bottom=0)
        axs[0].legend(loc='upper right', fontsize='small', framealpha=0.9)

        # --- Chart 2: Semantic Similarity ---
        axs[1].plot(run_labels, llm_sim_list, marker='*', color='green', linewidth=2, label='Similarity (0 to 1)')
        label_sim = f'Mean: {avg_sim:.3f} | Var: {var_sim:.4f}'
        axs[1].axhline(y=avg_sim, color='red', linestyle='--', linewidth=2, label=label_sim)
        axs[1].set_title(f'{selected_llm["name"]} ({llm_device}) - Semantic Accuracy')
        axs[1].set_ylabel('Cosine Similarity')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].set_ylim(0, 1.1)
        axs[1].legend(loc='lower right', fontsize='small', framealpha=0.9)

        for ax in axs:
            if actual_runs > 20:
                step = max(1, actual_runs // 10)
                ax.set_xticks(range(0, actual_runs, step))
                ax.set_xticklabels([run_labels[i] for i in range(0, actual_runs, step)])

        plt.tight_layout()
        save_path = f"{BASE_DIR_CHARTS}/LLM_Eval_{llm_name}_{llm_device}_{actual_runs}runs.png"
        plt.savefig(save_path, dpi=300)
        print(f"\n Benchmark completed. Chart saved: {save_path}")
        plt.show()

    elif mode_choice == "1":
        # ---------------------------------------------------------
        # MODE 1: FULL CASCADED PIPELINE
        # ---------------------------------------------------------
        selected_stt, stt_device = select_model_and_device("SELECT THE STT MODEL", AVAILABLE_STT)
        selected_llm, llm_device = select_model_and_device("SELECT THE LLM MODEL", AVAILABLE_LLM)
        selected_tts, tts_device = select_model_and_device("SELECT THE TTS MODEL", AVAILABLE_TTS)

        # Dynamic File Name Generation
        stt_name = selected_stt['name'].replace(" ", "")
        llm_name = selected_llm['name'].replace(" ", "")
        tts_name = selected_tts['name'].replace(" ", "")
        run_signature = f"{stt_name}{stt_device}_{llm_name}{llm_device}_{tts_name}{tts_device}"

        PATH_OUT_TXT = f"{BASE_DIR_OUTPUT}/Text_{run_signature}.txt"
        PATH_OUT_WAV = f"{BASE_DIR_OUTPUT}/Audio_{run_signature}.wav"
        PATH_CHART = f"{BASE_DIR_CHARTS}/StressTest_{run_signature}.png"
        PATH_CSV_REPORT = f"{BASE_DIR_CSV}/benchmark_{run_signature}.csv"

        print("\n" + "=" * 65)
        print(f"🚀 PIPELINE INITIALIZATION: {run_signature}")
        print("=" * 65)

        # Dynamic instantiation of modules by injecting the device
        stt = instantiate_model(selected_stt, stt_device)
        llm = instantiate_model(selected_llm, llm_device)
        tts = instantiate_model(selected_tts, tts_device)

        times_stt, times_llm, times_tts, times_total = [], [], [], []
        cpu_history, ram_history, llm_speed_tok_sec = [], [], []
        stt_rtf_list, tts_rtf_list, stt_wer_list = [], [], []

        durata_audio_input = librosa.get_duration(path=PATH_INPUT)

        for i in range(num_runs):
            print(f"\n--- Run {i + 1}/{num_runs} ---")
            psutil.cpu_percent(interval=None)
            ram_start = psutil.virtual_memory().percent

            t_run_start = time.perf_counter()

            # PHASE 1: STT
            t0 = time.perf_counter()
            extracted_text = stt.transcribe(PATH_INPUT)
            t1 = time.perf_counter()

            run_stt_time = t1 - t0
            times_stt.append(run_stt_time)
            rtf_stt = run_stt_time / durata_audio_input
            stt_rtf_list.append(rtf_stt)

            current_wer = wer(clean_text(REFERENCE_TEXT), clean_text(extracted_text))
            stt_wer_list.append(current_wer)
            print(f"🗣️ STT | RTF: {rtf_stt:.3f}x | WER: {current_wer:.3f}")

            # PHASE 2: LLM
            t2 = time.perf_counter()
            llm_response, num_tokens = llm.generate(extracted_text, PATH_OUT_TXT)
            t3 = time.perf_counter()

            run_llm_time = t3 - t2
            times_llm.append(run_llm_time)
            llm_speed_tok_sec.append(num_tokens / run_llm_time)

            # PHASE 3: TTS
            t4 = time.perf_counter()
            tts_success = tts.synthesize(llm_response, PATH_OUT_WAV)
            t5 = time.perf_counter()

            run_tts_time = t5 - t4
            times_tts.append(run_tts_time)

            if tts_success:
                durata_audio_output = librosa.get_duration(path=PATH_OUT_WAV)
                rtf_tts = run_tts_time / durata_audio_output
                tts_rtf_list.append(rtf_tts)
                print(f"🔊 TTS | RTF: {rtf_tts:.3f}x | Audio: {durata_audio_output:.2f}s")
            else:
                tts_rtf_list.append(0)

            t_run_end = time.perf_counter()
            times_total.append(t_run_end - t_run_start)
            cpu_history.append(psutil.cpu_percent(interval=None))
            ram_history.append(psutil.virtual_memory().percent)

        print("\n" + "=" * 65)
        print("📊 STRESS TEST RESULTS (MEAN)")
        print(
            f"🎙️ STT: {np.mean(times_stt):.2f}s | 🧠 LLM: {np.mean(times_llm):.2f}s | 🔊 TTS: {np.mean(times_tts):.2f}s")
        print("=" * 65)

        # CSV Creation (Hardware info included)
        df_report = pd.DataFrame({
            "Run": [i + 1 for i in range(num_runs)],
            "STT_Model": [f"{selected_stt['name']} ({stt_device})"] * num_runs,
            "LLM_Model": [f"{selected_llm['name']} ({llm_device})"] * num_runs,
            "TTS_Model": [f"{selected_tts['name']} ({tts_device})"] * num_runs,
            "STT_Time_s": [round(t, 2) for t in times_stt],
            "LLM_Time_s": [round(t, 2) for t in times_llm],
            "TTS_Time_s": [round(t, 2) for t in times_tts],
            "Total_Time_s": [round(t, 2) for t in times_total],
            "LLM_Speed_tok_sec": [round(s, 2) for s in llm_speed_tok_sec],
            "CPU_Usage_pct": [round(c, 1) for c in cpu_history],
            "RAM_Usage_pct": [round(r, 1) for r in ram_history],
            "STT_RTF": [round(r, 3) for r in stt_rtf_list],
            "TTS_RTF": [round(r, 3) for r in tts_rtf_list],
            "STT_WER": [round(w, 3) for w in stt_wer_list]
        })
        df_report.to_csv(PATH_CSV_REPORT, index=False, sep=";")
        print(f"📄 CSV Saved at: {PATH_CSV_REPORT}")

        # ==========================================
        # 📊 GENERATING CHARTS (MERGED DASHBOARD)
        # ==========================================
        print("\nGenerating charts...")

        fig, axs = plt.subplots(5, 1, figsize=(10, 20))
        run_labels = [f"Run {i + 1}" for i in range(num_runs)]

        # --- Chart 1: Execution Times ---
        axs[0].bar(run_labels, times_stt, label=f'STT ({selected_stt["name"]} on {stt_device})', color='#1f77b4')
        axs[0].bar(run_labels, times_llm, bottom=times_stt, label=f'LLM ({selected_llm["name"]} on {llm_device})',
                   color='#ff7f0e')
        axs[0].bar(run_labels, times_tts, bottom=np.array(times_stt) + np.array(times_llm),
                   label=f'TTS ({selected_tts["name"]} on {tts_device})', color='#2ca02c')
        axs[0].set_title('Execution Time per Run by Model')
        axs[0].set_ylabel('Seconds')
        axs[0].legend()
        axs[0].grid(axis='y', linestyle='--', alpha=0.7)

        # --- Chart 2: Token/s ---
        axs[1].plot(run_labels, llm_speed_tok_sec, marker='o', color='#d62728', linewidth=2)
        axs[1].set_title(f'LLM Generation Speed ({selected_llm["name"]} on {llm_device})')
        axs[1].set_ylabel('Token / sec')
        axs[1].grid(True, linestyle='--', alpha=0.7)

        # --- Chart 3: System Resources ---
        ax3 = axs[2]
        ax3_bis = ax3.twinx()
        line1 = ax3.plot(run_labels, cpu_history, marker='s', color='#9467bd', label='Avg CPU (%)', linewidth=2)
        line2 = ax3_bis.plot(run_labels, ram_history, marker='^', color='#8c564b', label='Peak RAM (%)', linewidth=2,
                             linestyle='--')
        ax3.set_title('System Resource Impact')
        ax3.set_ylabel('CPU Usage (%)', color='#9467bd')
        ax3_bis.set_ylabel('RAM Usage (%)', color='#8c564b')
        ax3.grid(True, linestyle='--', alpha=0.7)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')

        # --- Chart 4: Audio RTF ---
        axs[3].plot(run_labels, stt_rtf_list, marker='o', color='blue', label='STT RTF', linewidth=2)
        axs[3].plot(run_labels, tts_rtf_list, marker='s', color='orange', label='TTS RTF', linewidth=2)
        axs[3].axhline(y=1.0, color='red', linestyle='--', label='Real-Time (1.0x)')
        axs[3].set_title('Audio Models Efficiency (Real-Time Factor - Lower is Better)')
        axs[3].set_ylabel('RTF')
        axs[3].legend()
        axs[3].grid(True, linestyle='--', alpha=0.7)

        # --- Chart 5: STT WER ---
        axs[4].plot(run_labels, stt_wer_list, marker='^', color='green', linewidth=2)
        axs[4].set_title(f'STT Accuracy ({selected_stt["name"]} on {stt_device}) - Word Error Rate')
        axs[4].set_ylabel('WER')
        axs[4].set_ylim(bottom=0)
        axs[4].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(PATH_CHART, dpi=300)
        print(f"📊 Dashboard saved at: {PATH_CHART}")
        plt.show()
