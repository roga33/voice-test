import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

TEXT = "Hello... good morning!"
VOICE = ("A warm, friendly middle-aged man with a slightly husky voice, "
         "speaking gently and smiling as he talks, calm morning energy.")

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cpu",
    dtype=torch.float32,
)

wavs, sr = model.generate_voice_design(
    text=TEXT,
    language="English",
    instruct=VOICE,
    max_new_tokens=2048,
    temperature=0.9,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.05,
    non_streaming_mode=True,
)
sf.write("good_morning.wav", wavs[0], sr)
print("done")
