import torch
import sounddevice as sd

text = 'приветствую! ... я ... голосовой помощник Айкон!'
language = 'ru'
model_id = 'v3_1_ru'
sample_rate = 48000
speaker = 'xenia' # aidar, baya, kseniya, xenia, random
device = torch.device('cpu')
model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=language, speaker=model_id)
model.to(device)


def va_speak(what: str):
    audio = model.apply_tts(text=what, speaker=speaker, sample_rate=sample_rate)
    sd.play(audio, sample_rate)
    sd.wait()

va_speak(text)
