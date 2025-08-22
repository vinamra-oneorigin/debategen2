import torch
import numpy as np
import base64
import audioop
import wave
import torchaudio
import logging
import os
import librosa
import io
from pydub import AudioSegment

logger = logging.getLogger(__name__)

torch.set_num_threads(1)
torchaudio.set_audio_backend("soundfile")

model = torch.jit.load("snakers4_silero-vad_master/files/silero_vad.jit")

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound

def VAD_Amp(chunk) :
    try :
        audio_int16 = np.frombuffer(chunk, np.int16)
        audio_float32 = int2float(audio_int16)
        amplitude = np.max(np.abs(audio_int16))
        confidence = model(torch.from_numpy(audio_float32), 8000).item()
        return amplitude, confidence
    except Exception :
        logger.error("error occured, failed to VAmp score as: ", exc_info = True)


def payload_to_audio(payload) :
    chunk = base64.b64decode(payload)
    audio = audioop.ulaw2lin(chunk, 2)
    audio = audioop.ratecv(audio, 2, 1, 8000, 16000, None)[0]
    return audio

def write_wav(name, data_array) :
    try :
        wf = wave.open(name, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b''.join(data_array))
        wf.close()
    except Exception :
        logger.error("error occured, failed to save recorded audio as: ", exc_info = True)

def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs

def encode_to_mulaw(data, samplerate) :
    resampled_data = librosa.resample(np.frombuffer(data, dtype=np.int16).astype(float), orig_sr=samplerate, target_sr=8000)
    resampled_data = resampled_data.astype(np.float32)
    resampled_data /= np.max(np.abs(resampled_data))
    pcm_audio = np.array([int(sample * 32767) for sample in resampled_data], dtype=np.int16)
    mulaw_audio = audioop.lin2ulaw(pcm_audio, 2)
    base64_audio = base64.b64encode(mulaw_audio).decode("utf-8")
    return base64_audio

# def checkVoiceID(voice_id, reschedule_Flag) :
#     try :
#         PATH = os.getenv("INTRO_MESSAGE_PATH")
#         if not os.path.exists(PATH): os.makedirs(PATH)
#         files = os.listdir(PATH)

#         if reschedule_Flag == "false" :
#             if f"{voice_id}.txt" not in files :

#                 file_path = os.path.join(PATH, f"{voice_id}.txt")
#                 intro_payload, res_time = text_to_speech(os.getenv("INTRO_MESSAGE"), voice_id)
#                 text_file = open(file_path, "wb")
#                 text_file.write(intro_payload)
#                 text_file.close()
#                 print(f"successfully generated introductory message for {voice_id}")

#         else :

#             if f"rescheduled_{voice_id}.txt" not in files :

#                 file_path = os.path.join(PATH, f"rescheduled_{voice_id}.txt")
#                 intro_payload, res_time = text_to_speech(os.getenv("RESCHEDULE_MESSAGE"), voice_id)
#                 text_file = open(file_path, "wb")
#                 text_file.write(intro_payload)
#                 text_file.close()
#                 print(f"successfully generated rescheduled introductory message for {voice_id}")
#     except Exception :
#         logger.error(f"Failed to create intro audio as ", exc_info = True)
#         return {"message": f"Error generating pregenerated introduction Audio. for {voice_id}"}

def mp3_to_bytes(folder_path):
    data_array = []
    files = get_files_from_folder(folder_path)
    for file in files :
        if file.startswith('keyboard') :
            keyboard = True
        else :
            keyboard = False
        mp3 = f"{folder_path}/{file}"
        with open(mp3, 'rb') as file:
            bytes_data = file.read()
        data_array.append(convert_mp3_to_wav(bytes_data, keyboard))
    return data_array

def convert_mp3_to_wav(mp3_bytes, keyboard):
    if keyboard is True :
        framerate = 16000
    else :
        framerate = 22000
    mp3_audio = io.BytesIO(mp3_bytes)
    audio = AudioSegment.from_file(mp3_audio, format="mp3")
    normalized_audio = audio.apply_gain(50)
    normalized_audio = audio.set_channels(1)
    normalized_audio = audio.set_frame_rate(framerate)
    wav_io = io.BytesIO()
    normalized_audio.export(wav_io, format="wav")
    wav_bytes = wav_io.getvalue()
    return wav_bytes

def get_files_from_folder(folder_path):
    files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    return files