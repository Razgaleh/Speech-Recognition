# adapted from https://github.com/mozilla/DeepSpeech/blob/master/native_client/python/client.py
# which has a Mozzila Public License:
# https://github.com/mozilla/DeepSpeech/blob/master/LICENSE


from deepspeech import Model, version
import librosa as lr
import numpy as np
import os
import argparse
from jiwer import wer
import thinkdsp
import matplotlib as plt

filter = 0
volume = 1
# suppressing the tensorflow messages to allow a more user friendly cmd interface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# code for command line interface to choose language
# command : -lang -vol -noise
# command options: EN (English), ES (Spanish), IT (Italian)
# sample command: python3 deepspeech_file.py -lang ES -vol 50 -noise high
parser = argparse.ArgumentParser(description="Your Airport Virtual Assistant")
parser.add_argument(
    "-lang",
    nargs="*",
    metavar="lang",
    type=str,
    help="Choose your preferred language: EN , ES , IT, FR",
)
parser.add_argument(
    "-noise",
    nargs="*",
    metavar="noise",
    type=str,
    help="Choose noise levels between high or low",
)
parser.add_argument(
    "-vol",
    nargs="*",
    metavar="volume",
    type=int,
    help="Choose volume levels between 1-100",
)

args = parser.parse_args()

# language options
if args.lang == ["EN"]:
    print("Welcome! I am your airport virtual assistant.")
    print("English is set as your preferred language.")
    lang = "EN"
elif args.lang == ["ES"]:
    print("¡Bienvenido! Soy tu asistente virtual en el aeropuerto.")
    print("El español está configurado como su idioma preferido.")
    lang = "ES"
elif args.lang == ["IT"]:
    print("Benvenuto! Sono il tuo assistente virtuale in aeroporto.")
    print("L'italiano è impostato come lingua preferita.")
    lang = "IT"
elif args.lang == ["FR"]:
    print("Accueillir! Je suis votre assistant virtuel d'aéroport.")
    print("Français est défini comme votre langue préférée.")
    lang = "FR"

# noise level options to choose the filter
# the choices are : -noise high or -noise low
if args.noise == ["high"]:
    print("Low pass filter is activated")
    filter = 1
elif args.noise == ["low"]:
    print("High pass filter is activated")
    filter = 2


# volume options
# command example : -vol 50
if isinstance(args.vol, int):
    print("The volume is", args.vol)
    volume = args.vol

# audio file names
audio_files_EN = [
    "checkin.wav",
    "parents.wav",
    "suitcase.wav",
    "what_time.wav",
    "where.wav",
    "taxi.wav",  # recorded audio files by me
    "passport.wav",  # recorded audio files by me
]
audio_files_ES = [
    "checkin_es.wav",
    "parents_es.wav",
    "suitcase_es.wav",
    "what_time_es.wav",
    "where_es.wav",
]
audio_files_IT = [
    "checkin_it.wav",
    "parents_it.wav",
    "suitcase_it.wav",
    "what_time_it.wav",
    "where_it.wav",
]

audio_files_FR = ["passport_fr.wav", "taxi_fr.wav"]  # recorded audio files by me

# reference audio file transcription
Reference_EN = [
    "where is the checkin desk",
    "i have lost my parents",
    "please I have lost my suitcase",
    "what time is my plane",
    "where are the restaurants and shops",
    "i need a taxi",
    "i have lost my passport",
]

Reference_ES = [
    "donde estan los mostradores",
    "he perdido a mis padres",
    "por favor he perdido mi maleta",
    "a que hora es mi avion",
    "donde estan los restaurantes y las tiendas",
]

Reference_IT = [
    "dove e il bancone",
    "ho perso i miei genitori",
    "per favore ho perso la mia valigia",
    "a che ora e il mio aereo",
    "dove sono i ristoranti e i negozi",
]

Reference_FR = ["j'ai perdu mon passeport", "j'ai besoin d'un taxi"]

if lang == "EN":
    scorer = "deepspeech-0.9.3-models.scorer"
    model = "deepspeech-0.9.3-models.pbmm"
    audio_file_path = "./Ex4_audio_files/EN/"
    audio_files = audio_files_EN
    reference = Reference_EN
elif lang == "ES":
    scorer = "kenlm_es.scorer"
    model = "output_graph_es.pbmm"
    audio_file_path = "./Ex4_audio_files/ES/"
    audio_files = audio_files_ES
    reference = Reference_ES
elif lang == "IT":
    scorer = "kenlm_it.scorer"
    model = "output_graph_it.pbmm"
    audio_file_path = "./Ex4_audio_files/IT/"
    audio_files = audio_files_IT
    reference = Reference_IT
elif lang == "FR":
    scorer = "kenlm_fr.scorer"
    model = "output_graph_fr.pbmm"
    audio_file_path = "./Ex4_audio_files/FR/"
    audio_files = audio_files_FR
    reference = Reference_FR


assert os.path.exists(scorer), (
    scorer
    + "not found. Perhaps you need to download a scroere  from the deepspeech release page: https://github.com/mozilla/DeepSpeech/releases"
)
assert os.path.exists(model), (
    model
    + "not found. Perhaps you need to download a  model from the deepspeech release page: https://github.com/mozilla/DeepSpeech/releases"
)

ds = Model(model)
ds.enableExternalScorer(scorer)
results = []
for i in range(0, len(audio_files)):
    audio_file = audio_file_path + audio_files[i]
    assert os.path.exists(audio_file), audio_file + "does not exist"
    if filter == 1:
        wave = thinkdsp.read_wave(audio_file)
        spectrum = wave.make_spectrum()
        spectrum.low_pass(cutoff=5000, factor=0.01)
        wave.scale(volume)
        wave = spectrum.make_wave()
        wave.write(filename="temp_filtered_audio_file.wav")
        new_audio_file = "temp_filtered_audio_file.wav"
    elif filter == 2:
        wave = thinkdsp.read_wave(audio_file)
        spectrum = wave.make_spectrum()
        spectrum.high_pass(cutoff=100, factor=0.01)
        wave.scale(volume)
        wave = spectrum.make_wave()
        wave.write(filename="temp_filtered_audio_file.wav")
        new_audio_file = "temp_filtered_audio_file.wav"
    elif filter == 0:
        new_audio_file = audio_file

    desired_sample_rate = ds.sampleRate()

    audio = lr.load(new_audio_file, sr=desired_sample_rate)[0]
    audio = (audio * 32767).astype(np.int16)  # scale from -1 to 1 to +/-32767
    res = ds.stt(audio)
    # res = ds.sttWithMetadata(audio, 1).transcripts[0]
    results.append(res)

# uncomment to see the model's results
# print(results)

# calculating WER
error = round(wer(reference, results) * 100, 2)
print("WER for", lang, "is:")
print(error, "%")
