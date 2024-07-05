from pymusickit.key_finder import KeyFinder
import matplotlib as plt

from feature.extract import extract_feature

audio_path = '/Users/nurupo/Desktop/dev/audio/aimyon/あいみょん – 愛の花【OFFICIAL MUSIC VIDEO】.mp3'

key,mode = extract_feature(audio_path,"key")
print(key,mode)
