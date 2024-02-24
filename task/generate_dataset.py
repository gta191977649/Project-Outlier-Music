import os

import pandas as pd

import feature.extract as feature
from model.song import Song

def generate_dataset(base_path,dataset="music4all"):
    if dataset == "music4all":
        audio_folder = "audios"
        path_csv = os.path.join(base_path,"dataset.csv")
        if os.path.exists(path_csv):
            csv_index = pd.read_csv(path_csv)
            for _,item in csv_index.iterrows():
                id = item["id"]
                file = os.path.join(base_path,audio_folder,id+".mp3")
                if os.path.exists(file):
                    title = item["song"]
                    artist = item["artist"]
                    album = item["album_name"]
                    release = item["release"]
                    language = item["lang"]
                    tags = item["tags"]
                    song = Song(id=id, file=file, title=title, artist=artist)
                    song.album = album
                    song.release = release
                    song.language = language
                    song.tags = tags

                    song.save(file.replace(".mp3", ".h5"))
                    print(song)
                else:
                    print(f"{file} is not exists !!")


if __name__ == '__main__':
    PATH = "/Users/nurupo/Desktop/dev/Project-Outlier-Music/dataset/music4all"
    generate_dataset(PATH, dataset="music4all")

