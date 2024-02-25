import os
import pandas as pd
from model.song import Song
from multiprocessing import Pool
from tqdm import tqdm

key_number_map = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B"
}
def process_song(args):
    item, base_path, audio_folder = args  # Unpack arguments
    id = item["id"]
    file = os.path.join(base_path, audio_folder, id + ".mp3")
    h5_file = os.path.join(base_path, audio_folder, id + ".h5")
    if os.path.exists(file):
        if os.path.exists(h5_file):
            return f"{h5_file} exists, skipped"
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
        return f"Processed {file}"
    else:
        return f"{file} does not exist"


def init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def generate_dataset(base_path, dataset="music4all"):
    if dataset == "music4all":
        audio_folder = "audios"
        path_csv = os.path.join(base_path, "dataset.csv")

        if os.path.exists(path_csv):
            csv_index = pd.read_csv(path_csv)
            items = [item for _, item in csv_index.iterrows()]
            args = [(item, base_path, audio_folder) for item in items]  # Prepare arguments

            with Pool(processes=os.cpu_count(), initializer=init_worker) as pool:
                results = list(tqdm(pool.imap_unordered(process_song, args),
                                    total=len(items), desc="Processing songs"))
                
                for result in results:
                    print(result)


if __name__ == '__main__':
    PATH = "/home/nurupo/Desktop/dev/music4all"
    generate_dataset(PATH, dataset="music4all")
