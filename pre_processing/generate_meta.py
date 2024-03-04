import os
from model.song import Song

if __name__ == '__main__':
    PATH = "/home/nurupo/Desktop/dev/music4all/europe_aud"
    # loop all folder
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".mp3"):
                filepath = os.path.join(root, file)
                filename = os.path.splitext(file)[0]
                print(f"START PROCESS:{filename}")
                title = filename
                artist = "europe"
                album = "N/A"
                release = 2000
                language = "en"
                tags = "rock,pop"
                song = Song(id=title, file=filepath, title=title, artist=artist)
                song.album = album
                song.release = release
                song.language = language
                song.tags = tags

                song.save(filepath.replace(".mp3", ".h5"))
                print(f"Processed: {title}")

