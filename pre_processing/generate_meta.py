import os
from model.song import Song

if __name__ == '__main__':
    PATH = "/Users/nurupo/Desktop/dev/audio/aimyon"
    # loop all folder
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".mp3"):
                filepath = os.path.join(root, file)
                filename = os.path.splitext(file)[0]
                print(f"START PROCESS:{filename}")
                title = filename
                artist = "Nogizaka64"
                album = "N/A"
                release = 2000
                language = "ja"
                tags = "pop"
                song = Song(id=title, file=filepath, title=title, artist=artist)
                song.album = album
                song.release = release
                song.language = language
                song.tags = tags

                song.save(filepath.replace(".mp3", ".h5"))
                print(f"Processed: {title}")

