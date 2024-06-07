import os
from model.song import Song
keys = {
    "a-ha - Take On Me (Official Video) [Remastered in 4K]": {
        "key": "A",
        "mode": 'major',
    },
    "Maroon 5 - She Will Be Loved (Official Music Video)": {
        "key": "C",
        "mode": 'minor',
    },
    "Toto - Africa (Official HD Video)":{
        "key": "C#",
        "mode": 'minor',
    },
    "君はメロディー Short ver.  AKB48[公式]": {
        "key": "Ab",
        "mode":"major"
    },
}
if __name__ == '__main__':
    PATH = "F:\\dataset\\custom"
    # loop all folder
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".mp3"):
                filepath = os.path.join(root, file)
                filename = os.path.splitext(file)[0]
                print(f"START PROCESS:{filename}")
                title = filename
                artist = "Custom"
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

