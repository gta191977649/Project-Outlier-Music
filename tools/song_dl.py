from yt_dlp import YoutubeDL

if __name__ == '__main__':
    video_url = "https://www.youtube.com/watch?v=Bnx85z78O2w"
    download_folder = '/Users/nurupo/Desktop/dev/audio/test'

    options = {
        'format': 'bestaudio/best',
        'keepvideo': False,
        'outtmpl': f'{download_folder}/%(title)s.%(ext)s',
        'ignoreerrors': True,  # Skip unavailable videos
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '90',
        }]
    }

    with YoutubeDL(options) as ydl:
        ydl.download([video_url])
