from yt_dlp import YoutubeDL

if __name__ == '__main__':
    video_url = "https://www.youtube.com/watch?v=0xSiBpUdW4E&list=PLZzBQGYFOHDHQJHWBuwQmAHZaZ2EJxcAX&index=1"
    download_folder = r"/mnt/f/dataset/aimyon"

    options = {
        'format': 'bestaudio/best',
        'keepvideo': False,
        'outtmpl': f'{download_folder}/%(title)s.%(ext)s',
        'ignoreerrors': True,  # Skip unavailable videos
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }]
    }

    with YoutubeDL(options) as ydl:
        ydl.download([video_url])
