import os

if __name__ == '__main__':
    PATH = ""
    # loop all folder
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".mp3"):
                filepath = os.path.join(root, file)
                
