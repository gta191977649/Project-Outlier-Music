import requests
import os

def download_files(filename_list, base_url, save_folder):
    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Read the list of filenames
    with open(filename_list, 'r') as file:
        filenames = file.readlines()

    # Trim whitespace and download each file
    for filename in filenames:
        filename = filename.strip()
        download_url = base_url.format(filename=filename)
        print(download_url)
        response = requests.get(download_url)
        if response.status_code == 200:
            # Save the file to the specified folder
            with open(os.path.join(save_folder, f"{filename}.mid"), 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {filename}.midi successfully.")
        else:
            print(f"Failed to download {filename}. Check URL and internet connection.")

# Parameters
filename_list = "haydn.txt"  # Path to the text file containing the filenames
base_url = "https://kern.humdrum.org/cgi-bin/ksdata?location=musedata/haydn/quartet&file={filename}.krn&format=midi"
save_folder = "E:\\dev\\research\\dataset\\midi\\Haydn"  # Folder where files will be saved

# Function call
download_files(filename_list, base_url, save_folder)
