import os
import pandas as pd

# Path to the CSV file containing the URLs and timestamps
csv_file_path = r"{PATH_TO_CSV_FILE}\dataset.csv"
# Directory to save the downloaded videos
output_dir = r"{PATH_TO_OUTPUT_FOLDER}"
# Path to the ffmpeg and yt-dlp binaries
bin_dir = r"{PATH_TO_FFMPEG_FOLDER}\ffmpeg-6.0-full_build\ffmpeg-6.0-full_build\bin"


def remove_timeskip_from_url(url):
    if 'facebook' in url or 'youtube' in url:
        url = url.split('&t=')[0]
    elif 'vimeo' in url:
        url =url.split('#t=')[0]
    elif 'dailymotion' in url:
        url = url.split('?t=')[0]
    elif 'twitter' in url:
        url+= url.split('/t=')[0]
    return url


# Read the CSV file using pandas
df = pd.read_csv(csv_file_path)

# Iterate over each row in the DataFrame
for _, row in df.iterrows():
    url = remove_timeskip_from_url(row['Link'])
    filename = row['File Name']
    counter = row['Counter']
    start_time = row['Start Time']
    end_time = row['End Time']
    start_time = start_time[:-3] + '.' + start_time[-2:]
    end_time = end_time[:-3] + '.' + end_time[-2:]

    try:
        os.system('@echo OFF')
        os.chdir(bin_dir)
        os.system("cls")
        os.system('title Downloading...')
        os.system(
            f'yt-dlp -P "{output_dir}" -f "bv+ba/b" -o "{filename}_{counter}.mp4" --download-sections "*{start_time}-{end_time}" {url}'
        )
    except Exception as e:
        print(f"Failed to download {url}: {e}")
