import os
import json
import soundfile as sf

def process_wav_file(file_path):
    try:
        data, sample_rate = sf.read(file_path)
        duration = len(data) / sample_rate


        file_name = os.path.basename(file_path)
        file_dir = os.path.dirname(file_path)

        content = {
            "key": "",
            "artist": "",
            "sample_rate": sample_rate,
            "file_extension": "wav",
            "description": "a @ music with * as the rhythm",
            "keywords": "",
            "duration": duration,
            "bpm": "",
            "genre": "",
            "title": "",
            "name": os.path.splitext(file_name)[0],
            "instrument": "Mix",
            "moods": []
        }

        json_file_path = os.path.join(file_dir, f"{os.path.splitext(file_name)[0]}.json")


        with open(json_file_path, 'w') as json_file:
            json.dump(content, json_file, indent=4)

        print(f"Created JSON file: {json_file_path}")

    except Exception as e:
        print(f"Error processing file: {file_path}")
        print(e)

def process_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                if 'Ancient' in file_path or 'Ballet' in file_path or 'Waacking' in file_path or 'Latin' in file_path:
                        
                    process_wav_file(file_path)

directory_path = "/path/to/audio_clips/"

process_directory(directory_path)