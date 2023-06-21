from pydub import AudioSegment
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Convert MP3 files to WAV.')
    parser.add_argument('--data', nargs='+', help='Data directories.')
    args = parser.parse_args()
    return args

def find_mp3_files(path):
    mp3_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".mp3") or file.endswith(".Mp3"):
                mp3_files.append(os.path.join(root, file))
    return mp3_files

def convert_mp3_to_wav(mp3_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    wav_file_path = mp3_file_path.rsplit('.', 1)[0] + '.wav'
    audio.export(wav_file_path, format='wav')

def main():
    arguments = parse_args()
    
    for path in arguments.data:
        mp3_files = find_mp3_files(path)

        for mp3_file in mp3_files:
            convert_mp3_to_wav(mp3_file)

if __name__ == "__main__":
    main()
