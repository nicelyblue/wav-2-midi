import os
from modules.evaluation import evaluate_transcription
from modules.audio_to_midi_converter import AudioToMIDIConverter

def main():
    train_data_dir = "training_data"
    audio_files_train = [os.path.join(train_data_dir, file) for file in os.listdir(train_data_dir) if file.endswith(".wav") or file.endswith(".Wav")]

    audio_file_convert = "path_to_audio_file_to_convert.wav"
    midi_file_output = "output_midi_file.mid"
    converter = AudioToMIDIConverter()
    converter.train(audio_files_train)
    converter.convert(audio_file_convert, midi_file_output)

    print(f"Successfully converted {audio_file_convert} to {midi_file_output}")

    precision, recall, f1_score = evaluate_transcription('output_midi_file.mid', 'ground_truth_midi.mid')
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

if __name__ == "__main__":
    main()