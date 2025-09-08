import os
from joblib import dump, load
from modules.evaluation import evaluate_transcription
from modules.audio_to_midi_converter import AudioToMIDIConverter

def main():
    train_data_dir = "training_data"
    audio_files_train = [os.path.join(train_data_dir, file) for file in os.listdir(train_data_dir) if file.endswith(".wav") or file.endswith(".Wav")]

    audio_file_convert = "testing_data\River Flows In You.wav"
    midi_file_output = "River Flows In You_Output_12_States.mid"
    converter = AudioToMIDIConverter(n_states=24)
    converter.train(audio_files_train)
    print(converter.model.monitor_.converged)

    dump(converter.model, 'model.joblib')
    
    converter.convert(audio_file_convert, midi_file_output)

    print(f"Successfully converted {audio_file_convert} to {midi_file_output}")

    precision, recall, f1_score = evaluate_transcription('River Flows In You_Output.mid', 'testing_data\River Flows In You.mid')
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

if __name__ == "__main__":
    main()