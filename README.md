# WAV 2 MIDI

A Python project for converting audio files (WAV/MP3) into MIDI files using machine learning (HMM), with evaluation utilities for transcription accuracy.

## Features

- **Audio to MIDI conversion** using a Hidden Markov Model (HMM) trained on Constant-Q Transform features.
- **MP3 to WAV batch conversion** utility.
- **MIDI writing** with note onsets and velocities.
- **Transcription evaluation** (precision, recall, F1) using PrettyMIDI.
- Modular code for feature extraction, state representation, and evaluation.

## Project Structure

```
wav_2_midi/
│
├── main.py                        # Main script: training, conversion, evaluation
├── mp3_2_wav.py                   # Utility: batch convert MP3 files to WAV
├── model.joblib                   # Saved HMM model (after training)
├── River Flows In You_Output.mid  # Example output MIDI file
├── River Flows In You_Output_12_States.mid # Example output MIDI (12-state version)
│
└── modules/
    ├── audio_to_midi_converter.py # AudioToMIDIConverter: HMM training & decoding
    ├── evaluation.py              # evaluate_transcription: MIDI evaluation metrics
    ├── feature_extractor.py       # FeatureExtractor: CQT, pitch, chroma, onsets, velocities
    ├── midi_writer.py             # MIDIWriter: Write MIDI files from decoded notes
    └── state_representation.py    # StateRepresentation: Map features to HMM states
```

## Requirements

- Python 3.7+
- [librosa](https://librosa.org/)
- [numpy](https://numpy.org/)
- [hmmlearn](https://hmmlearn.readthedocs.io/)
- [mido](https://mido.readthedocs.io/)
- [pretty_midi](https://github.com/craffel/pretty-midi)
- [pydub](https://github.com/jiaaro/pydub) (for MP3 to WAV conversion)
- [joblib](https://joblib.readthedocs.io/)

Install dependencies:
```sh
pip install numpy librosa hmmlearn mido pretty_midi pydub joblib
```

## Usage

### 1. Convert MP3 files to WAV

```sh
python wav_2_midi/mp3_2_wav.py --data <directory_with_mp3s>
```

### 2. Train HMM and Convert Audio to MIDI

- Place your training WAV files in `wav_2_midi/training_data/`.
- Place your test WAV file in `wav_2_midi/testing_data/`.

Run the main script:
```sh
python wav_2_midi/main.py
```
This will:
- Train an HMM on the training data.
- Save the model as `model.joblib`.
- Convert the test audio to MIDI (`River Flows In You_Output_12_States.mid`).
- Evaluate the transcription (precision, recall, F1).

### 3. Evaluate Transcription

The evaluation compares the generated MIDI with a ground truth MIDI using [`evaluate_transcription`](wav_2_midi/modules/evaluation.py).

## Module Overview

- [`main.py`](wav_2_midi/main.py): Orchestrates training, conversion, and evaluation.
- [`mp3_2_wav.py`](wav_2_midi/mp3_2_wav.py): Finds and converts all MP3s in a directory to WAV.
- [`modules/audio_to_midi_converter.py`](wav_2_midi/modules/audio_to_midi_converter.py): Main logic for HMM training and decoding.
- [`modules/feature_extractor.py`](wav_2_midi/modules/feature_extractor.py): Extracts CQT, pitch, chroma, onsets, and velocities from audio.
- [`modules/midi_writer.py`](wav_2_midi/modules/midi_writer.py): Writes MIDI files from decoded notes, onsets, and velocities.
- [`modules/state_representation.py`](wav_2_midi/modules/state_representation.py): Maps features to discrete HMM states and vice versa.
- [`modules/evaluation.py`](wav_2_midi/modules/evaluation.py): Evaluates MIDI transcription