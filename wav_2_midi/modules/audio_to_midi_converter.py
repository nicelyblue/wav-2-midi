import numpy as np
from hmmlearn import hmm
from feature_extractor import FeatureExtractor
from midi_writer import MIDIWriter
from state_representation import StateRepresentation

class AudioToMIDIConverter:
    def __init__(self, n_states=12, n_iter=100):
        self.n_states = n_states
        self.n_iter = n_iter
        self.model = None

    def train(self, audio_files):
        all_states = []

        for audio_file in audio_files:
            extractor = FeatureExtractor(audio_file)
            chroma = extractor.extract_cqt()

            state_representation = StateRepresentation()
            states = state_representation.cqt_to_state(chroma)
            all_states.append(states)

        states_dataset = np.hstack(all_states)
        X = np.column_stack([states_dataset]).reshape(-1, 1)

        self.model = hmm.MultinomialHMM(n_components=self.n_states, n_iter=self.n_iter)
        self.model.fit(X)

    def decode(self, audio_file):
        if self.model is None:
            raise ValueError("You must train the model before decoding.")

        extractor = FeatureExtractor(audio_file)
        chroma = extractor.extract_chroma()

        state_representation = StateRepresentation()
        states = state_representation.cqt_to_state(chroma)
        X = np.column_stack([states]).reshape(-1, 1)

        _, most_likely_states = self.model.decode(X)
        most_likely_midi_notes = [state_representation.state_to_pitch(state) for state in most_likely_states]

        return most_likely_midi_notes

    def convert(self, input_audio_file, output_midi_file):
        most_likely_midi_notes = self.decode(input_audio_file)
        midi_writer = MIDIWriter(most_likely_midi_notes)
        midi_writer.save(output_midi_file)