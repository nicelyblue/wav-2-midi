import numpy as np

class StateRepresentation:
    def __init__(self, n_states=12, min_pitch=21, max_pitch=108):
        self.n_states = n_states
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.pitch_range = max_pitch - min_pitch + 1
        self.pitch_bins = np.linspace(min_pitch, max_pitch, n_states)

    def cqt_to_state(self, cqt):
        state_sequence = np.argmax(cqt, axis=0)
        return state_sequence    

    def pitch_to_state(self, pitch):
        return np.argmin(np.abs(self.pitch_bins - pitch))

    def state_to_pitch(self, state):
        return self.pitch_bins[state]

    def chroma_to_state(self, chroma):
        return np.argmax(chroma, axis=1)

    def state_to_chroma(self, state):
        chroma = np.zeros(self.n_states)
        chroma[state] = 1
        return chroma
