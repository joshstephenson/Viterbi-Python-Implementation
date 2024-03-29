#!/usr/bin/env python

# algorithm adapted from:
# - https://ben.bolte.cc/viterbi
# With data from:
# - Speech and Language Processing
# - Dan Jurafsky and James H. Martin
# - Chapter 8
# - copyright 2021

import numpy as np
from typing import List, Tuple


class States:
    def __init__(self, filename):
        self.filename = filename
        with open(filename) as f:
            state_lines = [line.replace('\n', '').split(',') for line in f.readlines()]

        self.state_lookup = dict((l.upper(), i) for i, l in enumerate(state_lines.pop(0)[1:]))
        self.reverse_lookup = dict((v, k) for k, v in self.state_lookup.items())
        self.state_probabilities = dict((tr[0], [float(t) for t in tr[1:]]) for tr in state_lines)
        self.matrix = np.array([v for v in self.state_probabilities.values()])

    def __getitem__(self, key):
        return self.state_probabilities[key]

    def states_for_pos(self, current, last):
        if type(current) is str:
            current = current.upper()
            current_idx = self.state_lookup[current]
        else:
            current_idx = current
        last = last.upper()
        return self.state_probabilities[last][current_idx]

    def col(self, col):
        if type(col) is str:
            col = self.state_lookup[col]
        return self.matrix[:, col]


class Viterbi:
    def __init__(self):
        self.transitions = States('transitions.csv')
        self.emissions = States('emissions.csv')

    def time_step(self, observed_state, mu = None):
        if mu is None:
            emission = self.emissions.col(observed_state)
            initial = self.transitions['<start>']
            mu = initial * emission
            max_idx = None
        else:
            state_probabilities = mu * self.transitions.matrix[1:].T
            max_idx = np.argmax(state_probabilities, axis=1)
            max_values = state_probabilities[np.arange(len(max_idx)), max_idx]  # get the max values
            mu = max_values * self.emissions.col(observed_state)  # multiply max values by emission probabilities
        return mu, max_idx

    def trace_backwards(self, mu, state_matrix):
        state = np.argmax(mu)  # start at right-hand side max value
        sequence_prob = mu[state]
        state_sequence = [self.transitions.reverse_lookup[state]]
        for states in state_matrix[::-1]:  # reverse iterate
            state = states[state]
            state_sequence.append(self.transitions.reverse_lookup[state])

        return state_sequence[::-1], sequence_prob

    def run(self, string: str) -> Tuple[List[str], float]:
        """Runs the Viterbi algorithm to get the most likely state sequence.
        :param string: the string with observations to parse for parts-of-speech
        :returns: tuple of (most likely series of states, joint probability of that series)
        """

        observed_states = [s.upper() for s in string.split()]
        mu, _ = self.time_step(observed_states.pop(0))
        state_matrix = []
        # runs the forward pass, storing the most likely previous state at each step
        for observed_state in observed_states:
            mu, max_idx = self.time_step(observed_state, mu)
            state_matrix.append(max_idx)

        # Traces backwards to get the maximum likelihood sequence.
        return self.trace_backwards(mu, state_matrix)


model = Viterbi()
path, prob = model.run('Janet will back the bill')
print(f'   Max Sequence: {path}')
print(f'Seq Probability: {prob}')
