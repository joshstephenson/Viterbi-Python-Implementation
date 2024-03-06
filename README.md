# Viterbi Algorithm with Parts-of-Speech Example data

The __Viterbi__ algorithm is a common algorithm for parsing parts of speech given you have a matrix of transition states and emission states (see `transitions.csv` and `emissions.csv`).

The transition matrix defines the probability values for transitioning from one part of speech to another. The column is the `t` value, the row is the `t-1` value.

In this example:

 | Abbreviation | Part of Speech |
 |--------------|----------------|
 | DT | Determiner |
 | RB | Adverb |
 | NN | Noun |
 | JJ | Adjective |
 | VB | Verb | 
 | MD | Modal|
 | NNP | Noun Phrase |

How to run:
```
model = Viterbi()
path, prob = model.run('Janet will back the bill')
```
Output:
```
   Max Sequence: ['NNP', 'MD', 'VB', 'DT', 'NN']
Seq Probability: 2.013570710221386e-15
```
