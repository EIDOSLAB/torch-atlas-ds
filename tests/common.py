import random
import math

SENTENCES = [
    "The sun rises in the east and sets in the west.",
    "Cats are known for their graceful movements.",
    "She enjoys reading books in her free time.",
    "A quick brown fox jumps over the lazy dog.",
    "The park was filled with children playing games.",
    "He prefers tea over coffee in the morning.",
    "The train arrived at the station right on time.",
    "Winter is the coldest season of the year."
]

def select_random_sample(num_samples):
    random.seed(117)
    samples_multiplier = math.ceil(num_samples / len(SENTENCES))
    samples = SENTENCES * samples_multiplier
    random.shuffle(samples)
    samples = samples[:num_samples]
    assert len(samples) == num_samples
    return samples
