import random
import torch

## data loading and mixing logic

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)
    
    def split(self, test_ratio: float = 0.2):
        """
        Split this dataset into train and test sets while maintaining the same mixing ratio
        
        Args:
            test_ratio: Fraction of data to use for testing
            
        Returns:
            train_dataset, test_dataset: Two MixedSequenceDataset objects
        """
        # Split TinyStories sequences
        tiny_train = []
        tiny_test = []
        if self.has_tinystories:
            indices = list(range(len(self.tinystories_seqs)))
            random.shuffle(indices)
            split_idx = int(len(indices) * (1 - test_ratio))
            
            tiny_train = [self.tinystories_seqs[i] for i in indices[:split_idx]]
            tiny_test = [self.tinystories_seqs[i] for i in indices[split_idx:]]
        
        # Split other sequences
        other_train = []
        other_test = []
        if self.has_other:
            indices = list(range(len(self.other_seqs)))
            random.shuffle(indices)
            split_idx = int(len(indices) * (1 - test_ratio))
            
            other_train = [self.other_seqs[i] for i in indices[:split_idx]]
            other_test = [self.other_seqs[i] for i in indices[split_idx:]]
        
        # Create new datasets sharing the same p_tiny
        train_dataset = MixedSequenceDataset(tiny_train, other_train, self.p_tiny)
        test_dataset = MixedSequenceDataset(tiny_test, other_test, self.p_tiny)
        
        return train_dataset, test_dataset
