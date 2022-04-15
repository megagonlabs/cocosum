from typing import List

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CoCoTrip(Dataset):
    def __init__(self,
                 data: List[tuple],
                 pad_token_id: int = 0):
        self.data = data
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        input_ids, token_type_ids, counter_ids, counter_type_ids, output_ids, ins = zip(*batch)
        pad_token_id = self.pad_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=pad_token_id)
        counter_ids = pad_sequence(counter_ids, batch_first=True, padding_value=pad_token_id)
        counter_type_ids = pad_sequence(counter_type_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, token_type_ids, counter_ids, counter_type_ids, output_ids, ins
