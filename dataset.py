import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import os
import logging
import argparse
import numpy as np
from transformers import GPT2Tokenizer

logger = logging.getLogger(__name__)


### Prepare data

# create a dataset class for the gpt2 using torch
class LyricsDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we load the data into a list of examples that will contain the lyrics
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_' + str(block_size) + '_' + filename)
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            self.examples = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            self.examples = []
            lyrics = pd.read_csv(file_path)
            lyrics = lyrics.lyrics.dropna()
            text = ""
            for i in range(len(lyrics)):
              text += lyrics[i]+"\n"
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            for i in trange(0, len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size]))
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(self.examples, cached_features_file)
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)  


#main function for dataset script
def main(
    block_size=512,
    file_path=None,
    output_dir=None
):
    # load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # load the dataset
    dataset = LyricsDataset(tokenizer, file_path, block_size=512)
    # create the dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # print the first batch
    print(next(iter(dataloader)))
#create args parser of the dataset script
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", default=512, type=int, help="block size")
    parser.add_argument("--file_path", default=None, type=str, help="file path")
    parser.add_argument("--output_dir", default="/content/data", type=str, help="output directory")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(
        block_size=args.block_size,
        file_path=args.file_path,
        output_dir=args.output_dir
    )


