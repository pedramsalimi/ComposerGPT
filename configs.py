"""
Config script for the project contatins all the hyperparameters
"""
from argparse import Namespace
import argparse
# include all parameters for create a dataset like batch size and all parameters for training like epochs and learning rate and all parameters for the model like number of layers and all parameters for the optimizer like weight decay and all parameters for the scheduler like warmup steps and all parameters for the loss function like ignore index 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", default=512, type=int, help="block size")
    parser.add_argument("--file_path", default="/workspaces/codespaces-jupyter/lyrics.csv", type=str, help="file path")
    parser.add_argument("--output_dir", default="output", type=str, help="output directory")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--epochs", default=10, type=int, help="number of epochs")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="warmup steps")
    
    return parser.parse_args()

# create a namespace function as well for all the above parameters
def get_namespace():
    return Namespace(
        block_size=512,
        file_path="lyrics.csv",
        output_dir="output",
        batch_size=16,
        epochs=10,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=1000
    )
