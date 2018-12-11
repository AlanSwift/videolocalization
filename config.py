import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="video ")
    parser.add_argument("--data_dir", type=str, default="data", help="data folder")
    parser.add_argument("--log_dir", type=str, default="log", help="log folder")
    parser.add_argument("--log_filename", type=str, default="log.txt", help="log filename")
    parser.add_argument("--model_dir", type=str, default="model", help="model folder")
    parser.add_argument("--vocab_filename", type=str, default="", help="word embedding file")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--epoch", type=int, default=100, help="epoch number")
    
    return parser