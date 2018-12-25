import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import torch.optim as optim
from model.model import VideoLocalization
from gensim.models import KeyedVectors
from dataloader import Loader
from torch.utils.data import Dataset, DataLoader
from utils import get_log

def train(config):
    logger = get_log("log/log.txt")
    device = torch.device('cuda')
    word2vec = KeyedVectors.load_word2vec_format(config["word2vec"], binary=True)
    train_dataset = Loader(config, config['train_data'], word2vec)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    model = VideoLocalization(stack_num=5, video_dim=500, text_dim=300).to(device)
    model.train(True)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=0.001)

    for epoch in range(100):
        iter_cnt = 0
        loss_total = 0
        for i, data_iter in enumerate(train_loader):
            
            model.zero_grad()
            frame_vecs, frame_mask, ques_vecs, ques_mask, starts, ends = data_iter
            loss = model(frame_vecs.to(device), ques_vecs.to(device), frame_mask.to(device), ques_mask.to(device), starts.to(device), ends.to(device))
            loss.backward()
            optimizer.step()
            
            logger.info("[epoch:{}] [step:{}] loss:{}".format(epoch, i, loss.item()))


if __name__ == '__main__':
    config= {
        "learning_rate": 1e-3,
        "lr_decay_n_iters": 3000,
        "lr_decay_rate": 0.8,
        "max_epoches": 10,
        "early_stopping": 5,
        "cache_dir": "../results/baseline/",
        "display_batch_interval": 100,
        "evaluate_interval": 5,

        "regularization_beta": 1e-7,
        "dropout_prob": 0.9,

        "batch_size": 64,

        "input_video_dim": 500,
        "max_frames": 384,
        "input_ques_dim": 300,
        "max_words": 20,
        "hidden_size": 512,

        "word2vec": "data/word2vec.bin",

        "is_origin_dataset": True,
        "train_json": "data/activity-net/train.json",
        "val_json": "data/activity-net/val_1.json",
        "test_json": "data/activity-net/val_2.json",
        "train_data": "data/activity-net/train_data.json",
        "val_data": "data/activity-net/val_data.json",
        "test_data": "data/activity-net/test_data.json",
        "feature_path": "data/activity-c3d",
        "feature_path_tsn": "data/tsn_score"
    }
    train(config)