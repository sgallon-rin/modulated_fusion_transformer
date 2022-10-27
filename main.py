import argparse, os, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloaders import *
from models import *
from train import train
from utils.compute_args import compute_args


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model', type=str, default="Model_MAT", choices=["Model_MAT", "Model_MNT", "Model_MIST"])
    parser.add_argument('--layer', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--dropout_i', type=float, default=0.0)
    parser.add_argument('--dropout_r', type=float, default=0.1)
    parser.add_argument('--dropout_o', type=float, default=0.5)
    parser.add_argument('--multi_head', type=int, default=8)
    parser.add_argument('--ff_size', type=int, default=2048)
    parser.add_argument('--word_embed_size', type=int, default=300)
    parser.add_argument('--bidirectional', type=bool, default=False)
    # MISA config
    parser.add_argument('--use_bert', type=str2bool, default=True)
    parser.add_argument('--use_cmd_sim', type=str2bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rnncell', type=str, default='lstm')

    # Data
    parser.add_argument('--lang_seq_len', type=int, default=50)
    parser.add_argument('--audio_seq_len', type=int, default=50)
    parser.add_argument('--video_seq_len', type=int, default=60)
    parser.add_argument('--audio_feat_size', type=int, default=80)
    parser.add_argument('--video_feat_size', type=int, default=512)

    # Training
    parser.add_argument('--output', type=str, default='ckpt/')
    parser.add_argument('--name', type=str, default='exp0/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=99)
    parser.add_argument('--lr_base', type=float, default=0.00005)
    parser.add_argument('--lr_decay', type=float, default=0.2)
    parser.add_argument('--lr_decay_times', type=int, default=2)
    parser.add_argument('--grad_norm_clip', type=float, default=-1)
    parser.add_argument('--eval_start', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=3)
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999999))

    # Dataset and task
    parser.add_argument('--dataset', type=str, choices=['MELD', 'MOSEI', 'MOSI', 'IEMOCAP', 'VGAF'], default='MOSEI')
    parser.add_argument('--task', type=str, choices=['sentiment', 'emotion'], default='sentiment')
    parser.add_argument('--task_binary', type=bool, default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Base on args given, compute new args
    args = compute_args(parse_args())

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DataLoader
    train_dset = eval(args.dataloader)('train', args)
    eval_dset = eval(args.dataloader)('valid', args, train_dset.token_to_ix)

    train_loader = DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    eval_loader = DataLoader(eval_dset, args.batch_size, num_workers=8, pin_memory=True)

    # Net
    net = eval(args.model)(args, train_dset.vocab_size, train_dset.pretrained_emb).cuda()
    print("Total number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e6) + "M")
    net = net.cuda()

    # Create Checkpoint dir
    if not os.path.exists(os.path.join(args.output, args.name)):
        os.makedirs(os.path.join(args.output, args.name))

    # Run training
    eval_accuracies = train(net, train_loader, eval_loader, args)
    open('best_scores.txt', 'a+').write(args.output + "/" + args.name + ","
                                        + str(max(eval_accuracies)) + "\n")