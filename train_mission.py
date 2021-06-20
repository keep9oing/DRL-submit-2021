import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import os

from solver import solver_Attention
from Dataset import Mission
from utils import get_ref_reward


parser = argparse.ArgumentParser()

parser.add_argument("--coverage_num", type=int, default=4)
parser.add_argument("--visiting_num", type=int, default=4)
parser.add_argument("--pick_place_num", type=int, default=4)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--num_tr_dataset", type=int, default=10000)
parser.add_argument("--num_te_dataset", type=int, default=2000)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--use_cuda", type=bool, default=True)
parser.add_argument("--beta", type=float, default=0.9)
args = parser.parse_args()


if __name__ =="__main__":
    if args.use_cuda:
        use_pin_memory = True
    else:
        use_pin_memory = False

    print("GENERATING TRAINING DATASET")
    train_dataset = Mission.MissionDataset(num_visit=args.visiting_num, num_coverage=args.coverage_num, num_pick_place=args.pick_place_num,
                                           num_samples=args.num_tr_dataset,
                                           random_seed=100)
    print("GENERATING EVALUATION DATASET")
    eval_dataset = Mission.MissionDataset(num_visit=args.visiting_num, num_coverage=args.coverage_num, num_pick_place=args.pick_place_num,
                                          num_samples=args.num_te_dataset,
                                          random_seed=200)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory)

    eval_loader = DataLoader(eval_dataset,
                             batch_size=args.num_te_dataset,
                             shuffle=False)

    print("CALCULATING HEURISTICS")
    heuristic_distance = torch.zeros(args.num_te_dataset)
    for i, pointset in tqdm(eval_dataset):
        heuristic_distance[i], _, _ = get_ref_reward(pointset)

    # Select agent model type
    model = solver_Attention(
        args.embedding_size,
        args.hidden_size,
        args.visiting_num+args.coverage_num+args.pick_place_num+1,
        2, 10)

    if args.use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=3.0 * 1e-4)

    # Train loop
    moving_avg = torch.zeros(args.num_tr_dataset)
    if args.use_cuda:
        moving_avg = moving_avg.cuda()

    #generating first baseline
    print("GENERATING FIRST BASELINE")
    for (indices, sample_batch) in tqdm(train_data_loader):
        if args.use_cuda:
            sample_batch = sample_batch.cuda()
        rewards, _, _ = model(sample_batch)
        moving_avg[indices] = rewards

    #Training
    print("TRAINING START")
    model.train()
    for epoch in tqdm(range(args.num_epochs)):
        for batch_idx, (indices, sample_batch) in enumerate(train_data_loader):
            if args.use_cuda:
                sample_batch = sample_batch.cuda()
            rewards, log_probs, action = model(sample_batch)
            moving_avg[indices] = moving_avg[indices] * args.beta + rewards * (1.0 - args.beta)
            advantage = rewards - moving_avg[indices]
            log_probs = torch.sum(log_probs, dim=-1)
            log_probs[log_probs < -100] = -100
            loss = (advantage * log_probs).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()


        model.eval()
        ret = []
        for i, batch in eval_loader:
            if args.use_cuda:
                batch = batch.cuda()
            R, _, _ = model(batch)

        R = R.cpu()
        print("[at epoch %d]RL model generates solution with %0.2f %% gap between heuristic solution." %(
            epoch,
            ((R - heuristic_distance)/heuristic_distance*100).mean().detach().numpy()))
        print("AVG R", R.mean().detach().numpy())
        model.train()

    print("SAVE MODEL")
    os.makedirs('./Models', exist_ok=True)
    save_path = "./Models/simple_model_C%d_V%d_D%d.pth" %(args.coverage_num, args.visiting_num, args.pick_place_num)
    torch.save(model, save_path)
