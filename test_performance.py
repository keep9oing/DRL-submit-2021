import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import Mission
from utils import get_ref_reward


parser = argparse.ArgumentParser()

parser.add_argument("--coverage_num", type=int, default=4)
parser.add_argument("--visiting_num", type=int, default=4)
parser.add_argument("--pick_place_num", type=int, default=4)
parser.add_argument("--num_te_dataset", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--use_cuda", type=bool, default=True)
parser.add_argument("--model_path", type=str, default=None)

args = parser.parse_args()


if args.use_cuda:
  use_pin_memory = True
else:
  use_pin_memory = False

print("Model:", args.model_path)
assert args.model_path is not None, "model_path should be specified"

load_path = args.model_path
model = torch.load(load_path)

model.eval()
test_dataset = Mission.MissionDataset(num_visit=args.visiting_num, num_coverage=args.coverage_num, num_pick_place=args.pick_place_num,
                                      num_samples=args.num_te_dataset,
                                      random_seed=332)

test_data_loader = DataLoader(
  test_dataset,
  batch_size = args.num_te_dataset,
  shuffle=True,
  pin_memory=use_pin_memory)

print("CALCULATING HEURISTICS")
heuristic_distance = torch.zeros(args.num_te_dataset)
for i, pointset in tqdm(test_dataset):
  heuristic_distance[i], _, lkh_solution = get_ref_reward(pointset)

for i, mission in test_data_loader:
  if args.use_cuda:
      mission = mission.cuda()

  cost, _, solution = model(mission)

cost = cost.cpu()

print("solution cost:", cost.mean().detach().numpy())
print("LKH cost:", heuristic_distance.mean().detach().numpy())

print("Performance gap:", ((cost - heuristic_distance)/heuristic_distance*100).mean().detach().numpy(), "%")
