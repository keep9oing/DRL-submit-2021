import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
import random
from tqdm import tqdm


class MissionDataset_VCP(Dataset):

    def __init__(self, num_visit, num_coverage, num_pick_place, num_samples, random_seed=111):
        super(MissionDataset_VCP, self).__init__()
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.data_set = []

        for l in tqdm(range(num_samples)):
            task_list = []

            # Generate area coverage task
            if num_coverage > 0:
                area_list = []
                while len(area_list) < num_coverage:
                    point = torch.FloatTensor(np.random.uniform(0.1, 0.9, 2))
                    area_info = torch.FloatTensor(np.random.uniform(0.05, 0.08, 1))
                    end_point = point.clone()

                    # Non-overlapping
                    if any(np.linalg.norm(point-A[:2]) < area_info + A[-3] + 0.05 for A in area_list):
                        continue

                    data = torch.cat((point, area_info, end_point), axis=0)

                    area_list.append(data)

                task_list = task_list + area_list

            # Generate point visitation task
            if num_visit > 0:
                visit_list = []
                while len(visit_list) < num_visit:
                    point = torch.FloatTensor(np.random.uniform(0.1, 0.99, 2))
                    area_info = torch.zeros(1)
                    end_point = point.clone()

                    # Non-overlapping
                    if num_coverage > 0:
                        if any(np.linalg.norm(point - A[:2]) < A[-3] + 0.01 for A in area_list):
                            continue

                    data = torch.cat((point, area_info, end_point), axis=0)

                    visit_list.append(data)

                task_list = task_list + visit_list

            # Generate Pick&Place task
            if num_pick_place > 0:
                pick_place_list = []
                while len(pick_place_list) < num_pick_place:
                    pick_point = torch.FloatTensor(np.random.uniform(0.1,0.99,2))
                    place_point = pick_point + [(-1)**random.randint(0,1), (-1)**random.randint(0,1)]*np.random.uniform(0.02, 0.09, 2)
                    area_info = torch.FloatTensor([np.linalg.norm(pick_point-place_point)])

                    # Non-overlapping
                    if num_coverage > 0:
                        if any((np.linalg.norm(pick_point - A[:2]) < A[-3] + 0.01) or (np.linalg.norm(place_point - A[:2]) < A[-3] + 0.01) for A in area_list):
                            continue

                    data = torch.cat((pick_point, area_info, place_point), axis=0)

                    pick_place_list.append(data)

                task_list = task_list + pick_place_list

            task_list = torch.stack(task_list)

            self.data_set.append(torch.cat((torch.zeros(1, 5), task_list),axis=0))

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx, self.data_set[idx]


class MissionDataset_VC(Dataset):

    def __init__(self, num_visit, num_coverage, num_samples, random_seed=111):
        super(MissionDataset_VC, self).__init__()
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.data_set = []

        for l in tqdm(range(num_samples)):
            # Generate area coverage task
            area_list = []
            while len(area_list) < num_coverage:
                point = torch.FloatTensor(np.random.uniform(0.1, 0.9, 2))
                area_info = torch.FloatTensor(np.random.uniform(0.05, 0.08, 1))

                # Non-overlapping
                if any(np.linalg.norm(point-A[:2]) < area_info + A[-1] + 0.05 for A in area_list):
                    continue

                data = torch.cat((point, area_info), axis=0)

                area_list.append(data)

            area_list = torch.stack(area_list)


            # Generate point visitation task
            visit_list = []
            while len(visit_list) < num_visit:
                point = torch.FloatTensor(np.random.uniform(0.1, 0.99, 2))
                area_info = torch.zeros(1)

                # Non-overlapping
                if any(np.linalg.norm(point - A[:2]) < A[-1] + 0.01 for A in area_list):
                    continue

                data = torch.cat((point, area_info), axis=0)

                visit_list.append(data)

            visit_list = torch.stack(visit_list)

            self.data_set.append(torch.cat((torch.zeros(1,3), area_list, visit_list),axis=0))

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx, self.data_set[idx]


def test(args):
    train_loader = DataLoader(MissionDataset_VCP(args.visiting_num, args.coverage_num, args.pick_place_num, 10, 12), batch_size=1, shuffle=True, num_workers=1)
    for  (batch_idx, task_list_batch) in train_loader:
        print(batch_idx, task_list_batch)
        task_plot(task_list_batch[0].cpu().numpy(), args)

if __name__ == "__main__":
    import argparse
    from Mission_plot import task_plot

    parser = argparse.ArgumentParser()
    parser.add_argument("--visiting_num", type=int, default=10)
    parser.add_argument("--coverage_num", type=int, default=2)
    parser.add_argument("--pick_place_num", type=int, default=2)
    args = parser.parse_args()

    test(args)
