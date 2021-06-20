# DRL-submit-2021

# Mission planning with heterogeneous tasks
This work is for building UAV mission planning environment and train solution via reinforcement learning([Pointer Network + REINFORCE](https://arxiv.org/abs/1611.09940))

Tested Heterogeneous tasks are coveraging, visiting and pick&place.

[LKH3.0](http://webhotel4.ruc.dk/~keld/research/LKH-3/) is used to compare the performance of the model.


## Dependency
### Training environment of author
* OS: Ubuntu 20.04
* GPU: Geforce RTX 2080
### python library
* pytorch==1.8.1
* numpy==1.20.3
* tqdm==4.60.0
* matplotlib==3.4.1

## Execution
### Train model
Any arguments(The number of each task, batch size, size of dataset etc.) in the train_mission.py file are available.
```bash
python train_mission.py --num_epochs 100
```
### Test performance of the model
Calculate performance gap between the model and the LKH3.0
```bash
python test_performance --model_path="Models/simple_model_C4_V4_D4.pth" --num_te_dataset 1000
```
### Visualize sample solutions
Visualize the result of one sample
```bash
python sample_plot.py --model_path="Models/simple_model_C4_V4_D4.pth" --seed 456
```
그림

참조
