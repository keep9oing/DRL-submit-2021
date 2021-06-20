# DRL-submit-2021

# Mission planning with heterogeneous tasks
This work is for building UAV mission planning environment and train solution via reinforcement learning([Pointer Network + REINFORCE](https://arxiv.org/abs/1611.09940))

Tested Heterogeneous tasks are coveraging, visiting and pick&place.

[LKH3.0](http://webhotel4.ruc.dk/~keld/research/LKH-3/) is used to compare the performance of the model.

![20210620_213734](https://user-images.githubusercontent.com/31655488/122674363-c405f800-d20f-11eb-8b40-fd6918cbbe50.png)

## Dependency
### Training environment of author
* OS: Ubuntu 20.04
* GPU: Geforce RTX 2080
### python library
* pytorch==1.8.1
* numpy==1.20.3
* tqdm==4.60.0
* matplotlib==3.4.1
### Heuristic solver
* [LKH3.0](http://webhotel4.ruc.dk/~keld/research/LKH-3/)
Executable file is needed and should be at the same folder with the codes.

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
Visualize the result of one sample. Can check various samples with different seed.
```bash
python sample_plot.py --model_path="Models/simple_model_C4_V4_D4.pth" --seed 456
```

![sdfsfd](https://user-images.githubusercontent.com/31655488/122674316-85703d80-d20f-11eb-89e9-ad481f83b22d.png)

## Reference
* https://github.com/Rintarooo/TSP_DRL_PtrNet
* https://github.com/ita9naiwa/TSP-solver-using-reinforcement-learning
* https://arxiv.org/abs/1611.09940
* https://arxiv.org/abs/1803.08475
