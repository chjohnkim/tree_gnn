# *"Towards Robotic Tree Manipulation"* : Leveraging Graph Representations

The goal for this project is to learn the complex physics model of trees. More specifically, we want to predict the deformation of trees when an external contact is applied. Obtaining a more accurate model of tree is important for the robot, as it can lead to safer and more robust manipulation in agriculture. For this purpose, we use recent advancements in Graph Neural Networks and take advantage of graph-like tree structures to learn and predict the dynamics of tree deformation. We share our custom collected synthetic dataset as well as our codebase. 

Visit the [project website](https://kantor-lab.github.io/tree_gnn/) for more information and videos.
Our paper can be found [here](https://arxiv.org/abs/2311.07479). 

## 1. Dependencies

- [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- networkx
- torch_scatter
- klampt
- pykin
- urdfpy
- matplotlib
- tqdm
- wandb 

## 2. Downloading the Dataset
The dataset for training the forward model and contact policy can be downloaded from this [link](https://drive.google.com/drive/folders/1R2wOhUQV8XjORfNR_Q-nOLRTagR9kz3S?usp=drive_link).

## 3. Training and Testing 
To train a model, edit ```cfg/train_cfg.yaml``` based on where your dataset is and what model you are training. Then train your model by running one of the following scripts:
```
python train_forward_model.py
python train_contact_policy.py
```

Once the model is trained, you can test your model by setting the data and model weights path in the ```cfg/test_cfg.yaml```. You can then run one of the following scripts:
1. To test the model:
```
python test_forward_model.py
python test_contact_policy.py
```
2. To visualize the model in Isaac Gym:
```
python visualize_forward_model.py
python visualize_contact_policy.py
python visualize_robot_policy.py
```
3. To evaluate the models:
```
python plot_forward_model.py
python plot_contact_policy.py
python plot_robot_policy.py
python analyze_contact_policy.py
```


## Code contributed and maintained by:
- John Kim: [chunghek@andrew.cmu.edu]()
