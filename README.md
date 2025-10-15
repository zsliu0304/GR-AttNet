# GR-AttNet
GR-AttNet: Robotic Grasping with Lightweight Spatial Attention Mechanism

## Dataset

Download and extract the Cornell Grasping dataset and run the following command: 
-   [Cornell Grasping Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php)
-   [Jacquard Dataset](https://jacquard.liris.cnrs.fr/)
-   [Multi-cornell Dataset](https://github.com/ivalab/grasp_multiObject)

The Cornell dataset needs to be processed.
```
python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>

````

## Environment
- Clone this repository and install required libraries
```
git clone https://github.com/zsliu0304/GR-AttNet.git

```
-Please set up your environment according to GR-CNN(https://github.com/skumra/robotic-grasping)

## Training
```
python train.py
````

## Testing
```
python test.py
````





## References
1. [Antipodal Robotic Grasping using GR-ConvNet](https://github.com/skumra/robotic-grasping)
2. [GR-ConvNet-grasping](https://github.com/Loahit5101/GR-ConvNet-grasping/blob/main/README.md)
3. [Multi-cornell](https://github.com/ivalab/grasp_multiObject_multiGrasp)
4. [Pytorch-TensorRT Tutorials](https://github.com/pytorch/TensorRT/tree/master/examples)
