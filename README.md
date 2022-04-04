
# Learning Transferable 3D Adversarial Cloaks for Deep Trained Detectors #

## Prerequisites
Our human meshes, background images, and yolov2 model file can be downloaded from:
https://mega.nz/file/pZtUCKza#6AF3AkIYxiWXysqoo78nbjKoTCos6-PwU_UBaSntIA8

After extracting the .zip file, your directory should contain
```
./data/background
./data/meshes
./data/test_background
./data/yolov2
```

PyTorch 1.8.0 and Torchvision 0.8.1 (tested in python 3.8):
```
pip install torch torchvision
```
PyTorch3D v0.2.5:
```
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.2.5'
```

To train/test on Faster R-CNN:
```
git clone https://github.com/potterhsu/easy-faster-rcnn.pytorch.git faster_rcnn
cd faster_rcnn
python support/setup.py develop
```
Faster R-CNN pretrained checkpoint can be found here:
https://github.com/potterhsu/easy-faster-rcnn.pytorch

Move the `model-180000.pth` checkpoint to `faster_rcnn/model-180000.pth`.

## Training
To train an adversarial patch:
```
python train.py --mesh_dir=data/meshes --epochs=100 --num_bgs=1024 --num_test_bgs=1024 --batch_size=12 --num_angles_train=1 --angle_range_train=0 --num_angles_test=21 --angle_range_test=10 --idx=idx/chest_legs1.idx --detector=yolov2 --patch_dir=example_logos/fasterrcnn_chest_legs
```
During training, the script will save the adversarial texture atlas and face indices to `patch_save.pt` and `idx_save.pt` respectively. The `--idx` argument specifies the indices of the patch. To generate new .idx files, see the Blender script `face_sampler.py`.

## Testing
To test a patch:
```
python train.py --test_only --mesh_dir=data/humans --num_test_bgs=1024 --num_angles_test=21 --angle_range_test=10 --patch_dir=my_patch --detector=yolov2
```
The `--patch_dir` argument specifies a folder that contains the `patch_save.pt` and `idx_save.pt` files. 
