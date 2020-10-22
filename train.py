import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from skimage.io import imread

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    OpenGLPerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    BlendParams,
    SoftSilhouetteShader
)

import sys
import os

from MeshDataset import MeshDataset
from BackgroundDataset import BackgroundDataset
from darknet import Darknet
from loss import TotalVariation, dis_loss, calc_acc

from torchvision.utils import save_image

class Patch():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Create pytorch3D renderer
        self.renderer = self.create_renderer()

        # Datasets
        self.mesh_dataset = MeshDataset(config.mesh_dir, device)
        self.bg_dataset = BackgroundDataset(config.bg_dir, config.img_size, max_num=config.num_bgs)
        self.test_bg_dataset = BackgroundDataset(config.test_bg_dir, config.img_size, max_num=config.num_test_bgs)

        # Initialize adversarial patch, and TV loss
        self.patch = torch.rand((100, 100, 3), device=device, requires_grad=True)
        self.total_variation = TotalVariation().to(device)

        # Yolo model:
        self.dnet = Darknet(self.config.cfgfile)
        self.dnet.load_weights(self.config.weightfile)
        self.dnet = self.dnet.eval()
        self.dnet = self.dnet.to(self.device)
        
    def attack(self):
        train_bgs = DataLoader(
            self.bg_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=1)

        optimizer = torch.optim.SGD([self.patch], lr=1.0, momentum=0.9)

        for epoch in range(self.config.epochs):
            ep_loss = 0.0
            ep_acc = 0.0
            n = 0.0

            for mesh in self.mesh_dataset:
                # Copy mesh for each camera angle
                mesh = mesh.extend(self.num_angles)
                mesh_texture = mesh.textures.maps_padded()

                for bg_batch in train_bgs:
                    bg_batch = bg_batch.to(self.device)

                    optimizer.zero_grad()
                    
                    # Apply patch to mesh texture (hard coded for now)
                    mesh_texture[:, 575:675, 475:575, :] = self.patch[None]

                    # Render mesh onto background image
                    # images = self.render_mesh_on_bg(mesh, bg)
                    images = self.render_mesh_on_bg_batch(mesh, bg_batch)
                    # print('images: ', images.shape)
                    reshape_img = images[:,:,:,:3].permute(0, 3, 1, 2)
                    reshape_img = reshape_img.to(self.device)

                    # Run detection model on images
                    output = self.dnet(reshape_img)

                    # Compute losses:
                    d_loss = dis_loss(output, self.dnet.num_classes, self.dnet.anchors, self.dnet.num_anchors, 0)
                    acc_loss = calc_acc(output, self.dnet.num_classes, self.dnet.num_anchors, 0)

                    tv = self.total_variation(self.patch)
                    tv_loss = tv * 2.5
                    
                    loss = d_loss + torch.sum(torch.max(tv_loss, torch.tensor(0.1).to(self.device)))

                    ep_loss += loss.item()
                    ep_acc += acc_loss.item()
                    
                    n += bg_batch.shape[0]

                    # TODO: need to remove retain_graph
                    loss.backward(retain_graph=True)
                    optimizer.step()
            
            # Save image and print performance statistics
            save_image(self.patch.cpu().detach().permute(2, 0, 1), self.config.output + '_{}.png'.format(epoch))
            print('epoch={} loss={} success_rate={}'.format(
              epoch, 
              (ep_loss / n), 
              (ep_acc / n) / self.num_angles)
            )

            self.test_patch()
    
    def test_patch(self):
        angle_success = torch.zeros(self.num_angles)
        total_loss = 0.0
        n = 0.0
        for mesh in self.mesh_dataset:
            mesh = mesh.extend(self.num_angles)
            mesh_texture = mesh.textures.maps_padded()
            for bg in self.test_bg_dataset:
                
                mesh_texture[:, 575:675, 475:575, :] = self.patch[None]

                images = self.render_mesh_on_bg(mesh, bg)
                reshape_img = images[:,:,:,:3].permute(0, 3, 1, 2)
                reshape_img = reshape_img.to(self.device)
                output = self.dnet(reshape_img)

                d_loss = dis_loss(output, self.dnet.num_classes, self.dnet.anchors, self.dnet.num_anchors, 0)

                for angle in range(self.num_angles):
                    acc_loss = calc_acc(output[angle], self.dnet.num_classes, self.dnet.num_anchors, 0)
                    angle_success[angle] += acc_loss.item()

                tv = self.total_variation(self.patch)
                tv_loss = tv * 2.5
                
                loss = d_loss + torch.sum(torch.max(tv_loss, torch.tensor(0.1).to(self.device)))

                total_loss += loss.item()
                n += 1.0
        
        unseen_success_rate = angle_success.mean() / len(self.test_bg_dataset)
        print('Unseen bg success rate: ', unseen_success_rate.item())

    def create_renderer(self):
        self.num_angles = self.config.num_angles
        azim = torch.linspace(-1 * self.config.angle_range, self.config.angle_range, self.num_angles)

        R, T = look_at_view_transform(dist=1.0, elev=0, azim=azim)

        T[:, 1] = -85
        T[:, 2] = 200

        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        
        raster_settings = RasterizationSettings(
            image_size=self.config.img_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        lights = PointLights(device=self.device, location=[[0.0, 85, 100.0]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights
            )
        )
        return renderer

    def render_mesh_on_bg(self, mesh, bg_img, location=None, x_translation=0, y_translation=0):
        images = self.renderer(mesh)
        bg = bg_img.unsqueeze(0)
        bg_shape = bg.shape
        new_bg = torch.zeros(bg_shape[2], bg_shape[3], 3)
        new_bg[:,:,0] = bg[0,0,:,:]
        new_bg[:,:,1] = bg[0,1,:,:]
        new_bg[:,:,2] = bg[0,2,:,:]

        human = images[:, ..., :3]
        
        human_size = self.renderer.rasterizer.raster_settings.image_size

        if location is None:
            dH = bg_shape[2] - human_size
            dW = bg_shape[3] - human_size
            location = (
                dW // 2 + x_translation,
                dW - (dW // 2) - x_translation,
                dH // 2 + y_translation,
                dH - (dH // 2) - y_translation
            )

        contour = torch.where((human == 1).cpu(), torch.zeros(1).cpu(), torch.ones(1).cpu())
        new_contour = torch.zeros(self.num_angles, bg_shape[2], bg_shape[3], 3)
        
        new_contour[:,:,:,0] = F.pad(contour[:,:,:,0], location, "constant", value=0)
        new_contour[:,:,:,1] = F.pad(contour[:,:,:,1], location, "constant", value=0)
        new_contour[:,:,:,2] = F.pad(contour[:,:,:,2], location, "constant", value=0)

        new_human = torch.zeros(self.num_angles, bg_shape[2], bg_shape[3], 3)
        new_human[:,:,:,0] = F.pad(human[:,:,:,0], location, "constant", value=0)
        new_human[:,:,:,1] = F.pad(human[:,:,:,1], location, "constant", value=0)
        new_human[:,:,:,2] = F.pad(human[:,:,:,2], location, "constant", value=0)

        final = torch.where((new_contour == 0).cpu(), new_bg.cpu(), new_human.cpu())
        return final

    def render_mesh_on_bg_batch(self, mesh, bg_imgs, location=None, x_translation=0, y_translation=0):
        num_bgs = bg_imgs.shape[0]

        images = self.renderer(mesh) # (num_angles, 416, 416, 4)
        images = torch.cat(num_bgs*[images], dim=0) # (num_angles * num_bgs, 416, 416, 4)

        bg_shape = bg_imgs.shape

        # bg_imgs: (num_bgs, 3, 416, 416) -> (num_bgs, 416, 416, 3)
        bg_imgs = bg_imgs.permute(0, 2, 3, 1)

        # bg_imgs: (num_bgs, 416, 416, 3) -> (num_bgs * num_angles, 416, 416, 3)
        bg_imgs = bg_imgs.repeat_interleave(repeats=self.num_angles, dim=0)

        # human: RGB channels of render (num_angles * num_bgs, 416, 416, 3)
        human = images[:, ..., :3]
        human_size = self.renderer.rasterizer.raster_settings.image_size

        if location is None:
            dH = bg_shape[2] - human_size
            dW = bg_shape[3] - human_size
            location = (
                dW // 2 + x_translation,
                dW - (dW // 2) - x_translation,
                dH // 2 + y_translation,
                dH - (dH // 2) - y_translation
            )

        contour = torch.where((human == 1), torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        new_contour = torch.zeros(self.num_angles * num_bgs, bg_shape[2], bg_shape[3], 3, device=self.device)
        
        new_contour[:,:,:,0] = F.pad(contour[:,:,:,0], location, "constant", value=0)
        new_contour[:,:,:,1] = F.pad(contour[:,:,:,1], location, "constant", value=0)
        new_contour[:,:,:,2] = F.pad(contour[:,:,:,2], location, "constant", value=0)

        new_human = torch.zeros(self.num_angles * num_bgs, bg_shape[2], bg_shape[3], 3, device=self.device)
        new_human[:,:,:,0] = F.pad(human[:,:,:,0], location, "constant", value=0)
        new_human[:,:,:,1] = F.pad(human[:,:,:,1], location, "constant", value=0)
        new_human[:,:,:,2] = F.pad(human[:,:,:,2], location, "constant", value=0)

        # output: (num_angles * num_bgs, 416, 416, 3)
        final = torch.where((new_contour == 0), bg_imgs, new_human)
        return final

def main():
    import argparse
    parser = argparse.ArgumentParser()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--mesh_dir', type=str, default='data/meshes')
    parser.add_argument('--bg_dir', type=str, default='data/background')
    parser.add_argument('--test_bg_dir', type=str, default='data/test_background')
    parser.add_argument('--output', type=str, default='out/patch')
    
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--num_bgs', type=int, default=10)
    parser.add_argument('--num_test_bgs', type=int, default=2)
    parser.add_argument('--num_angles', type=int, default=21)
    parser.add_argument('--angle_range', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--cfgfile', type=str, default="cfg/yolo.cfg")
    parser.add_argument('--weightfile', type=str, default="weights/yolo.weights")
    
    config = parser.parse_args()
    trainer = Patch(config, device)

    trainer.attack()

if __name__ == '__main__':
    main()