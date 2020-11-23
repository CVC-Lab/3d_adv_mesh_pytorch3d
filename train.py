import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import sys
import random

from torch.utils.data import DataLoader, Dataset
from skimage.io import imread
from PIL import Image

from pytorch3d.io import load_objs_as_meshes, load_obj
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

from MeshDataset import MeshDataset
from BackgroundDataset import BackgroundDataset
from darknet import Darknet
from loss import TotalVariation, dis_loss, calc_acc, TotalVariation_3d

from torchvision.utils import save_image
import torchvision
import random

from PIL import ImageDraw
from faster_rcnn.dataset.base import Base as DatasetBase
from faster_rcnn.backbone.base import Base as BackboneBase
from faster_rcnn.bbox import BBox
from faster_rcnn.model import Model as FasterRCNN
from faster_rcnn.roi.pooler import Pooler
from faster_rcnn.config.eval_config import EvalConfig as Config

class Patch():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Create pytorch3D renderer
        self.renderer = self.create_renderer()

        # Datasets
        self.mesh_dataset = MeshDataset(config.mesh_dir, device, max_num=config.num_meshes)
        self.bg_dataset = BackgroundDataset(config.bg_dir, config.img_size, max_num=config.num_bgs)
        self.test_bg_dataset = BackgroundDataset(config.test_bg_dir, config.img_size, max_num=config.num_test_bgs)

        # Initialize adversarial patch
        self.patch = None
        self.idx = None

        # Yolo model:
        self.dnet = Darknet(self.config.cfgfile)
        self.dnet.load_weights(self.config.weightfile)
        self.dnet = self.dnet.eval()
        self.dnet = self.dnet.to(self.device)

        if self.config.patch_dir is not None:
          self.patch = torch.load(self.config.patch_dir + '/patch_save.pt').to(self.device)
          self.idx = torch.load(self.config.patch_dir + '/idx_save.pt').to(self.device)

        self.test_bgs = DataLoader(
          self.test_bg_dataset, 
          batch_size=1, 
          shuffle=True, 
          num_workers=1)
  
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10

    def attack_faster_rcnn(self):
        path_to_checkpoint='model-180000.pth'
        dataset_name="coco2017"
        backbone_name="resnet101"
        prob_thresh=0.6

        dataset_class = DatasetBase.from_name(dataset_name)
        backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
        model = FasterRCNN(backbone, dataset_class.num_classes(), pooler_mode=Config.POOLER_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
        model.load(path_to_checkpoint)

        train_bgs = DataLoader(
            self.bg_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=1)

        if self.patch is None or self.idx is None:
          self.initialize_patch()
        
        mesh = self.mesh_dataset.meshes[0]
        total_variation = TotalVariation_3d(mesh, self.idx).to(self.device)

        optimizer = torch.optim.SGD([self.patch], lr=1e-1, momentum=0.9)

        for epoch in range(self.config.epochs):
            ep_loss = 0.0
            ep_acc = 0.0
            n = 0.0

            for mesh in self.mesh_dataset:
                # Copy mesh for each camera angle
                mesh = mesh.extend(self.num_angles_train)

                for bg_batch in train_bgs:
                    bg_batch = bg_batch.to(self.device)

                    optimizer.zero_grad()

                    texture_image = mesh.textures.atlas_padded()

                    # Random patch augmentation
                    contrast = torch.FloatTensor(1).uniform_(self.min_contrast, self.max_contrast).to(self.device)
                    brightness = torch.FloatTensor(1).uniform_(self.min_brightness, self.max_brightness).to(self.device)
                    noise = torch.FloatTensor(self.patch.shape).uniform_(-1, 1) * self.noise_factor
                    noise = noise.to(self.device)
                    augmented_patch = (self.patch * contrast) + brightness + noise

                    # Clamp patch to avoid PyTorch3D issues
                    clamped_patch = augmented_patch.clone().clamp(min=1e-6, max=0.99999)
                    mesh.textures._atlas_padded[:,self.idx,:,:,:] = clamped_patch
      
                    mesh.textures.atlas = mesh.textures._atlas_padded
                    mesh.textures._atlas_list = None

                    # Render mesh onto background image
                    rand_translation = torch.randint(
                      -self.config.rand_translation, 
                      self.config.rand_translation, 
                      (2,)
                      )

                    images = self.render_mesh_on_bg_batch(mesh, bg_batch, self.num_angles_train, x_translation=rand_translation[0].item(),
                                                          y_translation=rand_translation[1].item())
                    
                    reshape_img = images[:,:,:,:3].permute(0, 3, 1, 2)
                    reshape_img = reshape_img.to(self.device)

                    # image_tensor, scale = dataset_class.preprocess(reshape_img, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)
                    detection_bboxes, detection_classes, detection_probs, _ = \
                        model.eval().forward(reshape_img.cuda())
                    # detection_bboxes /= scale

                    kept_indices = detection_probs > prob_thresh
                    detection_bboxes = detection_bboxes[kept_indices]
                    detection_classes = detection_classes[kept_indices]
                    detection_probs = detection_probs[kept_indices]
                    human_dets = torch.where(detection_classes == 1, torch.ones(1), torch.zeros(1)).cuda()

                    disap_loss = torch.mean(human_dets * detection_probs)

                    tv = total_variation(self.patch)
                    tv_loss = tv * 2.5
                    
                    loss = disap_loss + tv_loss
                    
                    n += bg_batch.shape[0]

                    if torch.isnan(loss).item():
                      continue

                    ep_loss += loss.item()

                    loss.backward(retain_graph=True)
                    optimizer.step()
            
            # Save image and print performance statistics
            print('tv={}, dis={}'.format(tv_loss, disap_loss))
            patch_save = self.patch.cpu().detach().clone()
            idx_save = self.idx.cpu().detach().clone()
            torch.save(patch_save, 'patch_save.pt')
            torch.save(idx_save, 'idx_save.pt')
            
            print('epoch={} loss={}'.format(
              epoch, 
              (ep_loss / n)
              )
            )

            if epoch % 5 == 0:
              self.test_patch()
              self.change_cameras('train')

    def attack(self):
        train_bgs = DataLoader(
            self.bg_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=1)

        if self.patch is None or self.idx is None:
          self.initialize_patch()
        
        mesh = self.mesh_dataset.meshes[0]
        total_variation = TotalVariation_3d(mesh, self.idx).to(self.device)

        optimizer = torch.optim.SGD([self.patch], lr=1e-1, momentum=0.9)
        
        for epoch in range(self.config.epochs):
            ep_loss = 0.0
            ep_acc = 0.0
            n = 0.0

            for mesh in self.mesh_dataset:
                # Copy mesh for each camera angle
                mesh = mesh.extend(self.num_angles_train)

                for bg_batch in train_bgs:
                    bg_batch = bg_batch.to(self.device)

                    # To enable random camera distance training, uncomment this line:
                    # self.change_cameras('train', camera_dist=random.uniform(1.4, 3.0))

                    optimizer.zero_grad()

                    texture_image = mesh.textures.atlas_padded()

                    # Random patch augmentation
                    contrast = torch.FloatTensor(1).uniform_(self.min_contrast, self.max_contrast).to(self.device)
                    brightness = torch.FloatTensor(1).uniform_(self.min_brightness, self.max_brightness).to(self.device)
                    noise = torch.FloatTensor(self.patch.shape).uniform_(-1, 1) * self.noise_factor
                    noise = noise.to(self.device)
                    augmented_patch = (self.patch * contrast) + brightness + noise

                    # Clamp patch to avoid PyTorch3D issues
                    clamped_patch = augmented_patch.clone().clamp(min=1e-6, max=0.99999)
                    mesh.textures._atlas_padded[:,self.idx,:,:,:] = clamped_patch
      
                    mesh.textures.atlas = mesh.textures._atlas_padded
                    mesh.textures._atlas_list = None

                    # Render mesh onto background image
                    rand_translation = torch.randint(
                      -self.config.rand_translation, 
                      self.config.rand_translation, 
                      (2,)
                      )

                    images = self.render_mesh_on_bg_batch(mesh, bg_batch, self.num_angles_train, x_translation=rand_translation[0].item(),
                                                          y_translation=rand_translation[1].item())
                    
                    reshape_img = images[:,:,:,:3].permute(0, 3, 1, 2)
                    reshape_img = reshape_img.to(self.device)

                    # Run detection model on images
                    output = self.dnet(reshape_img)

                    d_loss = dis_loss(output, self.dnet.num_classes, self.dnet.anchors, self.dnet.num_anchors, 0)
                    acc_loss = calc_acc(output, self.dnet.num_classes, self.dnet.num_anchors, 0)

                    tv = total_variation(self.patch)
                    tv_loss = tv * 2.5
                    
                    loss = d_loss + tv_loss

                    ep_loss += loss.item()
                    ep_acc += acc_loss.item()
                    
                    n += bg_batch.shape[0]

                    loss.backward(retain_graph=True)
                    optimizer.step()
            
            # Save image and print performance statistics
            patch_save = self.patch.cpu().detach().clone()
            idx_save = self.idx.cpu().detach().clone()
            torch.save(patch_save, 'patch_save.pt')
            torch.save(idx_save, 'idx_save.pt')

            save_image(reshape_img[0].cpu().detach(), "TEST_RENDER.png")
        
            print('epoch={} loss={} success_rate={}'.format(
              epoch, 
              (ep_loss / n), 
              (ep_acc / n) / self.num_angles_train)
            )

            if epoch % 5 == 0:
              self.test_patch()
              self.change_cameras('train')
    
    def test_patch(self):
        self.change_cameras('test')
        angle_success = torch.zeros(self.num_angles_test)
        total_loss = 0.0
        n = 0.0
        for mesh in self.mesh_dataset:
            mesh = mesh.extend(self.num_angles_test)
            for bg_batch in self.test_bgs:
                bg_batch = bg_batch.to(self.device)

                texture_image=mesh.textures.atlas_padded()
                                
                clamped_patch = self.patch.clone().clamp(min=1e-6, max=0.99999)
                mesh.textures._atlas_padded[:,self.idx,:,:,:] = clamped_patch
      
                mesh.textures.atlas = mesh.textures._atlas_padded
                mesh.textures._atlas_list = None
                
                rand_translation = torch.randint(
                  -self.config.rand_translation, 
                  self.config.rand_translation, 
                  (2,)
                  )

                images = self.render_mesh_on_bg_batch(mesh, bg_batch, self.num_angles_test, x_translation=rand_translation[0].item(),
                                                y_translation=rand_translation[1].item())
                
                reshape_img = images[:,:,:,:3].permute(0, 3, 1, 2)
                reshape_img = reshape_img.to(self.device)
                output = self.dnet(reshape_img)
                
                for angle in range(self.num_angles_test):
                    acc_loss = calc_acc(output[angle], self.dnet.num_classes, self.dnet.num_anchors, 0)
                    angle_success[angle] += acc_loss.item()

                n += bg_batch.shape[0]
        
        save_image(reshape_img[0].cpu().detach(), "TEST.png")
        unseen_success_rate = torch.sum(angle_success) / (n * self.num_angles_test)
        print('Angle success rates: ', angle_success / n)
        print('Unseen bg success rate: ', unseen_success_rate.item())

    def test_patch_faster_rcnn(self, path_to_checkpoint: str, dataset_name: str, backbone_name: str, prob_thresh: float):
        dataset_class = DatasetBase.from_name(dataset_name)
        backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
        model = FasterRCNN(backbone, dataset_class.num_classes(), pooler_mode=Config.POOLER_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
        model.load(path_to_checkpoint)

        angle_success = torch.zeros(self.num_angles_test)
        total_loss = 0.0
        n = 0.0
        with torch.no_grad():
            for mesh in self.mesh_dataset:
                mesh = mesh.extend(self.num_angles_test)
                for bg_batch in self.test_bgs:
                    bg_batch = bg_batch.to(self.device)
                    
                    texture_image=mesh.textures.atlas_padded()
                    clamped_patch = self.patch.clone().clamp(min=1e-6, max=0.99999)
                    mesh.textures._atlas_padded[:,self.idx,:,:,:] = clamped_patch
          
                    mesh.textures.atlas = mesh.textures._atlas_padded
                    mesh.textures._atlas_list = None

                    rand_translation = torch.randint(
                      -self.config.rand_translation, 
                      self.config.rand_translation, 
                      (2,)
                      )

                    images = self.render_mesh_on_bg_batch(
                      mesh, 
                      bg_batch, 
                      self.num_angles_test, 
                      x_translation=rand_translation[0].item(),
                      y_translation=rand_translation[1].item()
                      )

                    reshape_img = images[:,:,:,:3].permute(0, 3, 1, 2)
                    save_image(reshape_img[0].cpu().detach(), "TEST_PRE.png")

                    for angle in range(self.num_angles_test):
                        image = torchvision.transforms.ToPILImage()(reshape_img[angle,:,:,:].cpu())
                        # image_tensor, scale = dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)
                        image_tensor = reshape_img[angle, ..., :]
                        scale = 1.0
                        save_image(image_tensor.cpu().detach(), "TEST_POST.png")

                        img = Image.open('TEST_POST.png').convert('RGB')
                        img = torchvision.transforms.ToTensor()(image)
                        image_tensor = img.cuda()

                        detection_bboxes, detection_classes, detection_probs, _ = \
                            model.eval().forward(image_tensor.unsqueeze(dim=0).cuda())
                        detection_bboxes /= scale

                        kept_indices = detection_probs > prob_thresh
                        detection_bboxes = detection_bboxes[kept_indices]
                        detection_classes = detection_classes[kept_indices]
                        detection_probs = detection_probs[kept_indices]

                        draw = ImageDraw.Draw(image)

                        for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
                            color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
                            bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                            category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]

                            draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color, width=3)
                            draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)
                        if angle==0:
                            image.save("out/images/test_%d.png" % n)

                    n += 1.0

    def initialize_patch(self):
        print('Initializing patch...')
        # Code for sampling faces:
        # mesh = self.mesh_dataset.meshes[0]
        # box = mesh.get_bounding_boxes()
        # max_x = box[0,0,1]
        # max_y = box[0,1,1]
        # max_z = box[0,2,1]
        # min_x = box[0,0,0]
        # min_y = box[0,1,0]
        # min_z = box[0,2,0]

        # len_z = max_z - min_z
        # len_x = max_x - min_x
        # len_y = max_y - min_y

        # verts = mesh.verts_padded()
        # v_shape = verts.shape
        # sampled_verts = torch.zeros(v_shape[1]).to('cuda')

        # for i in range(v_shape[1]):
        #   #original human1 not SMPL
        #   #if verts[0,i,2] > min_z + len_z * 0.55 and verts[0,i,0] > min_x + len_x*0.3 and verts[0,i,0] < min_x + len_x*0.7 and verts[0,i,1] > min_y + len_y*0.6 and verts[0,i,1] < min_y + len_y*0.7:
        #   #SMPL front
        #   if verts[0,i,2] > min_z + len_z * 0.55 and verts[0,i,0] > min_x + len_x*0.35 and verts[0,i,0] < min_x + len_x*0.65 and verts[0,i,1] > min_y + len_y*0.65 and verts[0,i,1] < min_y + len_y*0.75:
        #   #back
        #   #if verts[0,i,2] < min_z + len_z * 0.5 and verts[0,i,0] > min_x + len_x*0.35 and verts[0,i,0] < min_x + len_x*0.65 and verts[0,i,1] > min_y + len_y*0.65 and verts[0,i,1] < min_y + len_y*0.75:
        #   #leg
        #   #if verts[0,i,0] > min_x + len_x*0.5 and verts[0,i,0] < min_x + len_x and verts[0,i,1] > min_y + len_y*0.2 and verts[0,i,1] < min_y + len_y*0.3:
        #     sampled_verts[i] = 1

        # faces = mesh.faces_padded()
        # f_shape = faces.shape

        # sampled_planes = list()
        # for i in range(faces.shape[1]):
        #   v1 = faces[0,i,0]
        #   v2 = faces[0,i,1]
        #   v3 = faces[0,i,2]
        #   if sampled_verts[v1]+sampled_verts[v2]+sampled_verts[v3]>=1:
        #     sampled_planes.append(i)
        
        # Sample faces from index file:
        sampled_planes = np.load(self.config.idx_dir).tolist()
        idx = torch.Tensor(sampled_planes).long().to(self.device)
        self.idx = idx
        patch = torch.rand(len(sampled_planes), 1, 1, 3, device=(self.device), requires_grad=True)
        self.patch = patch

    def create_renderer(self):
        self.num_angles_train = self.config.num_angles_train
        self.num_angles_test = self.config.num_angles_test

        azim_train = torch.linspace(-1 * self.config.angle_range_train, self.config.angle_range_train, self.num_angles_train)
        azim_test = torch.linspace(-1 * self.config.angle_range_test, self.config.angle_range_test, self.num_angles_test)

        # Cameras for SMPL meshes:
        camera_dist = 2.2
        R, T = look_at_view_transform(camera_dist, 6, azim_train)
        train_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        self.train_cameras = train_cameras

        R, T = look_at_view_transform(camera_dist, 6, azim_test)
        test_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        self.test_cameras = test_cameras
        
        raster_settings = RasterizationSettings(
            image_size=self.config.img_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        lights = PointLights(device=self.device, location=[[0.0, 85, 100.0]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=train_cameras, 
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=self.device, 
                cameras=train_cameras,
                lights=lights
            )
        )

        return renderer
    
    def change_cameras(self, mode, camera_dist=2.2):
      azim_train = torch.linspace(-1 * self.config.angle_range_train, self.config.angle_range_train, self.num_angles_train)
      azim_test = torch.linspace(-1 * self.config.angle_range_test, self.config.angle_range_test, self.num_angles_test)

      R, T = look_at_view_transform(camera_dist, 6, azim_train)
      train_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
      self.train_cameras = train_cameras

      R, T = look_at_view_transform(camera_dist, 6, azim_test)
      test_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
      self.test_cameras = test_cameras

      if mode == 'train':
        self.renderer.rasterizer.cameras=self.train_cameras
        self.renderer.shader.cameras=self.train_cameras
      elif mode == 'test':
        self.renderer.rasterizer.cameras=self.test_cameras
        self.renderer.shader.cameras=self.test_cameras

    def render_mesh_on_bg(self, mesh, bg_img, num_angles, location=None, x_translation=0, y_translation=0):
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
        new_contour = torch.zeros(num_angles, bg_shape[2], bg_shape[3], 3)
        
        new_contour[:,:,:,0] = F.pad(contour[:,:,:,0], location, "constant", value=0)
        new_contour[:,:,:,1] = F.pad(contour[:,:,:,1], location, "constant", value=0)
        new_contour[:,:,:,2] = F.pad(contour[:,:,:,2], location, "constant", value=0)

        new_human = torch.zeros(num_angles, bg_shape[2], bg_shape[3], 3)
        new_human[:,:,:,0] = F.pad(human[:,:,:,0], location, "constant", value=0)
        new_human[:,:,:,1] = F.pad(human[:,:,:,1], location, "constant", value=0)
        new_human[:,:,:,2] = F.pad(human[:,:,:,2], location, "constant", value=0)

        final = torch.where((new_contour == 0).cpu(), new_bg.cpu(), new_human.cpu())
        return final

    def render_mesh_on_bg_batch(self, mesh, bg_imgs, num_angles, location=None, x_translation=0, y_translation=0):
        num_bgs = bg_imgs.shape[0]

        images = self.renderer(mesh) # (num_angles, 416, 416, 4)
        images = torch.cat(num_bgs*[images], dim=0) # (num_angles * num_bgs, 416, 416, 4)

        bg_shape = bg_imgs.shape

        # bg_imgs: (num_bgs, 3, 416, 416) -> (num_bgs, 416, 416, 3)
        bg_imgs = bg_imgs.permute(0, 2, 3, 1)

        # bg_imgs: (num_bgs, 416, 416, 3) -> (num_bgs * num_angles, 416, 416, 3)
        bg_imgs = bg_imgs.repeat_interleave(repeats=num_angles, dim=0)

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
        new_contour = torch.zeros(num_angles * num_bgs, bg_shape[2], bg_shape[3], 3, device=self.device)
        
        new_contour[:,:,:,0] = F.pad(contour[:,:,:,0], location, "constant", value=0)
        new_contour[:,:,:,1] = F.pad(contour[:,:,:,1], location, "constant", value=0)
        new_contour[:,:,:,2] = F.pad(contour[:,:,:,2], location, "constant", value=0)

        new_human = torch.zeros(num_angles * num_bgs, bg_shape[2], bg_shape[3], 3, device=self.device)
        new_human[:,:,:,0] = F.pad(human[:,:,:,0], location, "constant", value=0)
        new_human[:,:,:,1] = F.pad(human[:,:,:,1], location, "constant", value=0)
        new_human[:,:,:,2] = F.pad(human[:,:,:,2], location, "constant", value=0)

        # output: (num_angles * num_bgs, 416, 416, 3)
        final = torch.where((new_contour == 0), bg_imgs, new_human)
        return final

def main():
    import argparse
    import sys

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

    parser.add_argument('--patch_dir', type=str, default=None)
    parser.add_argument('--idx_dir', type=str, default='idx/chest_legs1.idx')
    
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--num_bgs', type=int, default=10)
    parser.add_argument('--num_test_bgs', type=int, default=2)
    parser.add_argument('--num_angles_test', type=int, default=1)
    parser.add_argument('--angle_range_test', type=int, default=0)
    parser.add_argument('--num_angles_train', type=int, default=1)
    parser.add_argument('--angle_range_train', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--rand_translation', type=int, default=50)
    parser.add_argument('--num_meshes', type=int, default=1)

    parser.add_argument('--cfgfile', type=str, default="cfg/yolo.cfg")
    parser.add_argument('--weightfile', type=str, default="data/yolov2/yolo.weights")
    
    parser.add_argument('--detector', type=str, default='yolov2')
    parser.add_argument('--test_only', action='store_true')

    config = parser.parse_args()
    trainer = Patch(config, device)
    
    # Faster RCNN setup to match the checkpoints
    Config.setup(image_min_side=800, image_max_side=1333, anchor_sizes="[64, 128, 256, 512]", rpn_post_nms_top_n=1000)

    # Uncomment this to manually run faster rcnn test on a trained patch
    # trainer.test_patch_faster_rcnn(
    #   path_to_checkpoint='/content/drive/My Drive/3D_Logo/model-180000.pth',
    #   dataset_name="coco2017", 
    #   backbone_name="resnet101", 
    #   prob_thresh=0.6)

    if config.test_only:
        if config.detector == 'yolov2':
            trainer.test_patch() 
        elif config.detector == 'faster_rcnn':
            trainer.test_patch_faster_rcnn(
                path_to_checkpoint='faster_rcnn/model-180000.pth',
                dataset_name="coco2017", 
                backbone_name="resnet101", 
                prob_thresh=0.6
            )
    else:
        if config.detector == 'yolov2':
            trainer.attack() 
        elif config.detector == 'faster_rcnn':
            trainer.attack_faster_rcnn()

if __name__ == '__main__':
    main()