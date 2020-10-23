from torch.utils.data import DataLoader, Dataset
import fnmatch
import os
from pytorch3d.io import load_objs_as_meshes, load_obj

class MeshDataset(Dataset):
  def __init__(self, mesh_dir, device, shuffle=True):
    #/data/meshes/...
    self.len = len(fnmatch.filter(os.listdir(mesh_dir), '*.obj'))
    self.mesh_dir = mesh_dir
    self.shuffle = shuffle

    self.mesh_filenames = fnmatch.filter(os.listdir(mesh_dir), '*.obj')
    self.mesh_files = []
    for m in self.mesh_filenames:
      self.mesh_files.append(os.path.join(self.mesh_dir, m))

    print('Meshes: ', self.mesh_files)
    self.meshes = []
    for mesh in self.mesh_files:
      self.meshes.append(load_objs_as_meshes([mesh], device=device, create_texture_atlas = True))

  def __len__(self):
    return self.len
  
  def __getitem__(self, idx):
    return self.meshes[idx]
