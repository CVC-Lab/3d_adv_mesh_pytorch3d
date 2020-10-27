import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes, load_obj

def save_obj_with_texture( verts, faces, aux, vert_colors, filename:str ):
    """Saves a WavefrontOBJ object to a file from output of pytorch3d

    Warning: Contains no error checking!

    Warning: Assume only one material is used

    """
    # Assume no '.' in folder/file name
    prefix = filename.split('.')[0]
    print(prefix)

    verts = verts.cpu().numpy()
    with open( filename, 'w' ) as ofile:
        
        for i in range(verts.shape[0]):
            vtx = verts[i,:]
            vtc = vert_colors[i,:]
            ofile.write('v '+' '.join(['{}'.format(v) for v in vtx])+ ' '+ ' '.join(['{}'.format(v) for v in vtc])+ '\n')
        for tex in aux.verts_uvs.cpu().numpy():
            ofile.write('vt '+' '.join(['{}'.format(vt) for vt in tex])+'\n')
        # for nrm in aux.normals.cpu().numpy():
        #     ofile.write('vn '+' '.join(['{}'.format(vn) for vn in nrm])+'\n')
        # if not obj.mtlid:
        #     obj.mtlid = [-1] * len(obj.polygons)
        for f in faces.cpu().numpy():
            # UGLY! 
            ofile.write('f '+' '.join([('%d' % (vid+1)) for vid in f])+'\n')

if __name__=="__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    patch = torch.load('data/patch_save_2.pt').to(device)
    idx = torch.load('data/idx_save_2.pt').to(device)
    mesh = load_objs_as_meshes(['data/human.obj'], device=device, create_texture_atlas = True)
    verts, faces, aux = load_obj('data/human.obj', device=device, create_texture_atlas = True)
    
    # This is from pytorch3d current github code repo, which is not exist in stable releases
    atlas_packed=mesh.textures.atlas_packed()
    atlas_packed[idx,:,:,:] = patch
    t0 = atlas_packed[:, 0, -1]  # corresponding to v0  with bary = (1, 0)
    t1 = atlas_packed[:, -1, 0]  # corresponding to v1 with bary = (0, 1)
    t2 = atlas_packed[:, 0, 0]  # corresponding to v2 with bary = (0, 0)
    texture_image = torch.stack((t0, t1, t2), dim=1)
    face_id  = faces.verts_idx 
    
    print(verts.shape)
    vert_colors= np.zeros(verts.shape)
    for i in range(texture_image.shape[0]):
        each = texture_image[i,:,:].cpu()
        vert_colors[face_id[i,0]] = each[0,:]
        vert_colors[face_id[i,1]] = each[1,:]
        vert_colors[face_id[i,2]] = each[2,:]
        

    save_obj_with_texture(verts, face_id, aux, vert_colors, filename='out/test.obj')

