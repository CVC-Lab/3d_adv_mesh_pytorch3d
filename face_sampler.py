import bpy
import numpy

'''
This script is meant to be used inside of Blender.

When executed, this script will save the indices of all selected faces
to a file. To change the filename, please change 'MY_FILE.idx' on line 32.
'''

ob = bpy.context.active_object
me = ob.data

selfaces =[]

editmode = False
if ob.mode == 'EDIT':
    editmode =True
    bpy.ops.object.mode_set()
        
for f in me.polygons:
    if f.select:
        print(f.index)
        selfaces.append(f)
        
face_ids = []
for f in selfaces:
    face_ids.append(f.index)

np_arr = numpy.asarray(face_ids)

with open(r'MY_FILE.idx', 'wb') as f:
    numpy.save(f, np_arr)

#done editing, restore edit mode if needed
if editmode:
    bpy.ops.object.mode_set(mode = 'EDIT')