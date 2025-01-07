import bpy
import bmesh
import os
import math
import cv2

from pathlib import Path

def init():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setCamera(lens, x, y, z, radx, rady, radz, camera):
    scn = bpy.context.layer_collection
    cam1 = bpy.data.cameras.new(camera)
    cam1.lens = lens # 焦距
    cam_obj1 = bpy.data.objects.new(camera, cam1)
    cam_obj1.location = (x, y, z)
    cam_obj1.rotation_euler = (math.radians(radx), math.radians(rady), math.radians(radz))
    scn.collection.objects.link(cam_obj1)
    bpy.context.scene.camera = cam_obj1

def addMtl(obj, path):
    mat = bpy.data.materials.new(name="Material")
    obj.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")

    if os.path.exists(path):
        image = bpy.data.images.load(path)
        texture_node = nodes.new(type="ShaderNodeTexImage")
        texture_node.image = image
        links = mat.node_tree.links
        links.new(texture_node.outputs["Color"], principled_bsdf.inputs["Base Color"])

def getOverride(context):
    area = [a for a in context.window.screen.areas if a.type == 'VIEW_3D']
    region = [region for region in area[0].regions if region.type == 'WINDOW']

    override = {
        'window': context.window,
        'screen': context.window.screen,
        'area'  : area[0],
        'region': region[0],
        'scene': bpy.context.scene,
        'space': area[0].spaces[0]
    }
    return override

init()

override = getOverride(bpy.context)

bpy.ops.wm.obj_import(filepath="./results/0/mesh.obj")
obj = bpy.data.objects["mesh"]
obj.rotation_euler = (math.radians(0), math.radians(0), math.radians(270))
obj.location = (0, 0, 0)

# add color attribute
mat = bpy.data.materials.get("Material")
obj.data.materials.append(mat)
mat.use_nodes = True
nodes = mat.node_tree.nodes
principled_bsdf = nodes.get("Principled BSDF")
color_attribute = nodes.new(type="ShaderNodeVertexColor")
mat.node_tree.links.new(color_attribute.outputs["Color"], principled_bsdf.inputs["Base Color"])

# add new material
addMtl(obj, "./results/photo.png")

img = cv2.imread("./results/photo.png")
img_h = img.shape[0]

# select vector
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')

p = Path('./blenderScript/variable.txt')
n = float(p.read_text())

l = []

bm = bmesh.from_edit_mesh(obj.data)
for v in bm.verts:
    l.append(v.co.z)
l.sort()

h = l[len(l) - 1] - l[0]

r = []

bm = bmesh.from_edit_mesh(obj.data)
for v in bm.verts:
    if v.co.z > l[len(l) - 1] - (h / 1455 * img_h):
        r.append(v)
x = []
for v in r:
    x.append(v.co.x)
x.sort()
d = x[len(x) - 1] - x[0]
for v in bm.verts:
    if v.co.z > l[len(l) - 1] - (h / 1455 * img_h) and v.co.x > x[len(x) - 1] - d / n:
        v.select = True
    else:
        v.select = False
bm.select_flush(True)

bpy.context.object.active_material_index = 1
bpy.ops.object.material_slot_assign()
uv_map = obj.data.uv_layers.new(name="UVMap")
obj.data.uv_layers.active = obj.data.uv_layers["UVMap"]

# texturing
setCamera(15000, 0.015, -50, 0.03, 90, 0, 0, "CAMERA1")
override['space'].region_3d.view_perspective = 'CAMERA'

with bpy.context.temp_override(**override):
    bpy.ops.uv.project_from_view(camera_bounds=False, correct_aspect=True, scale_to_bounds=False)

bpy.ops.export_scene.fbx(filepath="./results/result.fbx", use_selection=True)

# make gif
setCamera(1500, 0, -100, 0, 90, 0, 0, "CAMERA2")
override['space'].region_3d.view_perspective = 'CAMERA'

# Create light datablock
light_data = bpy.data.lights.new(name="my-light-data", type='POINT')
light_data.energy = 10000

# Create new object, pass the light data 
light_object = bpy.data.objects.new(name="my-light", object_data=light_data)

# Link object to collection in context
bpy.context.collection.objects.link(light_object)

# Change light position
light_object.location = (0, -10, 0)

bpy.context.scene.render.image_settings.file_format = 'PNG'

for i in range(1, 16):
    obj.rotation_euler = (math.radians(0), math.radians(0), math.radians(24 * i))
    bpy.context.scene.render.filepath = "image/" + str(i) + ".png"
    bpy.ops.render.render(write_still=True)

bpy.ops.wm.save_as_mainfile(filepath="./results/result.blender")