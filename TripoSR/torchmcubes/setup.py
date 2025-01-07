# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['torchmcubes']

package_data = \
{'': ['*']}

setup_kwargs = {
    'name': 'torchmcubes',
    'version': '0.1.0',
    'description': 'torchmcubes: marching cubes for PyTorch',
    'long_description': "torchmcubes: marching cubes for PyTorch\n===\n\n[![Build (CPU)](https://github.com/tatsy/torchmcubes/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/tatsy/torchmcubes/actions/workflows/build.yml)\n\n> Marching cubes for PyTorch environment. Backend is implemented with C++ and CUDA.\n\n## Install\n\n```shell\npip install git+https://github.com/tatsy/torchmcubes.git\n```\n\n#### Build only\n\n```shell\n# After cloning this repo and moving into the root folder\npoetry build\n```\n\n## Usage\n\nSee [mcubes.py](./mcubes.py) for more details.\n\n```python\nimport numpy as np\n\nimport torch\nfrom torchmcubes import marching_cubes, grid_interp\n\n# Grid data\nN = 128\nx, y, z = np.mgrid[:N, :N, :N]\nx = (x / N).astype('float32')\ny = (y / N).astype('float32')\nz = (z / N).astype('float32')\n\n# Implicit function (metaball)\nf0 = (x - 0.35) ** 2 + (y - 0.35) ** 2 + (z - 0.35) ** 2\nf1 = (x - 0.65) ** 2 + (y - 0.65) ** 2 + (z - 0.65) ** 2\nu = 1.0 / f0 + 1.0 / f1\nrgb = np.stack((x, y, z), axis=-1)\nrgb = np.transpose(rgb, axes=(3, 2, 1, 0)).copy()\n\n# Test\nu = torch.from_numpy(u).cuda()\nrgb = torch.from_numpy(rgb).cuda()\nverts, faces = marching_cubes(u, 15.0)\ncolrs = grid_interp(rgb, verts)\n\nverts = verts.cpu().numpy()\nfaces = faces.cpu().numpy()\ncolrs = colrs.cpu().numpy()\n\n# Use Open3D for visualization (optional)\nimport open3d as o3d\n\nmesh = o3d.geometry.TriangleMesh()\nmesh.vertices = o3d.utility.Vector3dVector(verts)\nmesh.triangles = o3d.utility.Vector3iVector(faces)\nmesh.vertex_colors = o3d.utility.Vector3dVector(colrs)\nwire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)\no3d.visualization.draw_geometries([mesh, wire], window_name='Marching cubes (CUDA)')\n```\n\n## Screen shot\n\n![metaball.png](./metaball.png)\n\n## Copyright\n\nMIT License 2019-2024 (c) Tatsuya Yatagawa\n",
    'author': 'Tatsuya Yatagawa',
    'author_email': 'tatsy.mail@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'python_requires': '>=3.9',
}
from build_cxx import *
build(setup_kwargs)

setup(**setup_kwargs)
