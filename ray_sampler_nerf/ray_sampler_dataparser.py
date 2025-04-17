from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.cameras.cameras import Cameras, CameraType

from pathlib import Path
from typing import Literal, Optional, Tuple, Type
from dataclasses import dataclass, field
import numpy as np
import torch
import math
import importlib.util
import sys
import os

import mitsuba as mi

@dataclass
class RaySamplerDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: RaySamplerDataParser)
    """target class to instantiate"""
    data: Path = Path("rays.npz")
    """Path to a mitsuba xml scene file"""
    mitsuba_variant : Literal["cuda_ad_rgb", "llvm_ad_rgb","scalar_rgb"] = "cuda_ad_rgb"
    """Which mitsuba variant to load"""
    scale_factor: float = 1.0
    """How much to scale the region of interest by."""
    norm_scene : bool = True
    """If this is true, scene_scale is ignored, and origins are scaled to the unit sphere"""
    center_origin : bool = True
    """Whether to center the scene to the origin"""
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    pixel_size_modifier : float = 1.0
    """changes the pixel size, used for experimenting"""
    
 
@dataclass
class RaySamplerDataParser(DataParser):

    config: RaySamplerDataParserConfig

    def _generate_dataparser_outputs(self,split = "train") -> DataparserOutputs:
        """Load an npz file, which contains ray directions, origins, and colors [0,1].
        The loaded array should have the shape (num_samples,3,3)"""

        mi.set_variant(self.config.mitsuba_variant)

        #TODO: Might move these to __init__
        filename = self.config.data.name
        extension = str.split(filename,".")[1]

        if extension == "npz":
            data = np.load(self.config.data,allow_pickle=True)
            center = data["center"]
            rays = data["rays"]
            num_sampling_points = rays.shape[0] / data["group_factor"]

            radius = data["radius"]
            scale_factor = 1.0/radius if self.config.norm_scene else self.config.scale_factor
            radius = radius * scale_factor

            colors = torch.Tensor(rays[2])
            origins = torch.Tensor(rays[0]) 
            if self.config.center_origin:
                origins = (origins - center) * scale_factor
            else:
                origins = (origins - center) * scale_factor + center
            directions = torch.Tensor(rays[1])
            bbox = SceneBox(aabb = torch.Tensor(data["bbox"]))
            metadata = {
                "colors" : colors,
                "origins" : origins,
                "directions" : directions,
                "pixel_size" : self._get_pixel_area(radius,num_sampling_points) * self.config.pixel_size_modifier,
                "type" : extension
            }

            #Generate cam matrices from origins and directions
            cam_matrices = self._get_camera_matrices(origins,directions)
            f = self._get_focus_point(radius,num_sampling_points,"ortho") #fx, and fy, same for both directions
            output = DataparserOutputs(
                image_filenames=[],
                cameras = Cameras(cam_matrices,f,f,0.5,0.5,1,1,camera_type = CameraType.ORTHOPHOTO ), 
                alpha_color = torch.ones(3,dtype=torch.float32),
                scene_box = bbox, #Will be loaded from the npz file,
                dataparser_scale = self.config.scale_factor,
                metadata = metadata
            )

        elif extension == "xml":
            scene = mi.load_file(str(self.config.data))
            bbox = scene.bbox()
            corner = bbox.corner(0).torch()
            max = bbox.corner(7).torch()
            bbox_tensor = torch.stack([corner,max],dim = 0)
            metadata = {"scene" : scene, "type" : extension}
            output = DataparserOutputs(
                image_filenames = [],
                cameras= Cameras(torch.eye(4),1.0,1.0,0.5,0.5,1,1, camera_type= CameraType.PERSPECTIVE), #placeholder 
                alpha_color = torch.ones(3,dtype = torch.float32),
                scene_box = SceneBox(aabb = bbox_tensor),
                metadata = metadata
            )
        elif extension == "py":
            # Example usage
            module_name = 'my_module'
            print(self.config.data.name)
            module = self.import_file_by_filename(module_name, self.config.data)
            scene_dict = module.get_scene_dict()
            scene = mi.load_dict(scene_dict)
            bbox = scene.bbox()
            corner = bbox.corner(0).torch()
            max = bbox.corner(7).torch()
            bbox_tensor = torch.stack([corner,max],dim = 0)
            metadata = {"scene" : scene, "type" : extension}
            output = DataparserOutputs(
                image_filenames = [],
                cameras= Cameras(torch.eye(4),1.0,1.0,0.5,0.5,1,1, camera_type= CameraType.PERSPECTIVE), #placeholder 
                alpha_color = torch.ones(3,dtype = torch.float32),
                scene_box = SceneBox(aabb = bbox_tensor),
                metadata = metadata
            )
        else:
            raise ValueError(f"File {filename} is not an acceptable file format!")

        return output
    
    def _get_camera_matrices(
        self, 
        origins: torch.Tensor, 
        directions: torch.Tensor, 
        up : torch.Tensor = torch.Tensor([0,1,0])
    ) -> torch.Tensor:
        z = directions
        up = up.repeat(z.shape[0],1)
        y = torch.cross(z,up)
        x = torch.cross(y,z)
        rotmat = torch.stack([x,y,z],dim=1)
        cammat = torch.cat([rotmat,origins.unsqueeze(2)],dim=2)
        last_row = torch.Tensor([[0,0,0,1]]).repeat(cammat.shape[0],1).unsqueeze(1)
        cammat = torch.cat([cammat,last_row],dim=1)
        return cammat
    
    def _get_focus_point(self,radius : float, num_samples : int, camera_type : str = "ortho") -> float:
        if camera_type == "ortho":
            #Calculates the average distance between the samples, and returns the reciproc as fx (or half of it we'll see)
            return 2.0 / math.sqrt(self._get_pixel_area(radius,num_samples))
        elif camera_type == "perspective":
            #Return the focal point, such that the FOV corresponds to the plane with size of the average distance between the samples,
            #and the distance to the camera is 1.0
            return 1.0
        raise ValueError(f"Camera type {camera_type} not supported")
    
    def _get_pixel_area(self, radius, num_samples):
        return 4 * torch.pi * radius * radius / num_samples
    
    def import_file_by_filename(self, module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module





