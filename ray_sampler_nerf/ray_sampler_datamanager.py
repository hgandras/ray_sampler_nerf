from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion

from pathlib import Path
from typing import Tuple, Dict, Union, Type, List, Literal
from dataclasses import dataclass, field
from functools import cached_property

import torch
from torch.nn import Parameter 
from torch.utils.data import Dataset, DataLoader

from ray_sampler_nerf.ray_dataset import PreloadedRayDataset,PresamplerRayDataset,ParallelSampleRayDataset, RayDatasetBase, RayStream
from ray_sampler_nerf.ray_sampler_dataparser import RaySamplerDataParserConfig

import mitsuba as mi

@dataclass
class RaySamplerDataManagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: RaySamplerDataManager)
    """RaySamplerDataManager config."""
    dataparser : AnnotatedDataParserUnion = field(default_factory= RaySamplerDataParserConfig)
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch : int = 4096
    """Number of rays per batch for training."""
    eval_num_rays_per_batch:int = 4096
    """Number of rays per batch for evaluation."""
    pregenerate_samples : bool = False
    """Whether to pregenerate the ray samples, or calculate them parallel during training"""
    num_samples : int = 1_000_000
    """Number of rays to sample before training"""
    cone_angle : float = 60
    """The angle of the cone which is used for sampling the direction offset"""
    group_factor : int = 50
    """How many rays are grouped at the same origin"""
    hemisphere : bool = True
    "Whether to sample the rays from a sphere, or a hemisphere around the object"
    spp : int = 16
    """Number of samples per directions"""

class RaySamplerDataManager(DataManager):
    def __init__(
        self,
        config : RaySamplerDataManagerConfig,
        device: str = "cpu",
        world_size: int = 1,
        **kwargs
    ):
        
        self.config = config
        self.device = device
        self.world_size = world_size
        if self.config.dataparser is not None:
            self.config.dataparser.data = self.config.data
        self.dataparser = self.config.dataparser.setup()
        
        self.dataparser_outputs = self.dataparser.get_dataparser_outputs()
        self.type = self.dataparser_outputs.metadata["type"]
        #The datasets will be initialized based on the input file type
        
        #Create the datasets, based on the split
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        super().__init__()

    def create_train_dataset(self):
        """Create the training dataset."""
        if self.type == "npz":
            return  PreloadedRayDataset(self.dataparser_outputs,split = 'train')
        
        if self.config.pregenerate_samples:
            return PresamplerRayDataset(
                self.dataparser_outputs,
                self.config.num_samples,
                self.config.cone_angle,
                self.config.group_factor,
                self.config.hemisphere,
                self.config.spp,
                self.test_mode
            )
        else:
            return  ParallelSampleRayDataset(
                self.dataparser_outputs,
                self.config.train_num_rays_per_batch,
                self.config.cone_angle,
                self.config.group_factor,
                self.config.hemisphere,
                self.config.spp
            )
                
    def create_eval_dataset(self):
        """Create the evaluation dataset."""
        if self.type == "npz":
            return  PreloadedRayDataset(self.dataparser_outputs,split = 'train')
        
        if self.config.pregenerate_samples:
            return PresamplerRayDataset(
                self.dataparser_outputs,
                self.config.num_samples,
                self.config.cone_angle,
                self.config.group_factor,
                self.config.hemisphere,
                self.config.spp,
                self.test_mode
            )
        else:
            return  ParallelSampleRayDataset(
                self.dataparser_outputs,
                self.config.num_samples,
                self.config.cone_angle,
                self.config.group_factor,
                self.config.hemisphere
            )
    
    def setup_train(self):
        self.train_ray_stream = RayStream(self.train_dataset)
        if isinstance(self.train_ray_stream.input_dataset, ParallelSampleRayDataset):
            self.train_dataloader = DataLoader(
                self.train_ray_stream,
                batch_size=1, 
                num_workers=0
            )
        else:
            self.train_dataloader = DataLoader(
                self.train_ray_stream,
                batch_size=self.config.train_num_rays_per_batch, 
                num_workers=0
            )
        self.train_iter = iter(self.train_dataloader)
    
    def setup_eval(self):
        self.eval_ray_stream = RayStream(self.eval_dataset)
        self.eval_dataloader = DataLoader(
            self.eval_ray_stream, 
            batch_size=self.config.eval_num_rays_per_batch,  
            num_workers=0
        )
        self.eval_iter = iter(self.eval_dataloader)

    def next_train(self, step: int) -> Tuple[Union[RayBundle, Cameras], Dict]:
        self.train_count += 1
        # Somehow it is faster to index everything here than loadiing everything to a dataset and iterating 
        # through that
        ray_batch = next(self.train_iter)
        indices = ray_batch["image_idx"].squeeze()
        if self.type == "npz":
            directions = self.dataparser_outputs.metadata["directions"][indices]
            origins = self.dataparser_outputs.metadata["origins"][indices]
            images = self.dataparser_outputs.metadata["colors"][indices].to(self.device)
            pixel_size = self.dataparser_outputs.metadata["pixel_size"]

        else:
            directions = ray_batch["directions"].squeeze()
            origins = ray_batch["origins"].squeeze()
            images = ray_batch["image"].squeeze().to(self.device)
            pixel_size = self._get_pixel_area(
                1.0,
                self.config.num_samples
                )
        min = 0.66666
        max  = 1.5
        pixelsize_modifier = (max - min) * torch.randn(len(indices)) + min
        ray_bundle = RayBundle(
            origins = origins,
            directions = directions,
            pixel_area = pixel_size , #* pixelsize_modifier,
            camera_indices = indices[:,None]
        ).to(self.device)

        extension =  torch.ones(self.config.train_num_rays_per_batch,2)
        ids = indices[:,None]
        ray_indices = torch.cat([ids,extension], dim = 1).to(self.device)
        batch ={ "image" : images, "indices" : ray_indices}
        
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[Union[RayBundle, Cameras], Dict]:
        self.eval_count += 1
        ray_batch = next(self.eval_iter)
        ray_bundle = RayBundle(
            origins = ray_batch["origin"].to(self.device),
            directions = ray_batch["direction"].to(self.device),
            pixel_area = self.pixel_size
        )
        batch ={"image" : ray_batch["image"].to(self.device)}
        return ray_bundle, batch 

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        raise NotImplementedError
    
    def get_train_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for training."""
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for evaluation."""
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        """Returns the path to the data. This is used to determine where to save camera paths."""
        return self.config.dataparser.data
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.

        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}
    
    def _get_pixel_area(self, radius, num_samples):
        return 4 * torch.pi * radius * radius / num_samples