from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs

from ray_sampler_nerf.ray_sampler_utils import _sample_rays, _intersect_scene

from torch.utils.data import IterableDataset, Dataset, get_worker_info
import torch

import mitsuba as mi

from copy import deepcopy
import math
import random
from abc import abstractmethod
from typing import Dict

class RayDatasetBase(Dataset):
    def __init__(self, dataparser_outputs : DataparserOutputs, split : str = "train"):
        self.dataparser_outputs = dataparser_outputs
        self.scene_box = deepcopy(self.dataparser_outputs.scene_box)
        self.cameras = deepcopy(self.dataparser_outputs.cameras)
        self.metadata = {}

    def __len__(self):
        return self._get_len()

    def __getitem__(self,idx):
        return self._get_item(idx) 

    @abstractmethod
    def _get_len(self) -> int:
        """Returns the length of the dataset"""
        raise NotImplementedError
    
    @abstractmethod
    def _get_item(self,idx) -> Dict:
        """Returns an element"""
        raise NotImplementedError

class PreloadedRayDataset(RayDatasetBase):
    """
    Used when the ray samples are loaded from an npz file
    """
    def __init__(self, dataparser_outputs: DataparserOutputs, split: str = "train"):
        super().__init__(dataparser_outputs,split)

        self.scene_box = deepcopy(self.dataparser_outputs.scene_box)
        self.colors = deepcopy(self.dataparser_outputs.metadata["colors"])
        self.colors = self.colors[:,None,None,:]
        self.length = len(self.dataparser_outputs.metadata["origins"])

    def _get_len(self):
        return self.length

    def _get_item(self, idx):
        return {
            "image_idx": idx,
            "image": self.colors[idx]
            }

class PresamplerRayDataset(RayDatasetBase):
    """
    Takes the scene and the sampling parameters, and samples the scene.
    """
    def __init__(
            self, 
            dataparser_outputs : DataparserOutputs , 
            num_samples: int, 
            cone_angle: float,
            group_factor : int , 
            hemisphere: bool
        ):
        super().__init__(dataparser_outputs)
        self.scene = self.dataparser_outputs.metadata['scene']
        radius =  self.scene.bbox().bounding_sphere().radius
        center = self.scene.bbox().center().torch()
        sample_type = "hemisphere" if hemisphere else "sphere"
        self.o,self.d = _sample_rays(num_samples,radius,center,cone_max_angle = cone_angle, sample_form = sample_type,group_factor = group_factor)
        self.colors = _intersect_scene(self.scene,self.o,self.d,32)
        self.length = self.o.shape[0]

    def _get_len(self):
        return self.length

    def _get_item(self,idx):
        return {
            "image_idx" : idx,
            "origins" : self.o[idx],
            "directions" : self.d[idx],
            "images" : self.colors[idx],
        }


class ParallelSampleRayDataset(RayDatasetBase):
    """Samples the rays parallel during the training.
    1. In the first implementation simply sample rays whenever the querying happens
    2. The plan is to start a parallel process, that fills up a buffer if it is empty. We need to 
    take care, if the querying from the buffer is faster than the 
    """
    def __init__(
            self,
            dataparser_outputs : DataparserOutputs, 
            num_samples: int, 
            cone_angle: float,
            group_factor : int, 
            hemisphere: bool
        ):
        """
        Args:
        scene : mi.scene mitsuba scene laoded from an xml file or python dict
        num_samples : int 
        """
        super().__init__(dataparser_outputs)

        self.scene = self.dataparser_outputs.metadata['scene']
        self.num_samples = num_samples #Here it is the batch size
        self.cone_angle = cone_angle
        self.hemisphere = hemisphere
        self.group_factor = group_factor

        self.buffer = [] #Contains lists 

    def _get_len(self):
        return self.num_samples
    
    def _get_item(self, idx):
        radius =  self.scene.bbox().bounding_sphere().radius
        center = self.scene.bbox().center().torch()
        group_factor = 1
        o,d = _sample_rays(self.num_samples,radius,center,cone_max_angle = self.cone_angle, sample_form = "hemisphere",group_factor = group_factor)
        colors = _intersect_scene(self.scene,o,d,32)
        length = o.shape[0]
        return {
            "image_idx" : idx,
            "origins" : o,
            "directions" : d,
            "images" : colors,
        }

class RayStream(IterableDataset):
    def __init__(
        self,
        dataset : RayDatasetBase,
        sampling_seed : int = 3333,
    ):
        self.input_dataset = dataset
        self.sampling_seed = sampling_seed

    def __iter__(self):
        dataset_indices = list(range(len(self.input_dataset)))
        worker_info = get_worker_info()
        if worker_info is not None:  # if we have multiple processes
            per_worker = int(math.ceil(len(dataset_indices) / float(worker_info.num_workers)))
            slice_start = worker_info.id * per_worker
        else:  # we only have a single process
            per_worker = len(self.input_dataset)
            slice_start = 0
        worker_indices = dataset_indices[
            slice_start : slice_start + per_worker
        ]  # the indices of the datapoints in the dataset this worker will load
        r = random.Random(self.sampling_seed)
        r.shuffle(worker_indices)
        i = 0  # i refers to what image index we are outputting: i=0 => we are yielding our first image,camera

        while True:
            if i >= len(worker_indices):
                r.shuffle(worker_indices)
                i = 0
            id = worker_indices[i]
            i += 1
            yield self.input_dataset[id]
            