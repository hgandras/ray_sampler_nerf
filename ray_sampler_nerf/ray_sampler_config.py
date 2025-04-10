from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

from ray_sampler_nerf.ray_sampler_dataparser import RaySamplerDataParserConfig
from ray_sampler_nerf.ray_sampler_datamanager import RaySamplerDataManagerConfig

ray_sampler = MethodSpecification(
    config = TrainerConfig(
        method_name = "ray_sampler_nerf",
        steps_per_eval_batch = 500,
        steps_per_save = 2000,
        max_num_iterations = 1000,
        mixed_precision = True,
        pipeline = VanillaPipelineConfig(
            datamanager = RaySamplerDataManagerConfig(
                dataparser = RaySamplerDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            average_init_density=0.01,
            camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer"
    ),
    description = "A nerf method that uses rays sampled from synthetic datasets, and not images."
    
)
