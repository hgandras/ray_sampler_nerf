from typing import Literal

import torch

import mitsuba as mi

def load_sensor(r, phi, theta, width = 512,height = 512, spp = 32,fov = 60):
    # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
    origin = mi.ScalarTransform4f().rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])

    return mi.load_dict({
        'type': 'perspective',
        'fov': fov,
        'to_world': T().look_at(
            origin=origin,
            target=[0, 0, 0],
            up=[0, 0, 1]
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': spp
        },
        'film': {
            'type': 'hdrfilm',
            'width': width,
            'height': height,
            'rfilter': {
                'type': 'box',
            },
            'pixel_format': 'rgb',
        },
    })

def _sample_sphere(samples):
    z = 1 - 2 * samples[:,0]
    z_null = torch.stack([1 - z**2,torch.zeros_like(z)])
    r = torch.sqrt(torch.max(z_null,dim = 0)[0])
    phi = 2 * torch.pi * samples[:,1]
    return torch.stack([r * torch.cos(phi),r * torch.sin(phi),z],dim = 1)


def _sample_hemisphere(samples):
    z = samples[:,0]
    z_null = torch.stack([1 - z**2,torch.zeros_like(z)])
    r = torch.sqrt(torch.max(z_null,dim = 0)[0])
    phi = 2 * torch.pi * samples[:,1]
    return torch.stack([r * torch.cos(phi),r * torch.sin(phi),z],dim = 1)

def _sample_rays(
      num_samples,
      radius : float = 1.0,
      center : torch.Tensor = torch.Tensor([0,0,0]),
      cone_max_angle = torch.pi / 6 ,
      sample_form : Literal["sphere", "hemisphere"] = "sphere",
      group_factor: int = 1
    ):
    """
    Sample rays uniformly from the sphere around the scene. I might also give small perturbations to the ray directions, so
    they dont just point to the center of the scene.
    """
    device = "cuda" if mi.variant() == "cuda_ad_rgb" else "cpu"

    samples = torch.rand(int(num_samples/group_factor),2)

    if sample_form == "sphere":
        origins = _sample_sphere(samples)
    elif sample_form == "hemisphere":
        origins = _sample_hemisphere(samples)
    else:
        raise ValueError(f"Unknown sample form {sample_form}")

    origins = origins.repeat(1,group_factor).reshape(-1,3)
    # Perturb the ray directions
    dirs = -origins
    dirs = dirs / torch.norm(dirs,dim = 1,keepdim = True)
    # Call the cone sampling function, and convert the outputs to the correct coordinate
    # systems.
    # 0.Sample cone
    offsets = _sample_cone(torch.cos(torch.tensor([cone_max_angle])),num_samples)
    # 1.Create local systems
    z_loc = dirs
    up = torch.tensor([0,0,1],dtype = torch.float32).expand_as(z_loc)
    y_loc = torch.cross(z_loc,up,dim = 1)
    #y_loc = y_loc / torch.norm(y_loc,dim = 1,keepdim = True)
    x_loc = torch.cross(y_loc,z_loc,dim = 1)
    locals = torch.stack([x_loc,y_loc,z_loc],dim = 2)
    # 2. Rotate the samples, which will result in the new directions
    offsets = offsets.reshape([offsets.shape[0], offsets.shape[1], 1])
    dirs = locals @ offsets
    origins = origins * radius * 1.5 + center
    return origins.to(device),dirs.squeeze(-1).to(device)

def _intersect_scene(scene ,origins : torch.Tensor,dirs : torch.Tensor, spp : int):
    """
    Intersects the scene with the generated rays.
    """
    #Convert to mitsuba rays
    o,d = mi.Vector3f(origins.T),mi.Vector3f(dirs.T)
    device = "cuda" if mi.variant() == "cuda_ad_rgb" else "cpu"
    sampler = mi.load_dict({
                "type" : "independent"
            })
    
    rays = mi.Ray3f(o,d)
    #Intersect the scene
    sum = torch.zeros_like(origins,device = device)
    for i in range(spp):
        spec,_,_ = scene.integrator().sample(scene,sampler,rays)
        sum+=spec.torch().T
        if i > 0 :
            sampler.advance()
            sampler.schedule_state()
    return sum / spp

def _sample_cone(cosThetaMax ,num_samples = 5000):
    uvs = torch.rand(num_samples,2)
    cosTheta = (1 - uvs[:,0]) + uvs[:,0] * cosThetaMax
    sinTheta = torch.sqrt(1 - cosTheta**2)
    phi = 2 * torch.pi * uvs[:,1]
    return  torch.stack([torch.cos(phi) * sinTheta, torch.sin(phi) * sinTheta, cosTheta], dim = 1)

def _sample_images(scene , w : int,h : int,sensors ) -> torch.Tensor:
    images = torch.empty((0,w,h,3))
    num_images = len(sensors)
    if mi.variant() == "cuda_ad_rgb":
        denoiser = mi.OptixDenoiser(input_size=(w,h))
    print(f"Image 0/{num_images} rendered", end = "\r")
    for imgID,sensor in enumerate(sensors):
        render = mi.render(scene, sensor = sensor, integrator = scene.integrator())
        if mi.variant() == 'cuda_ad_rgb':
            render = denoiser(render)
        mi.util.write_bitmap(f"./data/bunny/images/img_{imgID}.jpg",render)
        render = render.torch().to("cpu")
        render = render.reshape((1,render.shape[0], render.shape[1],render.shape[2]))
        images = torch.cat((images,render))
        print(f"Image {imgID + 1}/{num_images} rendered", end = "\r")
    return images


def _sample_sensors(scene,w : int,h : int,num_cameras : int):
    """
    Generates random cameras
    Returns the view matrices of the cameras
    """
    sensors = []
    matrices = torch.empty((0,4,4))

    radius =  scene.bbox().bounding_sphere().radius
    center = scene.bbox().center().torch()
    min_radius = radius * 2.0
    max_radius = radius * 2.0
    range = max_radius - min_radius
    #Sample hemisphere uniformly
    thetas = torch.acos(torch.rand(num_cameras)) * 180/torch.pi
    phis = torch.rand(num_cameras) * 360
    cam_positions = _sample_sphere(torch.rand(num_cameras,2))

    for origin in cam_positions:
        origin = mi.ScalarVector3f(origin)
        r = torch.randn(1) * range + min_radius
        sensor = load_sensor(w,h,r,origin)
        sensors.append(sensor)
        #GSPLAT USES DIFFERENT COORDINATE SYSTEM, SO CAMERA NEED TO BE ROTATED AROUND ITS OWN Z AXIS!!!!!:)
        matrix = mi.ScalarTransform4f().look_at(origin = origin, target = [0,0,0], up = [0,0,-1]).matrix.numpy()
        matrix = torch.reshape(matrix,(1,4,4))
        matrices = torch.cat((matrices,torch.Tensor(matrix)), dim = 0)
    return sensors,matrices