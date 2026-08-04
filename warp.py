import torch
import torch.nn.functional as F
from torchvision.io import read_image
import matplotlib.pyplot as plt

CAM = 3  
target_loc = "./kitti_data/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/0000000000.png"
source_loc = "./kitti_data/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/0000000001.png"
calib_loc  = "./kitti_data/2011_09_26/calib_cam_to_cam.txt"

def load_image(path):
    img = read_image(path).float() / 255.0
    return img.unsqueeze(0)

def parse_intrinsics(path, cam):
    with open(path) as f:
        for line in f:
            if line.startswith(f"P_rect_0{cam}:"):
                vals = [float(x) for x in line.split()[1:]]
                P = torch.tensor(vals).reshape(3, 4)
                return P[:, :3]

def make_pixel_grid(H, W):
    v, u = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    ones = torch.ones_like(u)
    return torch.stack([u, v, ones], dim=0).reshape(3, -1).float()

def to_grid_sample_coords(u, v, H, W):
    u_n = 2.0 * u / (W - 1) - 1.0
    v_n = 2.0 * v / (H - 1) - 1.0
    return torch.stack([u_n, v_n], dim=-1).reshape(1, H, W, 2)

def backproject(depth, K, pix):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = pix[0]   
    v = pix[1] 

    X = (u - cx) * depth / fx 
    Y = (v - cy) * depth / fy 
    Z = depth

    return torch.stack([X, Y, Z], dim=0) 
    
def transform(points, pose):
    ones = torch.ones(1, points.shape[1])   
    points_h = torch.cat([points, ones], dim=0)  

    transformed_h = pose @ points_h 
    return transformed_h[:3] 

def project(points, K):
    p = K @ points  
    u = p[0] / p[2]
    v = p[1] / p[2]
    return u,v 

def warp(source_img, depth, K, pose, H, W):
    pix = make_pixel_grid(H, W)            
    cam_pts = backproject(depth.reshape(-1), K, pix)
    src_pts = transform(cam_pts, pose)
    u, v = project(src_pts, K)
    grid = to_grid_sample_coords(u, v, H, W)
    return F.grid_sample(source_img, grid, align_corners=True)

if __name__ == "__main__":
    target = load_image(target_loc)
    source = load_image(source_loc)
    _, _, H, W = target.shape

    K = parse_intrinsics(calib_loc, CAM)
    depth = torch.full((H, W), 20.0)          

    identity = torch.eye(4)
    warped_identity = warp(source, depth, K, identity, H, W)
    diff = (warped_identity - target).abs().mean().item()
    print(f"identity-pose mean abs diff (should be ~0): {diff:.5f}")

    pose = torch.eye(4); pose[2, 3] = -1.0      
    warped = warp(source, depth, K, pose, H, W)

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    for a, img, t in zip(ax, [target, source, warped],
                         ["target", "source", "warped"]):
        a.imshow(img[0].permute(1, 2, 0).clamp(0, 1)); a.set_title(t); a.axis("off")
    plt.savefig("warp_result.png"); print("saved warp_result.png")