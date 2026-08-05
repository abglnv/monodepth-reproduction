import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np 

CAM = 2
DRIVE = "./kitti_data/2011_09_26/2011_09_26_drive_0001_sync"
CALIB = "./kitti_data/2011_09_26"
FRAME_T = "0000000000"
FRAME_S = "0000000001"

target_loc = f"{DRIVE}/image_02/data/{FRAME_T}.png"
source_loc = f"{DRIVE}/image_02/data/{FRAME_S}.png"
bin_loc    = f"{DRIVE}/velodyne_points/data/{FRAME_T}.bin"   
calib_loc = "./kitti_data/2011_09_26/calib_cam_to_cam.txt"

calib_cam  = f"{CALIB}/calib_cam_to_cam.txt"    # P_rect_02, R_rect_00
calib_velo = f"{CALIB}/calib_velo_to_cam.txt"   # R, T  (LiDAR -> camera)

def read_calib(path, key): 
    with open(path) as f: 
        for line in f: 
            if line.startswith(key + ":"): 
                return [float(x) for x in line.split()[1:]]

def lidar_to_depth(bin_path, cam_calib, velo_calib, H, W, cam=2): 
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pts[:, 3] = 1.0
    pts = torch.tensor(pts).T     

    R = torch.tensor(read_calib(velo_calib, "R")).reshape(3, 3)
    T = torch.tensor(read_calib(velo_calib, "T")).reshape(3, 1)
    Tr = torch.eye(4); Tr[:3, :3] = R; Tr[:3, 3:] = T

    # recitification for kitti 
    R_rect = torch.eye(4)
    R_rect[:3, :3] = torch.tensor(read_calib(cam_calib, "R_rect_00")).reshape(3, 3)

    P = torch.tensor(read_calib(cam_calib, f"P_rect_0{cam}")).reshape(3, 4)

    # projection 
    cam_pts = R_rect @ Tr @ pts    
    depth = cam_pts[2]              
    img = P @ cam_pts                 
    u = (img[0] / img[2]).round().long()
    v = (img[1] / img[2]).round().long()

    valid = (depth > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, d = u[valid], v[valid], depth[valid]

    depth_map = torch.zeros(H, W)
    order = torch.argsort(d, descending=True)   
    depth_map[v[order], u[order]] = d[order]
    return depth_map                

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

class SSIM(nn.Module): 
    def __init__(self):
        super().__init__()
        self.mu_pool   = nn.AvgPool2d(3, 1)
        self.sig_pool  = nn.AvgPool2d(3, 1)
        self.pad = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x, y = self.pad(x), self.pad(y)
        mu_x, mu_y = self.mu_pool(x), self.mu_pool(y)

        sigma_x  = self.sig_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_pool(x * y) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1)

def photometric_loss(ssim_m, i_w, i_t, alpha=0.85):
    l1   = (i_w - i_t).abs().mean()
    ssim = ssim_m(i_w, i_t).mean() 
    return alpha * ssim + (1 - alpha) * l1

if __name__ == "__main__":
    target = load_image(target_loc)
    source = load_image(source_loc)
    _, _, H, W = target.shape

    K = parse_intrinsics(calib_loc, CAM)
    depth = torch.full((H, W), 20.0)

    ssim_m = SSIM()       

    identity = torch.eye(4)
    warped_identity = warp(source, depth, K, identity, H, W)
    diff = (warped_identity - source).abs().mean().item()
    print(f"identity-pose mean abs diff (should be ~0): {diff:.5f}")

    pose = torch.eye(4); pose[2, 3] = -1.0
    warped = warp(source, depth, K, pose, H, W)

    loss_identity = photometric_loss(ssim_m, warped_identity, target).item()
    loss_warped   = photometric_loss(ssim_m, warped,          target).item()
    print(f"photometric loss (identity warp vs target): {loss_identity:.5f}")
    print(f"photometric loss (moved warp   vs target): {loss_warped:.5f}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    for a, img, t in zip(ax, [target, source, warped],
                         ["target", "source", "warped"]):
        a.imshow(img[0].detach().cpu().permute(1, 2, 0).clamp(0, 1))
        a.set_title(t)
        a.axis("off")
    plt.savefig("warp_result.png")
    print("saved warp_result.png")