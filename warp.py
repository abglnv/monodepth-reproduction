import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.models import resnet18
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
    u = p[0] / p[2].clamp(min=1e-3) 
    v = p[1] / p[2].clamp(min=1e-3) 
    return u,v 

def warp(source_img, depth, K, pose, H, W):
    pix = make_pixel_grid(H, W)            
    cam_pts = backproject(depth.reshape(-1), K, pix)
    src_pts = transform(cam_pts, pose)
    u, v = project(src_pts, K)
    grid = to_grid_sample_coords(u, v, H, W)
    return F.grid_sample(source_img, grid, align_corners=True, padding_mode="border")

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

class DepthNetowkr(nn.Module): 
    def __init__(self):
        super().__init__()
        # encoder 
        enc = resnet18(weights=None)
        self.enc = nn.Sequential(*list(enc.children())[:-2]) # without avgpool + fc
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2),             
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),  nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),   nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1),   nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),    nn.Sigmoid(),   
        )

    def forward(self, x):
        feat = self.enc(x)
        disp = self.dec(feat)                  # sigmoid, (0, 1)
        # disp -> metric depth (monodepth2 disp_to_depth), 0.1 .. 100 m
        min_disp, max_disp = 1 / 100, 1 / 0.1
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        return 1 / scaled_disp

class PoseNetwork(nn.Module): 
    def __init__(self): 
        super().__init__()
        enc = resnet18(weights=None)
        enc.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.enc = nn.Sequential(*list(enc.children())[:-2])

        self.pose_head = nn.Conv2d(512, 6, kernel_size=1)

    def forward(self, img1, img2): 
        x = torch.cat([img1, img2], dim=1) 
        feat = self.enc(x)    
        out = self.pose_head(feat)  
        out = out.mean(dim=[2, 3])  

        # 0.01 scaling (monodepth2): keeps initial rotation/translation near identity
        axisangle = 0.01 * out[:, :3]
        translation = 0.01 * out[:, 3:]
        return axisangle, translation


# functions from monodepth2 
def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def photometric_loss(ssim_m, i_w, i_t, alpha=0.85):
    l1   = (i_w - i_t).abs().mean()
    ssim = ssim_m(i_w, i_t).mean() 
    return alpha * ssim + (1 - alpha) * l1

if __name__ == "__main__":
    target = load_image(target_loc)
    source = load_image(source_loc)
    _, _, H, W = target.shape

    K = parse_intrinsics(calib_loc, CAM)

    depth_net = DepthNetowkr()         
    pose_net  = PoseNetwork()
    ssim_m    = SSIM()

    params = list(depth_net.parameters()) + list(pose_net.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-5)

    for step in range(100):
        # 1. predict
        depth = depth_net(target)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)
        axisangle, translation = pose_net(target, source)
        axisangle   = axisangle.unsqueeze(1)
        translation = translation.unsqueeze(1)
        pose = transformation_from_parameters(axisangle, translation)[0]

        # 2. warp
        warped = warp(source, depth, K, pose, H, W)

        # 3. score
        loss = photometric_loss(ssim_m, warped, target)

        # 4. learn
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        g = depth_net.dec[1].weight.grad          # first decoder conv's gradient (adjust index if needed)
        print("grad is None:", g is None, "| grad norm:", None if g is None else g.norm().item())
        optimizer.step()

        if step % 20 == 0:
            print(f"step {step:4d}   loss {loss.item():.5f}")

    depth_net.eval()
    with torch.no_grad():
        depth = depth_net(target)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)

    depth_map = depth[0, 0].cpu()

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].imshow(target[0].permute(1, 2, 0).clamp(0, 1)); ax[0].set_title("target"); ax[0].axis("off")
    ax[1].imshow(depth_map, cmap="magma"); ax[1].set_title("predicted depth"); ax[1].axis("off")
    plt.savefig("depth_result.png"); print("saved depth_result.png")

    gt = lidar_to_depth(bin_loc, calib_cam, calib_velo, H, W, cam=CAM)
    mask = gt > 0
    abs_rel = ((depth_map[mask] - gt[mask]).abs() / gt[mask]).mean()
    print(f"abs_rel vs LiDAR: {abs_rel.item():.4f}")