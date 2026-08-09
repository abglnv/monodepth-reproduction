import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

CAM = 2
TRAIN_H, TRAIN_W = 192, 640     
EPOCHS = 10
BATCH_SIZE = 4
SMOOTHNESS_WEIGHT = 1e-3      
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CALIB = "./kitti_data/2011_09_26"

DRIVES = sorted(glob(f"{CALIB}/*_sync"))
DRIVE = f"{CALIB}/2011_09_26_drive_0001_sync" 
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

def load_image(path, size=None):
    img = read_image(path).float() / 255.0
    if size is not None:
        img = F.interpolate(img.unsqueeze(0), size=size, mode="bilinear", align_corners=False)
        return img
    return img.unsqueeze(0)

def parse_intrinsics(path, cam):
    with open(path) as f:
        for line in f:
            if line.startswith(f"P_rect_0{cam}:"):
                vals = [float(x) for x in line.split()[1:]]
                P = torch.tensor(vals).reshape(3, 4)
                return P[:, :3]

def scale_intrinsics(K, src_hw, dst_hw):
    (src_h, src_w), (dst_h, dst_w) = src_hw, dst_hw
    K = K.clone()
    K[0] *= dst_w / src_w
    K[1] *= dst_h / src_h
    return K

class KittiPairs(torch.utils.data.Dataset):
    def __init__(self, drives, cam, size):
        if isinstance(drives, str):
            drives = [drives]
        self.pairs = []
        for drive in drives:
            data_dir = os.path.join(drive, f"image_0{cam}", "data")
            files = sorted(glob(os.path.join(data_dir, "*.png")))
            self.pairs += [(files[i], files[i + 1]) for i in range(len(files) - 1)]
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        target, source = self.pairs[i]
        # load_image returns (1,3,H,W); drop the batch dim so DataLoader can add its own
        return load_image(target, self.size)[0], load_image(source, self.size)[0]

def make_pixel_grid(H, W):
    v, u = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    ones = torch.ones_like(u)
    return torch.stack([u, v, ones], dim=0).reshape(3, -1).float()

def to_grid_sample_coords(u, v, H, W):
    # u, v: (B, H*W)  ->  (B, H, W, 2)
    u_n = 2.0 * u / (W - 1) - 1.0
    v_n = 2.0 * v / (H - 1) - 1.0
    return torch.stack([u_n, v_n], dim=-1).reshape(-1, H, W, 2)

def backproject(depth, K, pix):
    # depth: (B, 1, H, W) or (B, H*W);  pix: (3, H*W)  ->  (B, 3, H*W)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    d = depth.reshape(depth.shape[0], -1)   # (B, N)
    u = pix[0]                              # (N,), broadcasts over the batch
    v = pix[1]

    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d

    return torch.stack([X, Y, Z], dim=1)

def transform(points, pose):
    # points: (B, 3, N),  pose: (B, 4, 4)  ->  (B, 3, N)
    B, _, N = points.shape
    ones = torch.ones(B, 1, N, device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=1)

    transformed_h = pose @ points_h
    return transformed_h[:, :3]

def project(points, K):
    # points: (B, 3, N),  K: (3, 3) broadcasts over the batch
    p = K @ points
    u = p[:, 0] / p[:, 2].clamp(min=1e-3)
    v = p[:, 1] / p[:, 2].clamp(min=1e-3)
    return u, v

def warp(source_img, depth, K, pose, H, W):
    pix = make_pixel_grid(H, W).to(depth.device)
    cam_pts = backproject(depth, K, pix)
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

def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU())

class DepthNetowkr(nn.Module):
    CH_ENC = [64, 64, 128, 256, 512]
    CH_DEC = [16, 32, 64, 128, 256]

    def __init__(self, use_skips=True):
        super().__init__()
        self.use_skips = use_skips
        enc = resnet18(weights="IMAGENET1K_V1")
        self.enc0 = nn.Sequential(enc.conv1, enc.bn1, enc.relu)   #  64, H/2
        self.enc1 = nn.Sequential(enc.maxpool, enc.layer1)        #  64, H/4
        self.enc2 = enc.layer2                                    # 128, H/8
        self.enc3 = enc.layer3                                    # 256, H/16
        self.enc4 = enc.layer4                                    # 512, H/32

        self.up    = nn.ModuleList()  
        self.merge = nn.ModuleList()  
        for i in range(4, -1, -1):
            cin = self.CH_ENC[-1] if i == 4 else self.CH_DEC[i + 1]
            self.up.append(conv_block(cin, self.CH_DEC[i]))
            skip_ch = self.CH_ENC[i - 1] if (use_skips and i > 0) else 0
            self.merge.append(conv_block(self.CH_DEC[i] + skip_ch, self.CH_DEC[i]))

        self.disp = nn.Sequential(nn.Conv2d(self.CH_DEC[0], 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        x = (x - 0.45) / 0.225               

        f0 = self.enc0(x)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        feats = [f0, f1, f2, f3]

        y = self.enc4(f3)
        for j, i in enumerate(range(4, -1, -1)):
            y = self.up[j](y)
            y = F.interpolate(y, scale_factor=2, mode="nearest")
            if self.use_skips and i > 0:
                y = torch.cat([y, feats[i - 1]], dim=1)
            y = self.merge[j](y)

        disp = self.disp(y)                    # sigmoid, (0, 1)
        min_disp, max_disp = 1 / 100, 1 / 0.1
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        return disp, 1 / scaled_disp

class PoseNetwork(nn.Module): 
    def __init__(self): 
        super().__init__()
        enc = resnet18(weights="IMAGENET1K_V1")
        w1 = enc.conv1.weight.data
        enc.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        enc.conv1.weight.data = torch.cat([w1, w1], dim=1) / 2
        self.enc = nn.Sequential(*list(enc.children())[:-2])

        self.pose_head = nn.Conv2d(512, 6, kernel_size=1)

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], dim=1)
        x = (x - 0.45) / 0.225                
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

def get_smooth_loss(disp, img):
    grad_disp_x = (disp[:, :, :, :-1] - disp[:, :, :, 1:]).abs()
    grad_disp_y = (disp[:, :, :-1, :] - disp[:, :, 1:, :]).abs()

    grad_img_x = (img[:, :, :, :-1] - img[:, :, :, 1:]).abs().mean(1, keepdim=True)
    grad_img_y = (img[:, :, :-1, :] - img[:, :, 1:, :]).abs().mean(1, keepdim=True)

    grad_disp_x = grad_disp_x * torch.exp(-grad_img_x)
    grad_disp_y = grad_disp_y * torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def smoothness_loss(disp, img):
    mean_disp = disp.mean(2, True).mean(3, True)
    return get_smooth_loss(disp / (mean_disp + 1e-7), img)

def photometric_loss(ssim_m, i_w, i_t, alpha=0.85):
    l1   = (i_w - i_t).abs().mean()
    ssim = ssim_m(i_w, i_t).mean() 
    return alpha * ssim + (1 - alpha) * l1

if __name__ == "__main__":
    H, W = TRAIN_H, TRAIN_W

    dataset = KittiPairs(DRIVES, CAM, (H, W))
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    full_h, full_w = read_image(dataset.pairs[0][0]).shape[-2:]
    K = scale_intrinsics(parse_intrinsics(calib_loc, CAM), (full_h, full_w), (H, W)).to(DEVICE)

    depth_net = DepthNetowkr().to(DEVICE)
    pose_net  = PoseNetwork().to(DEVICE)
    ssim_m    = SSIM().to(DEVICE)

    params = list(depth_net.parameters()) + list(pose_net.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)

    for epoch in range(EPOCHS):
        running = 0.0
        for target, source in loader:
            target, source = target.to(DEVICE), source.to(DEVICE)

            # 1. predict
            disp, depth = depth_net(target)
            depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)
            axisangle, translation = pose_net(target, source)
            axisangle   = axisangle.unsqueeze(1)
            translation = translation.unsqueeze(1)
            pose = transformation_from_parameters(axisangle, translation)   # (B, 4, 4)

            # 2. warp
            warped = warp(source, depth, K, pose, H, W)

            # 3. score
            photo  = photometric_loss(ssim_m, warped, target)
            smooth = smoothness_loss(disp, target)
            loss   = photo + SMOOTHNESS_WEIGHT * smooth

            # 4. learn
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            running += loss.item()

        print(f"epoch {epoch:3d}   mean loss {running / len(loader):.5f}")

    target_full = load_image(target_loc)                 # full res, for display + GT comparison
    full_h, full_w = target_full.shape[-2:]

    depth_net.eval()
    with torch.no_grad():
        _, depth = depth_net(load_image(target_loc, (H, W)).to(DEVICE))
        depth = F.interpolate(depth, size=(full_h, full_w), mode="bilinear", align_corners=False)

    depth_map = depth[0, 0].cpu()
    target = target_full
    H, W = full_h, full_w

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].imshow(target[0].permute(1, 2, 0).clamp(0, 1)); ax[0].set_title("target"); ax[0].axis("off")
    ax[1].imshow(depth_map, cmap="magma"); ax[1].set_title("predicted depth"); ax[1].axis("off")
    plt.savefig("depth_result.png"); print("saved depth_result.png")

    gt = lidar_to_depth(bin_loc, calib_cam, calib_velo, H, W, cam=CAM)
    mask = gt > 0
    pred, g = depth_map[mask], gt[mask]

    ratio = g.median() / pred.median()
    abs_rel_raw = ((pred - g).abs() / g).mean()
    abs_rel = ((pred * ratio - g).abs() / g).mean()
    print(f"abs_rel vs LiDAR: {abs_rel.item():.4f} (median-scaled, scale {ratio.item():.2f})")
    print(f"  raw, unscaled:  {abs_rel_raw.item():.4f}   <- dominated by scale, not depth quality")