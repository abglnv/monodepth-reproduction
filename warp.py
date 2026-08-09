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
EPOCHS = 40
BATCH_SIZE = 4
SMOOTHNESS_WEIGHT = 1e-3
AUTOMASK = True                
NUM_SCALES = 4                
EVAL_STRIDE = 3                 
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

MIN_DEPTH, MAX_DEPTH = 1e-3, 80.0

def compute_errors(gt, pred):
    thresh = torch.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    rmse     = ((gt - pred) ** 2).mean().sqrt()
    rmse_log = ((gt.log() - pred.log()) ** 2).mean().sqrt()
    abs_rel  = ((gt - pred).abs() / gt).mean()
    sq_rel   = (((gt - pred) ** 2) / gt).mean()
    return torch.stack([abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])

METRIC_NAMES = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]

def garg_crop(h, w):
    m = torch.zeros(h, w, dtype=torch.bool)
    m[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)] = True
    return m

def eval_frames(drives, cam):
    out = []
    for drive in drives:
        for img in sorted(glob(os.path.join(drive, f"image_0{cam}", "data", "*.png"))):
            b = os.path.join(drive, "velodyne_points", "data",
                             os.path.basename(img).replace(".png", ".bin"))
            if os.path.exists(b):
                out.append((img, b))
    return out

@torch.no_grad()
def evaluate_depth(depth_net, frames, size, device, stride=1):
    was_training = depth_net.training
    depth_net.eval()
    total, n = torch.zeros(7), 0
    for img_path, bin_path in frames[::stride]:
        full_h, full_w = read_image(img_path).shape[-2:]
        gt = lidar_to_depth(bin_path, calib_cam, calib_velo, full_h, full_w, cam=CAM)

        disp = depth_net(load_image(img_path, size).to(device))[0]   # scale 0, full res
        _, pred = disp_to_depth(disp)
        pred = F.interpolate(pred, size=(full_h, full_w), mode="bilinear",
                             align_corners=False)[0, 0].cpu()

        mask = (gt > MIN_DEPTH) & (gt < MAX_DEPTH) & garg_crop(full_h, full_w)
        if mask.sum() == 0:
            continue
        g, p = gt[mask], pred[mask]
        p = p * (g.median() / p.median())          # median scaling, per frame
        p = p.clamp(MIN_DEPTH, MAX_DEPTH)
        total += compute_errors(g, p)
        n += 1
    if was_training:
        depth_net.train()
    return total / max(n, 1), n

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

class KittiTriplets(torch.utils.data.Dataset):
    def __init__(self, drives, cam, size):
        if isinstance(drives, str):
            drives = [drives]
        self.triplets = []
        for drive in drives:
            data_dir = os.path.join(drive, f"image_0{cam}", "data")
            files = sorted(glob(os.path.join(data_dir, "*.png")))
            self.triplets += [(files[i], files[i - 1], files[i + 1])
                              for i in range(1, len(files) - 1)]
        self.size = size

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, i):
        t, p, n = self.triplets[i]
        return (load_image(t, self.size)[0],
                load_image(p, self.size)[0],
                load_image(n, self.size)[0])

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

def disp_to_depth(disp, min_depth=0.1, max_depth=100.0):
    min_disp, max_disp = 1 / max_depth, 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    return scaled_disp, 1 / scaled_disp

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

        self.disp = nn.ModuleList([
            nn.Sequential(nn.Conv2d(self.CH_DEC[i], 1, 3, padding=1), nn.Sigmoid())
            for i in range(NUM_SCALES)
        ])

    def forward(self, x):
        x = (x - 0.45) / 0.225               

        f0 = self.enc0(x)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        feats = [f0, f1, f2, f3]

        y = self.enc4(f3)
        disps = {}
        for j, i in enumerate(range(4, -1, -1)):
            y = self.up[j](y)
            y = F.interpolate(y, scale_factor=2, mode="nearest")
            if self.use_skips and i > 0:
                y = torch.cat([y, feats[i - 1]], dim=1)
            y = self.merge[j](y)
            if i < NUM_SCALES:                 # i = 3,2,1,0 -> H/8, H/4, H/2, H
                disps[i] = self.disp[i](y)

        return [disps[s] for s in range(NUM_SCALES)]

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

def reprojection_loss(ssim_m, pred, target, alpha=0.85):
    l1   = (pred - target).abs().mean(1, keepdim=True)
    ssim = ssim_m(pred, target).mean(1, keepdim=True)
    return alpha * ssim + (1 - alpha) * l1

def min_reprojection_loss(ssim_m, warpeds, sources, target, automask=True):
    losses = torch.cat([reprojection_loss(ssim_m, p, target) for p in warpeds], dim=1)

    if automask:
        identity = torch.cat([reprojection_loss(ssim_m, s, target) for s in sources], dim=1)
        identity = identity + torch.randn_like(identity) * 1e-5   # break exact ties
        losses = torch.cat([identity, losses], dim=1)

    to_optimise, _ = losses.min(dim=1)
    return to_optimise.mean()

if __name__ == "__main__":
    H, W = TRAIN_H, TRAIN_W

    dataset = KittiTriplets(DRIVES, CAM, (H, W))
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    frames = eval_frames(DRIVES, CAM)
    print(f"{len(dataset)} triplets from {len(DRIVES)} drives at {H}x{W}, "
          f"batch {BATCH_SIZE}, device {DEVICE}")
    print(f"{len(frames[::EVAL_STRIDE])} eval frames (in-sample: these drives are also trained on)")

    full_h, full_w = read_image(dataset.triplets[0][0]).shape[-2:]
    K = scale_intrinsics(parse_intrinsics(calib_loc, CAM), (full_h, full_w), (H, W)).to(DEVICE)

    depth_net = DepthNetowkr().to(DEVICE)
    pose_net  = PoseNetwork().to(DEVICE)
    ssim_m    = SSIM().to(DEVICE)

    params = list(depth_net.parameters()) + list(pose_net.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)

    best_absrel, best_state = float("inf"), None
    for epoch in range(EPOCHS):
        running = 0.0
        for target, prev, nxt in loader:
            target, prev, nxt = target.to(DEVICE), prev.to(DEVICE), nxt.to(DEVICE)

            disps = depth_net(target)

            aa_p, t_p = pose_net(prev, target)
            aa_n, t_n = pose_net(target, nxt)
            pose_prev = transformation_from_parameters(aa_p.unsqueeze(1), t_p.unsqueeze(1), invert=True)
            pose_next = transformation_from_parameters(aa_n.unsqueeze(1), t_n.unsqueeze(1))

            loss = 0.0
            for s, disp in enumerate(disps):
                disp_up = disp if s == 0 else F.interpolate(
                    disp, size=(H, W), mode="bilinear", align_corners=False)
                _, depth = disp_to_depth(disp_up)

                warped = [warp(prev, depth, K, pose_prev, H, W),
                          warp(nxt,  depth, K, pose_next, H, W)]
                photo = min_reprojection_loss(ssim_m, warped, [prev, nxt], target, automask=AUTOMASK)

                color_s = target if s == 0 else F.interpolate(
                    target, size=disp.shape[-2:], mode="bilinear", align_corners=False)
                smooth = smoothness_loss(disp, color_s)

                loss = loss + photo + SMOOTHNESS_WEIGHT * smooth / (2 ** s)

            loss = loss / len(disps)

            # 4. learn
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            running += loss.item()

        metrics, n_eval = evaluate_depth(depth_net, frames, (H, W), DEVICE, stride=EVAL_STRIDE)
        line = "  ".join(f"{k} {v:.4f}" for k, v in zip(METRIC_NAMES, metrics.tolist()))
        print(f"epoch {epoch:3d}  loss {running / len(loader):.5f}  |  {line}")

        if metrics[0] < best_absrel:                      # checkpoint on abs_rel
            best_absrel = metrics[0].item()
            best_state = {k: v.detach().cpu().clone() for k, v in depth_net.state_dict().items()}

    print(f"\nbest abs_rel {best_absrel:.4f} over {n_eval} frames")
    depth_net.load_state_dict(best_state)
    torch.save(best_state, "depth_net_best.pt")
    print("saved depth_net_best.pt")

    target_full = load_image(target_loc)
    full_h, full_w = target_full.shape[-2:]

    depth_net.eval()
    with torch.no_grad():
        disp = depth_net(load_image(target_loc, (H, W)).to(DEVICE))[0]
        disp = F.interpolate(disp, size=(full_h, full_w), mode="bilinear", align_corners=False)
    disp_map = disp[0, 0].cpu().numpy()

    vmax = np.percentile(disp_map, 95)

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].imshow(target_full[0].permute(1, 2, 0).clamp(0, 1)); ax[0].set_title("target"); ax[0].axis("off")
    ax[1].imshow(disp_map, cmap="magma", vmin=disp_map.min(), vmax=vmax)
    ax[1].set_title("predicted disparity (bright = near)"); ax[1].axis("off")
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/depth_result.png"); print("saved assets/depth_result.png")