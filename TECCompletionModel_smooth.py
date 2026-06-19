import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import xarray as xr
import os

import sys

import glob
import time
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from ppgnss import gnss_utils, gnss_time
from const import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 或 "" 来禁用GPU

# 双分支卷积神经网络模型
class TECCompletionModel(nn.Module):
    def __init__(self):
        super(TECCompletionModel, self).__init__()
        
        # 背景分支处理 (低分辨率 17x14)，输出调整为(64, 201, 326)
        self.background_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (1, 17, 14) -> (32, 17, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (32, 17, 14) -> (32, 8, 7)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (32, 8, 7) -> (64, 8, 7)
            nn.ReLU(),
            nn.Upsample(size=(201, 326), mode='bilinear', align_corners=False), # (64, 8, 7) -> (64, 201, 326)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # (64, 201, 326) -> (64, 201, 326)
            nn.ReLU()
        )
        # 经过Sequential()处理后，background_branch的输出shape为 (64, 201, 326)
        
        # 观测分支处理 (双通道 201x326)
        # 针对输入尺寸201x326，建议适当增加卷积层深度和感受野，同时可适当下采样再上采样以提取更丰富特征
        self.observation_branch = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),   # (2, 201, 326) -> (32, 201, 326)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (32, 201, 326) -> (64, 201, 326)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (64, 201, 326) -> (64, 100, 163)
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (64, 100, 163) -> (128, 100, 163)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# (128, 100, 163) -> (128, 100, 163)
            nn.ReLU(),
            nn.Upsample(size=(201, 326), mode='bilinear', align_corners=False), # (128, 100, 163) -> (128, 201, 326)
            nn.Conv2d(128, 64, kernel_size=3, padding=1), # (128, 201, 326) -> (64, 201, 326)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # (64, 201, 326) -> (64, 201, 326)
            nn.ReLU()
        )
        
        # 特征融合与重建
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, background, observation):
        # 处理背景分支
        bg_features = self.background_branch(background)
        # 处理观测分支
        obs_features = self.observation_branch(observation)
        # print(bg_features.shape, obs_features.shape)
        # 特征拼接
        combined = torch.cat((bg_features, obs_features), dim=1)
        
        # 融合特征并重建
        output = self.fusion(combined)
        return output

# 残差块定义
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    
    def forward(self, x):
        return x + self.conv(x)

# 自定义数据集类
class TECDataset(Dataset):
    def __init__(self, background_files, tec_files, mask_files, mode='train'):
        self.background_files = background_files
        self.tec_files = tec_files
        self.mask_files = mask_files
        self.mode = mode
        self.ref_time = None
        
    def __len__(self):
        return len(self.tec_files)
    
    def __getitem__(self, idx):
        # 加载背景数据 (低分辨率)
        xr_iri = gnss_utils.loadobject(self.background_files[idx])
        
        # 修复：正确处理IRI数据维度
        iri_data = xr_iri.values
        
        # 如果数据是3维 [time=1, lat, lon]，去掉时间维度
        if iri_data.ndim == 3:
            if iri_data.shape[0] == 1:  # 只有一个时间点
                iri_data = iri_data[0]  # 去掉时间维度 -> [lat, lon]
            else:
                # 如果有多个时间点，取第一个
                iri_data = iri_data[0]
        
        # 确保数据是2维的 [lat, lon]
        if iri_data.ndim != 2:
            raise ValueError(f"IRI数据维度错误: {iri_data.shape}，期望2维 [lat, lon]")
        
        # 添加通道维度 [1, lat, lon] 以适应CNN输入
        background = torch.tensor(iri_data, dtype=torch.float32).unsqueeze(0)
        
        # 替换 NaN 为 0
        background[torch.isnan(background)] = 0
        
        # 加载观测数据和掩膜
        xr_grid = gnss_utils.loadobject(self.tec_files[idx])
        mask_data = gnss_utils.loadobject(self.mask_files[idx])
        
        # 创建观测分支的双通道输入
        obs_channel = xr_grid.fillna(0) * mask_data["train_mask"].fillna(0)
        mask_channel = mask_data["train_mask"].where(obs_channel.notnull(), 0)
        
        # 替换NaN值为0
        obs_channel = obs_channel.fillna(0)
        mask_channel = mask_channel.fillna(0)
        
        observation = torch.stack([
            torch.tensor(mask_channel.values, dtype=torch.float32),
            torch.tensor(obs_channel.values, dtype=torch.float32)
        ])
        
        # 创建标签
        label = torch.tensor(xr_grid.values, dtype=torch.float32).unsqueeze(0)
        label[torch.isnan(label)] = 0
        
        # 获取验证掩膜
        val_mask = torch.tensor(mask_data["val_mask"].values, dtype=torch.float32)
        val_mask[torch.isnan(val_mask)] = 0
        
        test_mask = torch.tensor(mask_data["test_mask"].values, dtype=torch.float32)
        test_mask[torch.isnan(test_mask)] = 0
        
        return background, observation, label, val_mask, test_mask










    # def __getitem__(self, idx):
    #     # 加载背景数据 (低分辨率)
    #     xr_iri = gnss_utils.loadobject(self.background_files[idx])
    #     self.ref_time = xr_iri.coords["time"].values[0]
        
    #     # 将IRI数据从经度坐标转换为时角坐标
    #     # 获取时间并计算时角偏移
    #     # if isinstance(self.ref_time, np.datetime64):
    #     #     ref_time_dt = pd.Timestamp(self.ref_time).to_pydatetime()
    #     # else:
    #     #     ref_time_dt = self.ref_time
    #     # hours_decimal = ref_time_dt.hour + ref_time_dt.minute/60 + ref_time_dt.second/3600
    #     # shift_degrees = hours_decimal * 15
        
    #     # # 获取经度坐标和分辨率
    #     # lon_coords = xr_iri.coords['lon'].values
    #     # lon_res = lon_coords[1] - lon_coords[0]  # 假设分辨率恒定
    #     # shift_points = int(round(shift_degrees / lon_res))  # 计算滚动点数
    #     # shift_points = 0
    #     # 滚动数据以适应时角坐标
    #     iri_data = xr_iri.values
    #     # if iri_data.ndim == 3:  # 包含时间维度
    #     #     iri_data_rolled = np.roll(iri_data, -shift_points, axis=2)  # 沿经度维度滚动
    #     # else:  # 只有纬度和经度
    #     #     iri_data_rolled = np.roll(iri_data, -shift_points, axis=1)  # 沿经度维度滚动
        
    #     # background = torch.tensor(iri_data_rolled, dtype=torch.float32)  # 添加通道维度
    #     background = torch.tensor(iri_data, dtype=torch.float32)  # 添加通道维度
    #     # print(background.shape)

    #     background[torch.isnan(background)] = 0  # 替换 NaN 为 0
        
    #     # 加载观测数据和掩膜
    #     xr_grid = gnss_utils.loadobject(self.tec_files[idx])
    #     mask_data = gnss_utils.loadobject(self.mask_files[idx])
    #     # print(xr_grid)
    #     # 创建观测分支的双通道输入
    #     obs_channel = xr_grid.fillna(0) * mask_data["train_mask"].fillna(0)
    #     # obs_channel = xr_grid * mask_data["train_mask"]

    #     mask_channel = mask_data["train_mask"].where(obs_channel.notnull(), 0)
    #     # plt.pcolor(xr_grid)
    #     # plt.savefig("obs_channel.png")
    #     # plt.close()
    #     # sys.exit(0)
    #     # 替换NaN值为0
    #     obs_channel = obs_channel.fillna(0)
    #     mask_channel = mask_channel.fillna(0)
        
    #     observation = torch.stack([
    #         torch.tensor(mask_channel.values, dtype=torch.float32),
    #         torch.tensor(obs_channel.values, dtype=torch.float32)
    #     ])
        
    #     # 创建标签
    #     label = torch.tensor(xr_grid.values, dtype=torch.float32).unsqueeze(0)
    #     label[torch.isnan(label)] = 0
    #     # 获取验证掩膜
    #     val_mask = torch.tensor(mask_data["val_mask"].values, dtype=torch.float32)
    #     val_mask[torch.isnan(val_mask)] = 0
        
    #     test_mask = torch.tensor(mask_data["test_mask"].values, dtype=torch.float32)
    #     test_mask[torch.isnan(test_mask)] = 0
    #     return background, observation, label, val_mask, test_mask


# 自定义损失函数
def masked_mse_loss(pred, target, mask):
    """仅在掩膜区域计算MSE损失"""
    # 确保mask与pred/target形状匹配
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)  # 添加通道维度
    
    # 应用掩膜
    masked_pred = pred * mask
    
    masked_target = target * mask
    
    
    # 计算MSE
    loss = F.mse_loss(masked_pred, masked_target, reduction='sum')
    valid_pixels = mask.sum()
    
    return loss / (valid_pixels + 1e-7)

def smoothness_loss(pred, distance=1, smooth_factor=0.1, order=2):
    """
    计算预测图像的平滑约束损失，使用卷积窗口考虑局部趋势
    
    参数:
    - pred: 预测图像 [batch_size, 1, height, width]
    - distance: 平滑约束的距离（决定卷积核大小）
    - smooth_factor: 平滑因子，控制平滑程度
    - order: 平滑阶数，1为一阶平滑，2为二阶平滑
    """
    batch_size, channels, height, width = pred.shape
    
    # 根据距离确定卷积核大小
    kernel_size = 2 * distance + 1
    
    if order == 1:
        # 一阶平滑：使用梯度约束
        # 水平方向一阶梯度
        sobel_x = torch.tensor([[-1, 0, 1], 
                               [-2, 0, 2], 
                               [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        # 垂直方向一阶梯度
        sobel_y = torch.tensor([[-1, -2, -1], 
                               [0, 0, 0], 
                               [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
    elif order == 2:
        # 二阶平滑：使用拉普拉斯算子约束曲率
        laplacian = torch.tensor([[0, 1, 0], 
                                 [1, -4, 1], 
                                 [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    
    # 将卷积核移动到与pred相同的设备
    device = pred.device
    if order == 1:
        sobel_x = sobel_x.to(device)
        sobel_y = sobel_y.to(device)
        # 计算梯度
        grad_x = F.conv2d(pred, sobel_x, padding=1)
        grad_y = F.conv2d(pred, sobel_y, padding=1)
        # 梯度平方和
        smooth_loss = torch.mean(grad_x**2) + torch.mean(grad_y**2)
        
    elif order == 2:
        laplacian = laplacian.to(device)
        # 计算拉普拉斯响应（曲率）
        curvature = F.conv2d(pred, laplacian, padding=1)
        smooth_loss = torch.mean(curvature**2)
    
    return smooth_factor * smooth_loss

# 更通用的版本：支持自定义距离和多种平滑方式
def advanced_smoothness_loss(pred, window_size=3, smooth_factor=0.1, mode='laplacian'):
    """
    高级平滑约束损失函数
    
    参数:
    - pred: 预测图像 [batch_size, 1, height, width]
    - window_size: 卷积窗口大小，必须为奇数
    - smooth_factor: 平滑因子
    - mode: 平滑模式，'laplacian'或'gaussian_laplacian'
    """
    assert window_size % 2 == 1, "窗口大小必须为奇数"
    
    if mode == 'laplacian':
        # 创建拉普拉斯卷积核
        kernel = torch.ones(1, 1, window_size, window_size, dtype=torch.float32)
        center = window_size // 2
        kernel[0, 0, center, center] = - (window_size * window_size - 1)
        kernel = kernel / (window_size * window_size)
        
    elif mode == 'gaussian_laplacian':
        # 高斯拉普拉斯算子（LoG）
        kernel = create_log_kernel(window_size)
    
    kernel = kernel.to(pred.device)
    
    # 应用卷积
    smooth_response = F.conv2d(pred, kernel, padding=window_size//2)
    
    # 计算损失
    smooth_loss = torch.mean(smooth_response**2)
    
    return smooth_factor * smooth_loss

def create_log_kernel(size, sigma=1.0):
    """创建高斯拉普拉斯卷积核"""
    kernel = torch.zeros(1, 1, size, size)
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            # 高斯拉普拉斯公式
            kernel[0, 0, i, j] = -(1 - (x**2 + y**2) / (2 * sigma**2)) * \
                                torch.exp(torch.tensor(-(x**2 + y**2) / (2 * sigma**2)))
    
    # 归一化，使核的和为0
    kernel = kernel - torch.mean(kernel)
    return kernel

# 组合损失函数
def combined_loss(pred, target, mask, smooth_window=1, smooth_factor=0.1, k=1):
    """
    组合MSE损失和平滑约束损失
    
    参数:
    - pred: 预测图像
    - target: 目标图像
    - mask: 掩膜
    - distance: 平滑约束的距离（像元数）
    - smooth_factor: 平滑因子
    """
    mse_loss = masked_mse_loss(pred, target, mask)
    # smooth_loss = smoothness_loss(pred, distance, smooth_factor)
    smooth_loss = advanced_smoothness_loss(pred, smooth_window, smooth_factor)
    
    total_loss = mse_loss + k * smooth_loss
    
    return total_loss, mse_loss, smooth_loss

# 训练函数
def train(model, device, train_loader, optimizer, epoch, smooth_window_size=1, smooth_factor=0.1, smooth_weight=1):
    model.train()
    total_loss = 0
    total_mse_loss = 0
    total_smooth_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (bg, obs, label, val_mask, test_mask) in enumerate(progress_bar):
        # print(bg.shape, obs.shape, label.shape)
        # print(obs[1,0,30:40, 60:70])
        # print(obs[1,1,30:40, 60:70])
        bg, obs, label, val_mask = bg.to(device), obs.to(device), label.to(device), val_mask.to(device)
        optimizer.zero_grad()
        output = model(bg, obs)
        # 使用组合损失函数
        loss, mse_loss, smooth_loss = combined_loss(output, label, val_mask, smooth_window_size, smooth_factor, smooth_weight)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        total_smooth_loss += smooth_loss.item()
        progress_bar.set_postfix({
            'loss': loss.item(), 
            'mse_loss': mse_loss.item(), 
            'smooth_loss': smooth_loss.item()
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_mse_loss = total_mse_loss / len(train_loader)
    avg_smooth_loss = total_smooth_loss / len(train_loader)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.6f}, MSE: {avg_mse_loss:.6f}, Smooth: {avg_smooth_loss:.6f}')
    return avg_loss, avg_mse_loss, avg_smooth_loss

# 验证函数
def validate(model, device, val_loader, distance=1, smooth_factor=0.1, smooth_weight=1):
    model.eval()
    val_loss = 0
    val_mse_loss = 0
    val_smooth_loss = 0
    
    with torch.no_grad():
        for bg, obs, label, val_mask, test_mask in tqdm(val_loader, desc='Validating'):
            bg, obs, label, val_mask = bg.to(device), obs.to(device), label.to(device), val_mask.to(device)
            output = model(bg, obs)
            loss, mse_loss, smooth_loss = combined_loss(output, label, val_mask, distance, smooth_factor, smooth_weight)
            val_loss += loss.item()
            val_mse_loss += mse_loss.item()
            val_smooth_loss += smooth_loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_mse_loss = val_mse_loss / len(val_loader)
    avg_val_smooth_loss = val_smooth_loss / len(val_loader)
    print(f'====> Validation loss: {avg_val_loss:.6f}, MSE: {avg_val_mse_loss:.6f}, Smooth: {avg_val_smooth_loss:.6f}')
    return avg_val_loss, avg_val_mse_loss, avg_val_smooth_loss

def test(model, device, test_loader, test_times, distance=1, smooth_factor=0.1):
    model.eval()
    test_loss = 0
    test_mse_loss = 0
    test_smooth_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for ibatch, (bg, obs, label, val_mask, test_mask) in enumerate(tqdm(test_loader, desc='Testing')):
            bg, obs, label, val_mask = bg.to(device), obs.to(device), label.to(device), val_mask.to(device)
            output = model(bg, obs)
            loss, mse_loss, smooth_loss = combined_loss(output, label, val_mask, distance, smooth_factor)
            test_loss += loss.item()
            test_mse_loss += mse_loss.item()
            test_smooth_loss += smooth_loss.item()
            
            # 保存结果用于可视化
            predictions.append(output.cpu())
            targets.append(label.cpu())

            # 获取当前批次的实际大小
            batch_size = bg.shape[0]
            
            # 只处理当前批次中存在的样本
            for ismp in range(batch_size):
                obs_mask = obs[ismp, 0, :, :].squeeze().cpu()
                # 计算正确的时间索引
                time_index = ibatch * 8 + ismp
                visualize_results(label.cpu()[ismp], output.cpu()[ismp], obs_mask, val_mask[ismp], test_mask[ismp], test_times[time_index])
            
    avg_test_loss = test_loss / len(test_loader)
    avg_test_mse_loss = test_mse_loss / len(test_loader)
    avg_test_smooth_loss = test_smooth_loss / len(test_loader)
    print(f'====> Test loss: {avg_test_loss:.6f}, MSE: {avg_test_mse_loss:.6f}, Smooth: {avg_test_smooth_loss:.6f}')
    
    return avg_test_loss, predictions, targets

# 结果可视化函数
def visualize_results(target, prediction, obs_mask, val_mask, test_mask, test_time):
    fig_dir = "results_final/figures"
    os.makedirs(fig_dir, exist_ok=True)  # 添加目录创建
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    diff0 = torch.abs(target - prediction).squeeze().numpy()
    diff0[obs_mask==0] = np.nan
    rms0 = np.sqrt(np.nanmean(np.square(diff0)))
    # 目标TEC图
    im0 = axes[0].pcolor(diff0, cmap='coolwarm', vmin=-5, vmax=5)
    axes[0].set_title(f'obs masked RMSE: {rms0:.2f} TECU')
    fig.colorbar(im0, ax=axes[0])
    
    # 预测TEC图
    im1 = axes[1].pcolor(prediction.squeeze().numpy(), cmap='viridis', vmin=0, vmax=120)
    axes[1].set_title('Predicted TEC')
    fig.colorbar(im1, ax=axes[1])
    
    # 差异图
    diff = torch.abs(target - prediction).squeeze().numpy()
    diff[val_mask.cpu()==0] = np.nan
    rms1 = np.sqrt(np.nanmean(np.square(diff)))
    im2 = axes[2].pcolor(diff, cmap='coolwarm', vmin=-5, vmax=5)
    axes[2].set_title(f'Val Masked RMSE: {rms1:.2f} TECU')
    fig.colorbar(im2, ax=axes[2])
    
    # 掩膜区域差异 mask.cpu().squeeze().numpy()
    diff1 = torch.abs(target - prediction).squeeze().numpy()
    diff1[test_mask.cpu()==0] = np.nan
    rms2 = np.sqrt(np.nanmean(np.square(diff1)))
    im3 = axes[3].pcolor(diff1, cmap='coolwarm', vmin=-5, vmax=5)
    axes[3].set_title(f'test Masked RMSE: {rms2:.2f} TECU')
    fig.colorbar(im3, ax=axes[3])
    # plt.title(f'{str(test_time)}')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'{str(test_time)}.png'))
    plt.close()

# 主函数
def main():
    trainning = False
    # 设置随机种子
    torch.manual_seed(43)
    np.random.seed(43)
    
    # 平滑约束参数
    smooth_window_size = SMOOTH_WINDOW_SIZE  # 平滑约束的距离（像元数）
    smooth_factor = SMOOTH_FAC  # 平滑因子，控制平滑程度
    smooth_weight = SMOOTH_WEIGHT  # 平滑loss相对于MSE loss 的权重
    year = 2023
    doy_from = 1
    doy_to = 361
    batch_size = 8
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Smoothness parameters - Smooth cells: {smooth_window_size}, Smooth Factor: {smooth_factor}")
    
    # 创建结果目录
    os.makedirs('results_final', exist_ok=True)
    os.makedirs('checkpoints_final', exist_ok=True)
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    # 加载数据文件路径 (实际使用时替换为真实路径)
    #data_dir = os.path.join(current_dir, "..", "data_new", "grids")
    data_dir = os.path.join(current_dir, "..", "data_final", "sgrids")

    # data_dir = "/home/nas/xwzheng/ss-dscnn/data/grids"
    background_files = []
    tec_files = []
    mask_files = []
    for doy in range(doy_from, doy_to+1):
        for hour in range(0, 24):
            for mi in range(0, 60, 10):
                ntime = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy-1) \
                    + datetime.timedelta(hours=hour) + datetime.timedelta(minutes=mi)
                # "2023-01-01T00:00:00Z"
                str_time = ntime.strftime("%Y-%m-%dT%H:%M")
                iri_file = os.path.join(data_dir, f"{str_time}_iri.obj")
                tec_file = os.path.join(data_dir, f"{str_time}_tec_slon.obj")
                mask_file = os.path.join(data_dir, f"{str_time}_mask.obj")
                # print(iri_file)
                if not os.path.isfile(iri_file) or not os.path.isfile(tec_file) or not os.path.isfile(mask_file): continue
                background_files.append(iri_file)
                tec_files.append(tec_file)
                mask_files.append(mask_file)
    
    # all_files = sorted(glob.glob(os.path.join(data_dir, "*.obj")))
    
    # background_files = [f for f in all_files if "iri" in f]
    # tec_files = [f for f in all_files if "tec" in f]
    # mask_files = [f for f in all_files if "mask" in f]

    # 确保文件数量匹配
    # assert len(background_files) == len(tec_files) == len(mask_files), "文件数量不匹配"
    print(f"找到 {len(background_files)} 个数据样本")
    # import sys
    # sys.exit(0)
    
    # 划分数据集 (60%训练, 20%验证, 20%测试) 
    total_samples = len(background_files)
    train_size = int(0.6 * total_samples)
    val_size = int(0.4 * total_samples)
    # test_size = total_samples - train_size - val_size
    
    indices = np.arange(total_samples)
    indices_in_order = np.arange(total_samples)
    # indices_in_order = np.arange(2*batch_size)
    # test_idx = indices[-test_size:]
    
    # np.random.shuffle(indices[:total_samples-test_size])
    np.random.shuffle(indices)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]

    # sys.exit(0)
    # 创建数据集
    train_dataset = TECDataset(
        [background_files[i] for i in train_idx],
        [tec_files[i] for i in train_idx],
        [mask_files[i] for i in train_idx],
        mode='train'
    )
    
    val_dataset = TECDataset(
        [background_files[i] for i in val_idx],
        [tec_files[i] for i in val_idx],
        [mask_files[i] for i in val_idx],
        mode='val'
    )
    
    # indices_in_order = indices_in_order[:16]
    test_dataset = TECDataset(
        [background_files[i] for i in indices_in_order],
        [tec_files[i] for i in indices_in_order],
        [mask_files[i] for i in indices_in_order],
        mode='test'
    )
    

        # 创建数据加载器
    
    # 创建数据加载器 - 丢弃不完整批次
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)


    # 初始化模型
    model = TECCompletionModel().to(device)
        
    # 打印模型结构
    # print(model)
        
    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    if trainning:
       
        

        # 训练参数
        num_epochs = 1000
        best_val_loss = float('inf')
        patience = 10  # 连续10个epoch没有改善就停止
        patience_counter = 0  # 没有改善的epoch计数器
        early_stop = False  # 早停标志

        # 训练历史记录
        train_history = []
        val_history = []
        train_mse_history = []
        val_mse_history = []
        train_smooth_history = []
        val_smooth_history = []

        # 训练循环
        for epoch in range(1, num_epochs + 1):
            # 早停检查
            if early_stop:
                print(f"Training stopped at epoch {epoch} due to early stopping")
                break
                
            start_time = time.time()
            
            # 训练和验证
            train_loss, train_mse, train_smooth = train(model, device, train_loader, optimizer, epoch, smooth_window_size, smooth_factor, smooth_weight)
            val_loss, val_mse, val_smooth = validate(model, device, val_loader, smooth_window_size, smooth_factor, smooth_weight)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 记录历史
            train_history.append(train_loss)
            val_history.append(val_loss)
            train_mse_history.append(train_mse)
            val_mse_history.append(val_mse)
            train_smooth_history.append(train_smooth)
            val_smooth_history.append(val_smooth)
            
            # 早停逻辑：保存最佳模型并更新计数器
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # 重置计数器
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'distance': smooth_window_size,
                    'smooth_factor': smooth_factor
                }, 'checkpoints_final/best_model.pth')
                print(f'✅ Saved new best model at epoch {epoch} with val loss {val_loss:.6f}')
            else:
                patience_counter += 1  # 没有改善，增加计数器
                print(f'⚠️ No improvement for {patience_counter}/{patience} epochs (best: {best_val_loss:.6f}, current: {val_loss:.6f})')
                
                # 检查是否触发早停
                if patience_counter >= patience:
                    early_stop = True
                    print(f'🛑 Early stopping triggered! No improvement for {patience} consecutive epochs.')
                    print(f'Best validation loss: {best_val_loss:.6f} at epoch {epoch - patience}')
            
            # 保存检查点
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'distance': smooth_window_size,
                    'smooth_factor': smooth_factor
                }, f'checkpoints_final/checkpoint_epoch_{epoch}.pth')
                print(f'💾 Checkpoint saved at epoch {epoch}')
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch} completed in {epoch_time:.2f} seconds')
            print('-' * 60)






        # # 训练循环
        # for epoch in range(1, num_epochs + 1):
        #     # 早停检查
        #     if patience_counter >= patience:
        #         print(f"Early stopping triggered at epoch {epoch} after {patience} epochs without improvement")
        #         break
        #     start_time = time.time()
            
        #     # 训练和验证
        #     train_loss, train_mse, train_smooth = train(model, device, train_loader, optimizer, epoch, smooth_window_size, smooth_factor, smooth_weight)
        #     val_loss, val_mse, val_smooth = validate(model, device, val_loader, smooth_window_size, smooth_factor, smooth_weight)
            
        #     # 更新学习率
        #     scheduler.step(val_loss)
            
        #     # 记录历史
        #     train_history.append(train_loss)
        #     val_history.append(val_loss)
        #     train_mse_history.append(train_mse)
        #     val_mse_history.append(val_mse)
        #     train_smooth_history.append(train_smooth)
        #     val_smooth_history.append(val_smooth)
            
        #     # 保存最佳模型并更新早停计数器
        #     if val_loss < best_val_loss:
        #         best_val_loss = val_loss
        #         patience_counter = 0  # 重置计数器
        #         torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'train_loss': train_loss,
        #             'val_loss': val_loss,
        #             'distance': smooth_window_size,
        #             'smooth_factor': smooth_factor
        #         }, 'checkpoints_new/best_model.pth')
        #         print(f'Saved new best model at epoch {epoch} with val loss {val_loss:.6f}')
        #     else:
        #         patience_counter += 1  # 没有改善，增加计数器
        #         print(f'No improvement in validation loss for {patience_counter} epochs (best: {best_val_loss:.6f}, current: {val_loss:.6f})')
            
        #     # 保存检查点
        #     if epoch % 10 == 0:
        #         torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'train_loss': train_loss,
        #             'val_loss': val_loss,
        #             'distance': smooth_window_size,
        #             'smooth_factor': smooth_factor
        #         }, f'checkpoints_new/checkpoint_epoch_{epoch}.pth')
            
        #     epoch_time = time.time() - start_time
        #     print(f'Epoch {epoch} completed in {epoch_time:.2f} seconds')
        #     # break
        
        # 绘制训练历史
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(train_history, label='Training Total Loss')
        plt.plot(val_history, label='Validation Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Total Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(train_mse_history, label='Training MSE Loss')
        plt.plot(val_mse_history, label='Validation MSE Loss')
        plt.plot(train_smooth_history, label='Training Smooth Loss')
        plt.plot(val_smooth_history, label='Validation Smooth Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Component Losses')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('results_final/training_history.png')
        plt.close()
    else:
        # 测试最佳模型
        print("Testing best model...")
        checkpoint = torch.load('checkpoints_final/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        # 从检查点加载平滑参数（如果存在）
        if 'distance' in checkpoint:
            smooth_window_size = checkpoint['distance']
        if 'smooth_factor' in checkpoint:
            smooth_factor = checkpoint['smooth_factor']
        print(f"Loaded model with distance={smooth_window_size}, smooth_factor={smooth_factor}")
        
        # 计算实际的时间点数量（舍弃不完整批次）
        total_batches = len(test_loader)  # 143个完整批次
        total_predictions = total_batches * batch_size  # 143 * 8 = 1144
        
        test_times = [datetime.datetime.strptime(fn[-29:-13], "%Y-%m-%dT%H:%M") for fn in test_dataset.tec_files]
        test_times = test_times[:total_predictions]  # 只取前1144个时间点
        
        print(f"原始时间点: {len(test_dataset.tec_files)}, 预测时间点: {len(test_times)}")
        print(f"完整批次: {total_batches}, 总预测样本: {total_predictions}")
        
        test_loss, predictions, targets = test(model, device, test_loader, test_times, smooth_window_size, smooth_factor)
        
        # 保存测试结果
        predicts_list = []
        targets_list = []
        lons = np.arange(LON_MIN, LON_MAX+RESOLUTION/2, RESOLUTION)
        lats = np.arange(LAT_MIN, LAT_MAX+RESOLUTION/2, RESOLUTION)
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            pred_np = pred.squeeze().numpy() if torch.is_tensor(pred) else pred.squeeze()
            target_np = target.squeeze().numpy() if torch.is_tensor(target) else target.squeeze()
            sample_time = test_times[i*8:(i+1)*8]  # 每个批次8个时间点
            
            predicts_list.extend(pred_np)
            targets_list.extend(target_np)
        
        # 创建最终的预测结果
        xr_preds = xr.DataArray(
            predicts_list, 
            dims = ["time", "lat", "lon"],
            coords={
                "time": test_times,
                "lat": lats,
                "lon": lons
            }
        )
        
        out_obj_fn = f'results_final/sgrid_test_results_{year:04d}_{doy_from:02d}_{doy_to:02d}.obj'
        gnss_utils.saveobject(xr_preds, out_obj_fn)
        print(f"{out_obj_fn} saved with {len(test_times)} time points.")

if __name__ == "__main__":
    main()