import os
import numpy as np
import xarray as xr
import pandas as pd
import datetime

from const import *
from ppgnss import gnss_utils
# from read_iono import show_stec
import matplotlib.pyplot as plt


def lon2slon1(lon, hour_dec):
    """
    经度+时间转时角（以太阳直射为0度）
    """
    return lon + hour_dec * 15 - 180


def lon2slon(lon, dt):
    """
    经度+时间转时角（以太阳直射为0度）
    """
    hour_dec = dt.hour + dt.minute/60. + dt.second/3600.
    return lon + hour_dec * 15 - 180

def pd_vtec2grid(pv_vtec_slon, lat_lon_box, current_time, resolution=0.5, xcoord="slon"):
    pd_vtec_filter = outlier_fit(pv_vtec_slon, current_time, xcoord=xcoord, norder=7, threshold=3.)
    # print(len(pd_vtec_filter[~pd_vtec_filter["valid"]]))
    pd_vtec_filter = pd_vtec_filter[pd_vtec_filter["valid"]]
    # valid_mask = pd_vtec_filter["valid"]
    hour_dec = current_time.hour + current_time.minute/60. + current_time.second/3600.
    if xcoord == "slon":
        x = pd_vtec_filter["slon"]
        lat_min, lat_max, lon_min, lon_max = lat_lon_box
        x_min, x_max = lon2slon(lon_min, current_time), lon2slon(lon_max, current_time)
    else:
        x = pd_vtec_filter["lon"]
        lat_min, lat_max, x_min, x_max = lat_lon_box
    
    points = np.array([x, pd_vtec_filter["lat"], pd_vtec_filter["vtec"]]).T
    shape = (int((x_max - x_min)/resolution)+1, int((lat_max - lat_min)/resolution)+1)
    xx = np.arange(x_min, x_max+resolution/2, resolution)
    yy = np.arange(lat_min, lat_max+resolution/2, resolution)
    
    data, inds = gnss_utils.points2grids(points, (x_min, lat_min), shape, (resolution, resolution))
    

    
    xr_data_slon = xr.DataArray(
        data["mean"],
        dims=("lat", "slon"),
        coords={"lat": yy, "slon": xx}
    )
    return xr_data_slon
    # fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
    # sc0 = axes[0].scatter(pd_vtec_filter["slon"], 
    #                      pd_vtec_filter["lat"], 
    #                      c=pd_vtec_filter["vtec"],
    #                      s=1)
    # cb0 = plt.colorbar(sc0, ax=axes[0])
    # cb0.set_label("VTEC")
    # pc1 = axes[1].pcolor(xx, yy, data["mean"])
    # cb1 = plt.colorbar(pc1, ax=axes[1])
    # cb1.set_label("Mean VTEC")
    # pc2 = axes[2].pcolor(xx, yy, data["max"]-data["min"])
    # cb2 = plt.colorbar(pc2, ax=axes[2])
    # cb2.set_label("Max-Min VTEC")
    # pc3 = axes[3].pcolor(xx, yy, data["count"])
    # cb3 = plt.colorbar(pc3, ax=axes[3])
    # cb3.set_label("Count")
    
    # pc4 = axes[4].pcolor(xx, yy, data["std"])
    # cb4 = plt.colorbar(pc4, ax=axes[4])
    # cb4.set_label("Std VTEC")
    
    # # plt.pcolor(xx, yy, data["mean"])
    # plt.savefig("grid_slon.png")
    # plt.close()


def outlier_fit(pd_vtec, current_time, xcoord="slon", norder=3, threshold=2.5):
    """
    使用球冠谐函数（Spherical Cap Harmonics）拟合剔除粗差，阶数为norder
    """
    import numpy as np
    from scipy.special import lpmv

    hour_dec = current_time.hour + current_time.minute/60. + current_time.second/3600.
    if xcoord == "slon":
        lon = pd_vtec["slon"].values
    else:
        lon = pd_vtec["lon"].values
    lat = pd_vtec["lat"].values
    z = pd_vtec["vtec"].values

    # 将经纬度转换为弧度
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    # 球冠谐函数基函数生成
    def sph_cap_harmonics(lat_rad, lon_rad, nmax):
        """
        生成球冠谐函数基函数矩阵
        """
        # 球冠谐函数的theta为余纬（0在北极，pi在南极）
        theta = np.pi/2 - lat_rad
        phi = lon_rad
        terms = []
        for n in range(nmax+1):
            for m in range(0, n+1):
                # 归一化缔合勒让德多项式
                Pnm = lpmv(m, n, np.cos(theta))
                # m=0时只加一次，m>0时加cos和sin两项
                if m == 0:
                    terms.append(Pnm)
                else:
                    terms.append(Pnm * np.cos(m*phi))
                    terms.append(Pnm * np.sin(m*phi))
        return np.vstack(terms).T

    X = sph_cap_harmonics(lat_rad, lon_rad, norder)

    # 只对有限有效数据做拟合
    mask_valid = np.isfinite(z) & np.all(np.isfinite(X), axis=1)
    X_valid = X[mask_valid]
    z_valid = z[mask_valid]
    if len(z_valid) < X.shape[1]:
        # 数据点太少，全部标记为无效
        pd_vtec["valid"] = False
        pd_vtec["res"] = np.nan
        return pd_vtec
    coef, *_ = np.linalg.lstsq(X_valid, z_valid, rcond=None)
    z_fit = X @ coef
    # 计算残差和中误差
    residual = z - z_fit
    sigma = np.nanstd(residual)
    # 标记有效性
    valid = np.abs(residual) <= threshold * sigma
    pd_vtec["res"] = residual
    pd_vtec["valid"] = valid
    return pd_vtec
    

def stec2grid(xr_stec_sel, lat_lon_box, current_time, resolution=0.5):
    # 1. 过滤高度角低于10°的数据
    ele = xr_stec_sel.sel(data='ele')
    mask = ele >= 10  # 高度角≥10°的位置为True
    filtered = xr_stec_sel.where(mask)

    # 2. 提取必要变量并计算TEC
    stec = filtered.sel(data='stec').drop_vars("data")
    mf = filtered.sel(data='mf').drop_vars("data")
    lon_vals = filtered.sel(data='lon').drop_vars("data")  # 重命名避免冲突
    lat_vals = filtered.sel(data='lat').drop_vars("data")  # 重命名避免冲突
    
    # 将经度转换为时角（经度减去当前时间转换为度）
    # 24小时对应360°，所以1小时=15°，1分钟=0.25°
    hours_decimal = current_time.hour + current_time.minute/60. + current_time.second/3600.
    longitude_hour_angle = lon_vals + hours_decimal * 15 
    longitude_hour_angle = longitude_hour_angle % 360 - 180

    # 计算TEC = STEC / MF，处理除零错误
    with np.errstate(divide='ignore', invalid='ignore'):
        tec = stec / mf
        tec = xr.where((mf == 0) | np.isnan(mf), np.nan, tec)

    # 3. 计算格网索引（修改点1：使用resolution参数）
    scale = 1 / resolution
    # 获取格网范围（核心修改：始终使用lat_lon_box创建固定格网）
    lat_min, lat_max, lon_min, lon_max = lat_lon_box
    slon_min, slon_max = lon2slon(lon_min, current_time), lon2slon(lon_max, current_time)
    lon_min_idx = int(np.floor(lon_min * scale))
    lon_max_idx = int(np.floor(lon_max * scale)) + 1
    
    slon_min_idx = int(np.floor(slon_min * scale))
    slon_max_idx = int(np.floor(slon_max * scale)) + 1
    
    lat_min_idx = int(np.floor(lat_min * scale))
    lat_max_idx = int(np.floor(lat_max * scale)) + 1
    # 4. 创建新的DataArray用于格网计算
    # 使用不同名称避免冲突
    grid_data = xr.Dataset({
        'tec_value': tec,  # 避免使用'tec'以免与现有坐标冲突
        'lon_value': lon_vals,  # 避免使用'lon'
        'lat_value': lat_vals  # 避免使用'lat'
    })
    valid_mask = np.isfinite(longitude_hour_angle) & np.isfinite(lat_vals)
    lon_vals_filtered = longitude_hour_angle.where(valid_mask, other=np.nan)
    lat_vals_filtered = lat_vals.where(valid_mask, other=np.nan)
    # 使用过滤后的值计算索引
    lon_idx = np.floor(lon_vals_filtered * scale).fillna(-9999).astype(int)
    lat_idx = np.floor(lat_vals_filtered * scale).fillna(-9999).astype(int)


    # 添加格网坐标
    grid_data = grid_data.assign_coords({
        'lon_grid': lon_idx,
        'lat_grid': lat_idx
    })

    # 5. 按格网分组并计算平均TEC
    # 堆叠所有维度以便分组
    stacked = grid_data.stack(all_points=('site', 'time', 'satellite'))
    
    valid_mask = (stacked['lon_grid'] != -9999) & (stacked['lat_grid'] != -9999)
    stacked = stacked.where(valid_mask, drop=True)

    # 按格网分组并计算平均值
    grouped = stacked.groupby(['lat_grid', 'lon_grid'])
    # grouped.median
    mean_tec = grouped.median(skipna=True)['tec_value']  # 跳过NaN值计算中位数
    # 创建实际地理坐标网格（核心修改）
    lon_centers = np.arange(lon_min_idx, lon_max_idx) * resolution + resolution/2
    lat_centers = np.arange(lat_min_idx, lat_max_idx) * resolution + resolution/2

    full_grid = xr.DataArray(
        dims=('lat', 'slon'),  # 使用有意义的维度名称，经度改为时角slon
        coords={
            'lat': lat_centers,   # 实际纬度中心点
            'slon': lon_centers  # 时角坐标
        },
        data=np.nan
    )
    # print(full_grid.shape, len(lon_centers), len(lat_centers), mean_tec.shape)
    # 将计算结果映射到完整格网
    # 首先将mean_tec的索引转换为实际坐标
    mean_tec = mean_tec.assign_coords({
        'lat_grid': mean_tec['lat_grid'] * resolution + resolution/2,
        'lon_grid': mean_tec['lon_grid'] * resolution + resolution/2
    }).rename({'lat_grid': 'lat', 'lon_grid': 'slon'})
    
    mean_tec = mean_tec.sel(lat=slice(lat_min, lat_max),
                            slon=slice(lon_min, lon_max))
    result = full_grid.combine_first(mean_tec)
    # print(result.shape)
    # result = xr.combine_by_coords(full_grid, mean_tec)
    # 移除不必要的坐标（修改点3）
    result.name = 'gridded_tec'
    result.attrs = {
        'units': 'TECu',
        'description': f'Averaged TEC in {resolution}° grids',
        'processing': 'Height angle >= 10°, TEC = STEC/MF',
        'original_longitude_range': f'{lon_min} to {lon_max} degrees',
        'hour_angle_reference_time': current_time.isoformat(),
        'conversion_note': 'slon is hour angle in degrees, convert back to longitude using: longitude = slon + (hour + minute/60 + second/3600) * 15'
    }
    # print(result.coords["lon"])
    # print(result.shape)
    plt.pcolor(result.values)
    plt.savefig("grid.png")
    plt.close()
    return result


def create_masks(gridded_tec, llbox, ref_epoch, erase_grid_size=2.0, train_ratio=0.7, val_ratio=0.15, xcoord="slon"):
    """
    生成数据掩膜：
    1. 基本掩膜：非NaN区域为1，NaN区域为0
    2-4. 训练、验证、测试掩膜：在interval°×interval°网格上生成后升采样到0.5°×0.5°
    
    参数:
        gridded_tec (xr.DataArray): 网格化的TEC数据
        interval (float): 采样网格大小(度)
        train_ratio (float): 训练数据比例
        val_ratio (float): 验证数据比例
        
    返回:
        masks (xr.Dataset): 包含4个掩膜的数据集
    """
    # print(gridded_tec.lat)
    # print(gridded_tec.slon)
    # 1. 创建基本掩膜 (nan为0，其他为1)
    base_mask = xr.where(gridded_tec.notnull(), 1, 0)
    base_mask.name = "base_mask"
    
    # 2. 创建低分辨率网格用于掩膜分配
    lat_min, lat_max, x_min, x_max = llbox
    x_min, x_max = lon2slon(x_min, ref_epoch), lon2slon(x_max, ref_epoch)
    lowres_lat = np.arange(lat_min, lat_max + erase_grid_size, erase_grid_size)
    lowres_slon = np.arange(x_min, x_max + erase_grid_size, erase_grid_size)
    
    # 初始化低分辨率掩膜
    lowres_mask = xr.DataArray(
        np.zeros((len(lowres_lat), len(lowres_slon))),
        dims=['lat', 'slon'],
        coords={'lat': lowres_lat, 'slon': lowres_slon}
    )

    # 3. 在低分辨率网格上随机分配掩膜
    # 创建类别数组：0=未分配, 1=训练, 2=验证, 3=测试
    categories = np.zeros(lowres_mask.shape)
    n_cells = categories.size
    
    # 计算各掩膜需要的网格数
    n_train = int(n_cells * train_ratio)
    n_val = int(n_cells * val_ratio)
    n_test = n_cells - n_train - n_val
    
    # 创建随机分配数组
    assignments = np.concatenate([
        np.ones(n_train),          # 训练
        np.ones(n_val) * 2,        # 验证
        np.ones(n_test) * 3        # 测试
    ])
    np.random.shuffle(assignments)
    
    # 应用分配
    categories.flat[:] = assignments
    lowres_mask[:] = categories.reshape(lowres_mask.shape)
    # 4. 将低分辨率掩膜升采样到原始分辨率
    # 使用最近邻插值保持类别不变
    highres_mask = lowres_mask.interp(
        lat=gridded_tec.lat,
        slon=gridded_tec.slon,  # 改为 slon
        method='nearest',
        kwargs={"fill_value": 0}
    )
    
    
    # 5. 创建最终掩膜
    train_mask = xr.where(highres_mask == 1, 1, 0) * base_mask
    val_mask = xr.where(highres_mask == 2, 1, 0) * base_mask
    test_mask = xr.where(highres_mask == 3, 1, 0) * base_mask
    
    # 创建数据集
    masks = xr.Dataset({
        "base_mask": base_mask,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask
    })
    
    # 添加属性
    masks.attrs = {
        "description": f"TEC data masks generated on {erase_grid_size}° grid and upsampled",
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1 - train_ratio - val_ratio,
        "original_resolution": f"{gridded_tec.lat.diff('lat').values[0]}° x {gridded_tec.slon.diff('slon').values[0]}°",
        "mask_resolution": f"{erase_grid_size}° x {erase_grid_size}°"
    }
    
    return masks


def interpolate_iri_to_10min(target_time, xr_iri, llbox):
    """
    将IRI数据插值到指定的10分钟时间戳，并截取经纬度范围
    
    参数:
        target_time (datetime): 目标时间戳 (10分钟分辨率)
        xr_iri (xr.DataArray): 原始IRI数据 (1小时分辨率)
        llbox (list): 经纬度范围 [lat_min, lat_max, lon_min, lon_max]
        
    返回:
        xr_slice (xr.DataArray): 插值后的数据切片
    """

    # 0. xr_iri的三个坐标为：time, lat, lon。
    # 0.1 当target 在 xr_iri.time中 时，则选择 time=target_time 切片
    # 0.2 当target 不在 xr_iri.time 中时，则首先选择 target_time 前、后时间切片（分别为xr_iri_previous, xr_iri_next)
    # 0.3 计算前、后时间切片与目标时间戳的时间差，根据时间差对切片进行平移
    # 0.4 根据时间差，转换为浮点小时数，计算xr_iri_previous何xr_iri_next 的平移参数。平移参数计算方式
    #  int(delta(hour) * 15 / dlon), dlon 为 xr_iri 的lon分辨率（或间隔） 

    import numpy as np
    import xarray as xr

    # 1. 检查目标时间是否在xr_iri的time坐标中
    if np.datetime64(target_time) in xr_iri.time.values:
        # 直接选择该时间切片
        xr_slice = xr_iri.sel(time=target_time)
    else:
        # 2. 找到目标时间前后的两个时间点
        times = xr_iri.time.values
        # 确保times已排序
        times_sorted = np.sort(times)
        # 找到比target_time小的最大时间点（前一个），和比target_time大的最小时间点（后一个）
        prev_times = times_sorted[times_sorted <= np.datetime64(target_time)]
        next_times = times_sorted[times_sorted > np.datetime64(target_time)]
        if len(prev_times) == 0 or len(next_times) == 0:
            raise ValueError("目标时间超出IRI数据时间范围，无法插值。")
        t0 = prev_times[-1]
        t1 = next_times[0]
        # 3. 取前后两个切片
        iri0 = xr_iri.sel(time=t0)
        iri1 = xr_iri.sel(time=t1)
        # 4. 计算时间差（小时，浮点数）
        delta_total = (np.datetime64(t1) - np.datetime64(t0)) / np.timedelta64(1, 'h')
        delta_target = (np.datetime64(target_time) - np.datetime64(t0)) / np.timedelta64(1, 'h')
        # 5. 计算需要平移的经度点数
        lon = iri0.lon.values
        dlon = np.abs(lon[1] - lon[0]) if len(lon) > 1 else 0.5  # 默认0.5度分辨率
        shift_points0 = int(round(delta_target * 15 / dlon))
        shift_points1 = int(round((delta_target - delta_total) * 15 / dlon))
        # 6. 对前后切片分别进行经度平移
        data0 = np.roll(iri0.values, shift_points0, axis=-1)
        data1 = np.roll(iri1.values, shift_points1, axis=-1)
        # 7. 线性插值
        weight1 = delta_target / delta_total
        weight0 = 1 - weight1
        interp_data = data0 * weight0 + data1 * weight1
        # 8. 构造新的DataArray
        xr_interp = xr.DataArray(
            interp_data,
            dims=iri0.dims,
            coords=iri0.coords,
            attrs=iri0.attrs
        )
        xr_interp = xr_interp.expand_dims(time=[np.datetime64(target_time)])
        xr_slice = xr_interp

    # 9. 截取经纬度范围
    lat_min, lat_max, lon_min, lon_max = llbox
    xr_slice = xr_slice.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max)
    )
    
    # if target_time in xr_iri.time:
    #     return xr_iri.sel(time=target_time)
    # else:
    #     # 1. 找到最接近的两个时间戳
    #     time_diff = np.abs((xr_iri.time - target_time).values)
    #     idx0, idx1 = np.argmin(time_diff), np.argmax(time_diff)
        
    #     # 2. 计算时间差
    #     t0, t1 = xr_iri.time[idx0].values, xr_iri.time[idx1].values
    #     dt = (t1 - t0) / time_diff[idx0]
        
    #     # 3. 线性插值到目标时间戳
    #     iri_interp = xr_iri.isel(time=[idx0, idx1]).interp(time=target_time, kwargs={"fill_value": "ext
    
    
    # # 1. 创建目标时间点数据集
    # # print(target_time)
    # target_ds = xr.Dataset({'time': [target_time]})
    
    # # 2. 插值到目标时间点 (线性插值)
    # iri_interp = xr_iri.interp(
    #     time=target_ds.time, 
    #     method='linear',
    #     kwargs={"fill_value": "extrapolate"}
    # )
    
    # # 3. 截取指定经纬度范围
    # lat_min, lat_max, lon_min, lon_max = llbox
    # xr_slice = iri_interp.sel(
    #     lat=slice(lat_min, lat_max),
    #     lon=slice(lon_min, lon_max)
    # )
    return xr_slice


# def show_pd_vtec(da, fig_fn, xcoord = "slon"):
#     start_time = da.time[0].values  # 获取第一个时间点
#     end_time = da.time[-1].values
    
#     mid_time = pd.to_datetime(start_time + (end_time - start_time)/2) 
#     hour_dec = mid_time.hour + mid_time.minute/60. + mid_time.second/3600.
#     print(mid_time)
#     import matplotlib.pyplot as plt
#     lons = da.sel(data="lon").values.flatten()
#     lats = da.sel(data="lat").values.flatten()
#     slons =np.array([lon2slon(lon, hour_dec) for lon in lons])
#     mfs = da.sel(data="mf").values.flatten()
#     stec = da.sel(data="stec").values.flatten()
#     # eles = da.sel(data="ele").values.flatten()
#     plt.scatter(slons, lats, s=1,c=stec/mfs)
#     plt.savefig(fig_fn)
#     plt.close()

if __name__ == "__main__":
    iri_filename = "/mnt/geodata/GIM/IRI2020/i202023_2024.obj"
    xr_iri = gnss_utils.loadobject(iri_filename)
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    stec_dir = os.path.join(current_dir, '..', 'data_final', 'stec')
    stec_dir = "/home/nas/xwzheng/ss-dscnn/data_final/stec"
    grid_dir = os.path.join(current_dir, "..", "data_final", "sgrids")
    if not os.path.isdir(grid_dir):
        os.makedirs(grid_dir, exist_ok=True)
    year = 2023
    # doy = 2
    lat_min, lat_max = LAT_MIN, LAT_MAX
    lon_min, lon_max = LON_MIN, LON_MAX
    resolution = RESOLUTION
    lat_lon_box = [lat_min, lat_max, lon_min, lon_max]
    erase_grid = ERASE_GRID
    doy_from = 1
    doy_to = 361
    for doy in range(doy_from, doy_to+1):
        obj_filename = os.path.join(stec_dir, f"{year:04d}_{doy:03d}.obj")
        xr_stec = gnss_utils.loadobject(obj_filename)
        interval = 10 # minutes
        date0 =datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1) + datetime.timedelta(minutes=interval)
        periods = int(24 * 60 / interval)
        ref_epochs = pd.date_range(date0, 
                                  periods=periods, 
                                  freq=f"{interval}min")
        for ref_epoch in ref_epochs:
            # ref_epoch = pd.to_datetime("2023-11-01T17:00:00")
            start_time= ref_epoch - datetime.timedelta(minutes=interval/2)
            end_time  = ref_epoch + datetime.timedelta(minutes=interval/2)
            xr_iri_interp = interpolate_iri_to_10min(ref_epoch, xr_iri, lat_lon_box)
            sites = [str(_) for _ in xr_stec.site.values]
            selected_sites_mask = ~np.isin(sites, valid_sites) #
            xr_stec_sel = xr_stec.sel(time=slice(start_time, end_time),
                                      site=selected_sites_mask)
            pd_vtec = gnss_utils.xr_obs2pd(xr_stec_sel)


            #这里修改了一下，改为了下边两行
            #pd_vtec_slon = gnss_utils.pd_obs2slon(pd_vtec, max_lon=None)


            pd_vtec['slon'] = pd_vtec.apply(lambda row: lon2slon(row['lon'], ref_epoch), axis=1)
            pd_vtec_slon = pd_vtec

            xr_data_slon = pd_vtec2grid(pd_vtec_slon, lat_lon_box, ref_epoch, resolution=resolution, xcoord="slon")
            # xr_grids = stec2grid(xr_stec_sel, lat_lon_box, ref_epoch)
            # masks = create_masks(xr_grids, lat_lon_box, 2, 0.7, 0.3)
            masks = create_masks(xr_data_slon, lat_lon_box, ref_epoch, erase_grid_size=erase_grid, train_ratio=0.7, val_ratio=0.15, xcoord="slon")
            # print(xr_grids.shape, masks["train_mask"].shape, xr_iri_interp.shape)
            out_iri_fn = os.path.join(grid_dir, f"{ref_epoch.strftime('%Y-%m-%dT%H:%M')}_iri.obj")
            out_grid_fn = os.path.join(grid_dir, f"{ref_epoch.strftime('%Y-%m-%dT%H:%M')}_tec_slon.obj")
            out_mask_fn = os.path.join(grid_dir, f"{ref_epoch.strftime('%Y-%m-%dT%H:%M')}_mask.obj")
            gnss_utils.saveobject(xr_data_slon, out_grid_fn)
            gnss_utils.saveobject(masks, out_mask_fn)
            gnss_utils.saveobject(xr_iri_interp, out_iri_fn)
            print(f"Saved {out_grid_fn}")
            print(f"Saved {out_mask_fn}")
            print(f"Saved {out_iri_fn}")

            #原来的代码将其余部分设置为0，我改了一下将其余部分设置为白色
            # def show_data_mask(xr_data_slon, masks, ref_epoch):
            #     fig, axes = plt.subplots(2,2, figsize=(10,10))
            #     axes[0][0].pcolor(xr_data_slon.slon, xr_data_slon.lat, xr_data_slon)
            #     axes[0][1].pcolor(xr_data_slon.slon, xr_data_slon.lat, xr_data_slon*masks["val_mask"])
            #     axes[1][0].pcolor(xr_data_slon.slon, xr_data_slon.lat, xr_data_slon*masks["test_mask"])
            #     out_fig = os.path.join(current_dir, "..", "data_new", "grids_figures", f"{ref_epoch.strftime('%Y-%m-%dT%H:%M')}mid_10min_mask.png")

                
            #     im = axes[1][1].pcolor(xr_data_slon.slon, xr_data_slon.lat, xr_data_slon*masks["train_mask"])
            #     plt.colorbar(im)
            #     plt.savefig(out_fig)
            #     plt.close()
                
            #     print(f"Saved {out_fig}")


            def show_data_mask(xr_data_slon, masks, ref_epoch):
                fig, axes = plt.subplots(2,2, figsize=(10,10))
    
                # 图1：完整数据
                im0 = axes[0][0].pcolor(xr_data_slon.slon, xr_data_slon.lat, xr_data_slon)
                axes[0][0].set_title('Full TEC Data')
                plt.colorbar(im0, ax=axes[0][0])
                
                # 图2：验证集 - 将非验证区域设为NaN（白色）
                val_data = xr_data_slon.where(masks["val_mask"] == 1)
                im1 = axes[0][1].pcolor(xr_data_slon.slon, xr_data_slon.lat, val_data)
                axes[0][1].set_title('Validation Set')
                plt.colorbar(im1, ax=axes[0][1])
                
                # 图3：测试集 - 将非测试区域设为NaN（白色）
                test_data = xr_data_slon.where(masks["test_mask"] == 1)
                im2 = axes[1][0].pcolor(xr_data_slon.slon, xr_data_slon.lat, test_data)
                axes[1][0].set_title('Test Set')
                plt.colorbar(im2, ax=axes[1][0])
                
                # 图4：训练集 - 将非训练区域设为NaN（白色）
                train_data = xr_data_slon.where(masks["train_mask"] == 1)
                im3 = axes[1][1].pcolor(xr_data_slon.slon, xr_data_slon.lat, train_data)
                axes[1][1].set_title('Training Set')
                plt.colorbar(im3, ax=axes[1][1])
                
                out_fig = f"data_final/grids_figures/{ref_epoch.strftime('%Y-%m-%dT%H:%M')}mid_10min_mask.png"
                plt.tight_layout()
                plt.savefig(out_fig)
                plt.close()
                
                print(f"Saved {out_fig}")




            
            show_data_mask(xr_data_slon, masks, ref_epoch)
            # break
        # break
