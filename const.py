valid_sites = ['sxxx', 'fjpt', 'lnsy', 'hrbn', 'xjss',
               'scsn', 'yanc', 'qhmy', 'qhmq', 'jsls', 
               'xiag', 'scbz', 'hnly', 'xjqh', 'xjkc', 
               'gxhc', 'tjbh', 'ahaq', 'ynys', 'xjml', 
               'nxzw', 'ynym', 'hets', 'sclh']
LAT_MIN, LAT_MAX = 15, 55  # 
LON_MIN, LON_MAX = 70, 135 # 
RESOLUTION = 0.2
ERASE_GRID = 2 # degree
SMOOTH_DIS = 2 # 平滑距离
SMOOTH_WINDOW_SIZE = int(SMOOTH_DIS/RESOLUTION) | 1      # 平滑像元数，确保为奇数
SMOOTH_FAC = 0.2      # 平滑因子
SMOOTH_WEIGHT = 1      # 平滑loss相对于MSE loss 的权重