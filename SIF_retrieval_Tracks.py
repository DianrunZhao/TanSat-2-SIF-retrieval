import xarray as xr
import numpy as np
from numpy import *
from scipy.spatial import cKDTree
import h5py
from datetime import datetime, timedelta
import os
import pickle
from scipy.stats import chi2
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import joblib
import time
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 设置日志配置
# logging.basicConfig(filename='process_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# import logging
# from tqdm import tqdm

# # ------- 日志基础配置（保留你原来的 basicConfig） -------
# logging.basicConfig(
#     filename='process_log.log',
#     level=logging.INFO,                       # 记得升到 INFO
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

logger = logging.getLogger(__name__)         # 以后都用 logger.info(...)
logger.setLevel(logging.INFO)

# 1) 文件手动指定编码
fh = logging.FileHandler('process_log.log', encoding='utf-8')
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

# # 2) （可选）保留一个简单的控制台输出
# ch = logging.StreamHandler()
# ch.setFormatter(logging.Formatter('%(message)s'))
# logger.addHandler(ch)

# logger.info('> 现在写入文件的中文都用 UTF‑8 编码啦')

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))   # 用 tqdm 写一行
        except Exception:
            pass
# tqdm 兼容
tq = TqdmLoggingHandler()
tq.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(tq)
# logger.addHandler(TqdmLoggingHandler())

def save(v, filename):
    with open(filename, 'wb') as f:
        pickle.dump(v, f)
    return filename


def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def hfffarred(x):
    return np.exp((-1 * (x - 740) ** 2) / 822)

def hffred(x):
    return np.exp((-1 * (x - 685) ** 2) / 200)


def model_func(x, a, b):
    return a + b * np.sqrt(x)


def convert_time_to_utc(deltatime, base_date_cst):
    base_date_utc = base_date_cst - np.timedelta64(8, 'h')
    time1 = deltatime.astype('timedelta64[ms]')
    target_datetimes = base_date_utc + time1
    return target_datetimes


def batch_interpolate_cosmean(latitudes, longitudes, days_of_year, cosmean_lookup_tables):
    assert len(latitudes) == len(longitudes), "纬度、经度和天数数组长度必须一致"
    cosmean_values = np.zeros(len(latitudes))
    unique_days = np.unique(days_of_year)
    interpolator = cosmean_lookup_tables[unique_days[0]]
    points = np.column_stack((latitudes, longitudes))
    cosmean_values = interpolator(points)
    return cosmean_values

# ---- 把这段独立成一个函数 ----
def init_worker_logging():
    logger = logging.getLogger(__name__)
    if not logger.handlers:              # 子进程第一次进来才加 Handler
        fh = logging.FileHandler('process_log.log', encoding='utf-8')
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

def process_nc_files(file11,outfname, wl0, ve, nve, ni, threshold_sza, threshold_vza, radiance_min, radiance_max, b1, b2,
                     bandin, popt, cosmean_lookup_tables,hff):
    init_worker_logging()
    try:
        piL01,lat1, lon1, sif1, sif740p1, std1, nn1, rad11, rad21, radmean1, err1 = [np.array([], dtype=np.float32) for _ in
                                                                               range(11)]
        sifdc1, sif740pdc1, correction_factor1, sza1, vza1, saa1, vaa1, ka21, ka2mark1, rad_mark1, time1 = [
            np.array([], dtype=np.float32) for _ in range(11)]

        bm=(wl0>=b1)&(wl0<=b2)
        bm773=(wl0>=772)&(wl0<=774)
        bm675=(wl0>=674)&(wl0<=676)
        fin=0

        with h5py.File(outfname, 'r') as ds:
            # gp1 = ds["DataField"]
            rad = array(ds["radiance"][:, :, bm], dtype=np.float32)
            # DN1 = array(gp1["DN"][:, :, 1229])
            # DN2 = array(gp1["DN"][:, :, 9])
            # rad1 = array(ds["radiance"][:, :, bm773])
            # rad2 = array(ds["radiance"][:, :, bm675])
            # gp1 = ds["GeolocationField"]
            lat = array(ds["latitude"][:]).astype(np.float32)
            lon = array(ds["longitude"][:]).astype(np.float32)

            if len(lat[0]) != 1500:
                logging.warning(f"File {outfname} not 1500***********************************")
                return None
            if rad.shape[0] != len(lat) or len(lon) != len(lat):
                logging.warning(f"File {outfname} skipped due to shape mismatch.")
                return None

            sza = array(ds["solar_zenith_angle"][:]).flatten().astype(np.float32)
            vza = array(ds["view_zenith_angle"][:]).flatten().astype(np.float32)
            saa = array(ds["solar_zenith_angle"][:]).flatten().astype(np.float32)
            vaa = array(ds["solar_zenith_angle"][:]).flatten().astype(np.float32)

            # gp1 = ds["RadCaliCoeff"]
            # K = array(gp1["CeofK"][:]).T[:, 788:944]
            # B = array(gp1["CeofB"][:]).T[:, 788:944]
            # rad = DN * K[None, :, :] + B[None, :, :]
            # K1 = array(gp1["CeofK"][:]).T[:, 1198:1230]
            # B1 = array(gp1["CeofB"][:]).T[:, 1198:1230]
            # K2 = array(gp1["CeofK"][:]).T[:, 9:38]
            # B2 = array(gp1["CeofB"][:]).T[:, 9:38]
            # rad1 = DN1 * K1[None, :] + B1[None, :]
            # rad2 = DN2 * K2[None, :] + B2[None, :]
            # rad1 = DN1 * K1[None, :, :] + B1[None, :, :]
            # rad2 = DN2 * K2[None, :, :] + B2[None, :, :]
            column_values = np.zeros(len(lat[0])).astype(np.uint16)#np.arange(len(lat[0])).astype(np.uint16)
            nn = np.tile(column_values, (len(lat), 1))
            mask = (sza < threshold_sza) & (vza < threshold_vza)
            new_shape = (rad.shape[0] * rad.shape[1], rad.shape[2])
            rad = rad.reshape(new_shape)[mask]
            # new_shape = (rad1.shape[0] * rad1.shape[1], rad1.shape[2])
            # rad1 = rad1.reshape(new_shape)[mask]
            # new_shape = (rad2.shape[0] * rad2.shape[1], rad2.shape[2])
            # rad2 = rad2.reshape(new_shape)[mask]
            # rad1 = rad1.flatten()[mask]
            # rad2 = rad2.flatten()[mask]
            # gp1 = ds["FrameAttribute"]
            time0 = array(ds["deltatime"][:])
            time11 = np.tile(time0, (1, len(lat[0]))).flatten()[mask] if len(time0) == len(lat) else \
                np.tile(array(time0[int(len(time0) / 2), :]), (len(lat), len(lat[0]))).flatten()[mask]
            lat, lon, sza, vza, saa, vaa, nn = lat.flatten()[mask], lon.flatten()[mask], sza[mask], vza[mask], saa[mask], vaa[mask], nn.flatten()[mask]

            if lat.size == 0:
                logging.warning(f"File {outfname} skipped due to empty data after masking.")
                return None

            for j in range(1):
                wl1 = wl0#[j]

                wlindex = (wl1 >= b1) & (wl1 <= b2)
                wl = wl1[wlindex]
                Ve = ve[j][:nve, :]

                wlbci = np.argmin(abs(wl - bandin))
                wlbc = wl[wlbci]
                powers = np.arange(1, ni + 1)[:, np.newaxis]
                ff = ((wl/1000.0) ** powers) * array(Ve[0])
                hf = hff(wl) / hff(wlbc)
                V1 = np.vstack((Ve, ff, hf))
                v = len(wl) - len(V1)
                confidence_level = 0.95
                chi2_low = chi2.ppf((1 - confidence_level) / 2, v) / v
                chi2_high = chi2.ppf(1 - (1 - confidence_level) / 2, v) / v
                sell = (nn == j)
                piL = array(rad[sell])
                # piL1 = array(rad1[sell])
                # piL2 = array(rad2[sell])
                # radmeana1 = np.mean(piL1, axis=1).astype(np.float32)
                # radmeana2 = np.mean(piL2, axis=1).astype(np.float32)
                lat1_sampled, lon1_sampled = lat[sell], lon[sell]
                nn0, sza_sampled, vza_sampled, saa_sampled, vaa_sampled, time_sampled =  nn[sell], sza[sell], vza[sell], saa[sell], vaa[sell], time11[sell]
                radmean = np.mean(piL, axis=1).astype(np.float32)
                radmin = np.min(piL, axis=1).astype(np.float32)
                rad_mask = (radmean < radiance_max) & (radmean > radiance_min) & (radmin > 0.5)
                rad_mark = np.zeros_like(radmean).astype(np.uint16)
                rad_mark[rad_mask] = 1
                if len(piL) == 0:
                    continue

                W, residuals, rank, s = np.linalg.lstsq(V1.T, piL.T, rcond=None)
                W = W.T
                epsilon = 1e-10
                snr_fit = model_func((np.maximum(piL,epsilon)), popt[0], popt[1])
                noise1 = (np.maximum(piL,epsilon)) / snr_fit# + epsilon
                ka2 = (np.sum(array(W @ V1 - piL) ** 2 / noise1 ** 2, axis=1) / v)
                ka2mark = np.zeros_like(ka2).astype(np.uint16)
                valid_chi2_mask = (ka2 < chi2_high) & (ka2 > chi2_low)
                ka2mark[valid_chi2_mask] = 1
                jac1 = V1.T
                errs = np.zeros(noise1.shape[0]).astype(float32)
                try:
                    for i in range(noise1.shape[0]):
                        # if i==298788:
                        #     cc=0
                        # if ~np.isnan(noise1[i][0]):
                        #     aaaaa=1
                        d2_inv = np.diag(1 /  np.maximum(noise1[i]**2, epsilon))#(noise1[i] ** 2))
                        jac1_T_d2_inv = jac1.T @ d2_inv
                        jac1_T_d2_inv_jac1 = jac1_T_d2_inv @ jac1
                        A = jac1_T_d2_inv_jac1[:-1, :-1]
                        B = jac1_T_d2_inv_jac1[:-1, -1]
                        C = jac1_T_d2_inv_jac1[-1, :-1]
                        D = jac1_T_d2_inv_jac1[-1, -1]
                        try:
                            # 首选 lstsq （对奇异也能给近似解，但有时 SVD 不收敛）
                            A_inv_B = np.linalg.lstsq(A, B, rcond=None)[0]

                        except np.linalg.LinAlgError:
                            try:
                                # 回退：显式伪逆
                                A_inv_B = np.linalg.pinv(A, rcond=1e-10) @ B
                            except np.linalg.LinAlgError:
                                # logging.warning(f"SVD 仍未收敛，像元 {i} 跳过")
                                errs[i] = np.nan
                                continue
                        # A_inv_B = np.linalg.lstsq(A, B, rcond=None)[0] #np.linalg.solve(A, B)
                        denom = D - C @ A_inv_B + epsilon
                        bottom_right_element = 1.0 / np.maximum(denom, epsilon)   #denom
                        errs[i] = bottom_right_element
                except Exception as e:
                    logging.error(f"Error processing file {outfname}: {e}")
                    return None
                res = (W @ V1 - piL)
                std = np.std(res, axis=1)
                sif = W[:, -1].astype(np.float32)
                sif740p = W[:, -1] / hff(wlbc).astype(np.float32)
                deltatime = array(time_sampled)
                base_date_cst = np.datetime64('2022-06-21T00:00:00')
                target_datetimes = convert_time_to_utc(deltatime[0], base_date_cst)
                dates = target_datetimes.astype('datetime64[D]')
                year_start = target_datetimes.astype('datetime64[Y]')
                day_of_year = (dates - year_start).astype(int)# + 1
                days_of_year = day_of_year % 365
                cosmean = batch_interpolate_cosmean(lat1_sampled, lon1_sampled, days_of_year, cosmean_lookup_tables)
                correction_factor = cosmean / sza_sampled#np.cos(np.radians(sza_sampled))
                sifdc, sif740pdc = sif * correction_factor, sif740p * correction_factor

                if len(piL01)==0:
                    piL01=piL
                else:
                    piL01 = np.vstack((piL01, piL))
                    # piL01 = np.concatenate((piL01, piL))

                lat1 = np.concatenate((lat1, lat1_sampled))
                lon1 = np.concatenate((lon1, lon1_sampled))
                sif1 = np.concatenate((sif1, sif))
                sif740p1 = np.concatenate((sif740p1, sif740p))
                radmean1 = np.concatenate((radmean1, radmean))
                std1 = np.concatenate((std1, array(std.T)[0]))
                nn1 = np.concatenate((nn1, nn0))
                # rad11 = np.concatenate((rad11, rad10))
                # rad21 = np.concatenate((rad21, rad20))
                err1 = np.concatenate((err1, errs))
                sifdc1 = np.concatenate((sifdc1, sifdc))
                sif740pdc1 = np.concatenate((sif740pdc1, sif740pdc))
                correction_factor1 = np.concatenate((correction_factor1, correction_factor))
                sza1 = np.concatenate((sza1, sza_sampled))
                vza1 = np.concatenate((vza1, vza_sampled))
                saa1 = np.concatenate((saa1, saa_sampled))
                vaa1 = np.concatenate((vaa1, saa_sampled))
                ka21 = np.concatenate((ka21, ka2))
                ka2mark1 = np.concatenate((ka2mark1, ka2mark))
                rad_mark1 = np.concatenate((rad_mark1, rad_mark))
                time1 = np.concatenate((time1, time_sampled))
                fin=1

        return fin,file11,rad11, rad21,piL01, lat1, lon1, sif1, sif740p1, radmean1, std1, nn1, err1, sifdc1, sif740pdc1, correction_factor1, sza1, vza1, saa1, vaa1, ka21, ka2mark1, rad_mark1, time1

    except Exception as e:
        logging.error(f"Error processing file {outfname}: {e}")
        return None


# def find_folders_with_keywords(root_path, keywords):
#     matching_folders = []
#     for dirpath, dirnames, _ in os.walk(root_path):
#         for dirname in dirnames:
#             if all([keyword.lower() in dirname.lower() for keyword in keywords]):
#                 matching_folders.append(dirname)
#     return matching_folders
def find_folders_with_keywords(root_path, keywords):
    matching_folders = []
    for dirpath, dirnames, _ in os.walk(root_path):
        for dirname in dirnames:
            if keywords.lower() in dirname.lower() :
                matching_folders.append(dirname)
    return matching_folders

def process_single_file(root_path2, file11, wl0, ve, nve, ni, threshold_sza, threshold_vza, radiance_min, radiance_max, b1, b2, bandin, popt, cosmean_lookup_tables,hff):
    try:
        # fname = file11 + ".h5"
        nc_file = os.path.join(root_path2, file11)
        if not os.path.exists(nc_file):
            print(f"文件不存在: {nc_file}")
            return None
        return process_nc_files(file11,nc_file, wl0, ve, nve, ni, threshold_sza, threshold_vza, radiance_min, radiance_max, b1, b2, bandin, popt, cosmean_lookup_tables,hff)

    except Exception as e:
        logging.error(f"Error in process_single_file for {file11}: {e}")
        return None


def main(yearr,i_start, i_end,input_path,aux_and_output_path,o2="O2A"):
    if o2=="O2A":
        min_wl = 747
        max_wl = 758
        nve = 6
        ni = 2
        band="FARRED"
        radiance_min = 25
        radiance_max = 200
        popti=[0,1/0.008172]
        hff=hfffarred
    else:
        min_wl=672
        max_wl=686
        nve = 4
        ni = 5
        band="RED"
        radiance_min = 0
        radiance_max = 80
        popti=[0,1/0.008727]
        hff=hffred


    popt_file=rF"{aux_and_output_path}\aux_data\snr_fit_{o2}.txt"
    if os.path.isfile(popt_file):
        popt = load(rF"{aux_and_output_path}\aux_data\snr_fit_{o2}.txt")
    else:
        popt=popti
    nc_file = rf"{aux_and_output_path}\aux_data\TanSat-2_{o2}_nadir_0621_track_01_part_01.nc"
    with h5py.File(nc_file, 'r') as ds:
        # gp1 = ds["nominal_wavelength"]
        wl0 = array(ds["nominal_wavelength"][:])

    root_path = os.path.join(input_path,o2)
    keywords = ['_']
    threshold_sza = 70
    threshold_vza = 60

    # min_wl = 747
    # max_wl = 758

    start_time = time.time()
    cosmean_lookup_tables = joblib.load(rF"{aux_and_output_path}\aux_data\cosszamean_lookup_tables_by_day_{yearr}.pkl")

    for i in range(i_start, i_end + 1):
        # plt.figure()
        # plt.plot(np.array(v[0])[0])
        # plt.show()
        root_path1 = root_path
        nn = 30
        if i in [1,3,5,7,8,10,12]:
            nn = 31
        if i ==2:
            nn=28
            if yearr%4==0:
                nn=29

        for j in range(nn):
            # piL11,lat1, lon1, sif1, sif740p1 = np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)
            # rad11, rad21, std1, nn1, radmean1, err1 = np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.uint16), np.array([], dtype=np.float32), np.array([], dtype=np.float32)
            # sifdc1, sif740pdc1, correction_factor1 = np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)
            # raa,sza1, vza1, saa1, vaa1, ka21, ka2mark1, rad_mark1, time1 = np.array([], dtype=np.float32),np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.uint16), np.array([], dtype=np.uint16), np.array([], dtype=np.uint16), np.array([], dtype=np.float32)
            lat_lst, lon_lst = [], []
            sif_lst, sif740p_lst = [], []
            rad1_lst, rad2_lst = [], []
            radmean_lst, std_lst = [], []
            nn_lst, err_lst = [], []
            sifdc_lst, sif740pdc_lst = [], []
            corr_lst = []
            sza_lst, vza_lst, saa_lst, vaa_lst, raa_lst = [], [], [], [], []
            ka2_lst, ka2mark_lst, radmark_lst = [], [], []
            time_lst = []

            root_path2 = root_path1 + rf"\\{yearr}{i:02d}{j+1:02d}"
            strr=f"{yearr}{i:02d}{j+1:02d}"
            matching_folders1 = find_folders_with_keywords(root_path2, strr)
            if len(matching_folders1)==0:
                continue

            v = load(rf'{aux_and_output_path}\training_data\{o2}\{yearr}\V\V-{band}{yearr}{i:02d}{j+1:02d}')

            # if len(matching_folders1) !=0:
            logger.info(f'> 开始处理日期 {yearr}{i:02d}{j+1:02d}')
            for ffile in matching_folders1:
                root_path3=os.path.join(root_path2,ffile)
                logger.info(f'  └─轨道 {ffile}准备中…')
                file_list = glob.glob(os.path.join(root_path3,rf"*{o2}*.nc"))
                # with ProcessPoolExecutor(max_workers=8) as executor:
                #     futures = [executor.submit(process_single_file, root_path3, file11, wl0, v, nve, ni, threshold_sza, threshold_vza, radiance_min, radiance_max, min_wl, max_wl, 757, popt, cosmean_lookup_tables) for file11 in file_list[:]]
                #     for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing file: {yearr}{i:02d}{j + 1:02d}-{ffile}"):
                with ProcessPoolExecutor(max_workers=8) as executor:
                    # futures_dict : {Future 对象 : 原始文件索引 idx}
                    futures_dict = {
                        executor.submit(
                            process_single_file,
                            root_path3,  # 路径
                            f,  # 文件名
                            wl0, v, nve, ni,
                            threshold_sza, threshold_vza,
                            radiance_min, radiance_max,
                            min_wl, max_wl, 757,
                            popt, cosmean_lookup_tables,hff
                        ): idx
                        for idx, f in enumerate(file_list)  # ← 枚举保证索引与 file_list 顺序一致
                    }

                    # 预先开一个结果列表，长度 = 文件数，稍后填充
                    ordered_results = [None] * len(file_list)

                    # ---------------------------------------------------------
                    # ➋ 仍用 as_completed ‑ 谁先算完先返回，提高吞吐
                    #    但把结果放回 ordered_results[idx] 位置，保持顺序
                    # ---------------------------------------------------------
                    for fut in tqdm(as_completed(futures_dict),
                                    total=len(futures_dict),
                                    desc=f"Processing file: {yearr}{i:02d}{j + 1:02d}-{ffile}"):

                        idx = futures_dict[fut]  # 取回原始顺序索引
                        file0111 = file_list[idx]  # ← 当次真正的文件名

                        try:
                            ordered_results[idx] = fut.result()  # 放到正确槽位
                            fin11=ordered_results[idx][0]
                            # file0111 = ordered_results[idx][1]
                            print(fin11, file0111)
                            logger.info(f'✓ 处理完成 {file0111}')
                        except Exception as e:
                            logger.error(f'✗ 处理 {file0111} 失败: {e}')
                            logging.error(f"文件处理时出错: {e}")
                            ordered_results[idx] = None  # 出错就占位 None
                try:
                    for result in ordered_results:
                        if result is None:
                            continue
                        (fin, file01, rad1, rad2, piL, lat, lon, sif, sif740p,
                         radmean, std, nn, err, sifdc, sif740pdc, correction_factor,
                         sza, vza, saa, vaa, ka2, ka2mark, rad_mark, time11) = result
                        print(fin,file01)
                        print(len(sif740p))
                        if len(lat) == 0:
                            continue

                        # ======== 后面拼接代码与你原来保持一致 ========
                        # if len(piL11)==0:
                        #     piL11=piL
                        # else:
                        #     piL11 = np.vstack((piL11, piL))

                        lat_lst.append(lat)
                        lon_lst.append(lon)

                        # sif_lst.append(sif)
                        sif740p_lst.append(sif740p)

                        # rad1_lst.append(rad1)
                        # rad2_lst.append(rad2)

                        radmean_lst.append(radmean)
                        std_lst.append(std)
                        nn_lst.append(nn)
                        err_lst.append(err)

                        # sifdc_lst.append(sifdc)
                        sif740pdc_lst.append(sif740pdc)
                        corr_lst.append(correction_factor)

                        sza_lst.append(sza)
                        vza_lst.append(vza)
                        # saa_lst.append(saa)
                        # vaa_lst.append(vaa)
                        raa_lst.append(saa - vaa)

                        ka2_lst.append(ka2)
                        ka2mark_lst.append(ka2mark)
                        radmark_lst.append(rad_mark)

                        time_lst.append(time11)

                        # lat1 = np.concatenate((lat1, lat))
                        # rad11 = np.concatenate((rad11, rad1))
                        # rad21 = np.concatenate((rad21, rad2))
                        # lon1 = np.concatenate((lon1, lon))
                        # sif1 = np.concatenate((sif1, sif))
                        # sif740p1 = np.concatenate((sif740p1, sif740p))
                        # print(len(sif740p1))
                        # radmean1 = np.concatenate((radmean1, radmean))
                        # std1 = np.concatenate((std1, std))
                        # nn1 = np.concatenate((nn1, nn))
                        # err1 = np.concatenate((err1, err))
                        # sifdc1 = np.concatenate((sifdc1, sifdc))
                        # sif740pdc1 = np.concatenate((sif740pdc1, sif740pdc))
                        # correction_factor1 = np.concatenate((correction_factor1, correction_factor))
                        # sza1 = np.concatenate((sza1, sza))
                        # vza1 = np.concatenate((vza1, vza))
                        # saa1 = np.concatenate((saa1, saa))
                        # vaa1 = np.concatenate((vaa1, vaa))
                        # ka21 = np.concatenate((ka21, ka2))
                        # ka2mark1 = np.concatenate((ka2mark1, ka2mark))
                        # rad_mark1 = np.concatenate((rad_mark1, rad_mark))
                        # time1 = np.concatenate((time1, time11))

                        # raa0 = saa - vaa
                except Exception as e:
                    logging.error(f"文件处理时出错: {e}")
                    # try:
                    #     result = future.result()
                    #     if result is not None:
                    #         (fin,file01,rad1, rad2,piL, lat, lon, sif, sif740p, radmean, std, nn, err, sifdc, sif740pdc, correction_factor, sza, vza, saa, vaa, ka2, ka2mark, rad_mark, time11) = result
                    #         print(fin)
                    #         print(len(sif740p))
                    #         if len(lat)==0:
                    #             continue
                    #
                    #         # if len(piL11)==0:
                    #         #     piL11=piL
                    #         # else:
                    #         #     piL11 = np.vstack((piL11, piL))
                    #         lat1 = np.concatenate((lat1, lat))
                    #
                    #         rad11 = np.concatenate((rad11, rad1))
                    #         rad21 = np.concatenate((rad21, rad2))
                    #         lon1 = np.concatenate((lon1, lon))
                    #         sif1 = np.concatenate((sif1, sif))
                    #         sif740p1 = np.concatenate((sif740p1, sif740p))
                    #         print(len(sif740p1))
                    #         radmean1 = np.concatenate((radmean1, radmean))
                    #         std1 = np.concatenate((std1, std))
                    #         nn1 = np.concatenate((nn1, nn))
                    #         err1 = np.concatenate((err1, err))
                    #         sifdc1 = np.concatenate((sifdc1, sifdc))
                    #         sif740pdc1 = np.concatenate((sif740pdc1, sif740pdc))
                    #         correction_factor1 = np.concatenate((correction_factor1, correction_factor))
                    #         sza1 = np.concatenate((sza1, sza))
                    #         vza1 = np.concatenate((vza1, vza))
                    #         saa1 = np.concatenate((saa1, saa))
                    #         vaa1 = np.concatenate((vaa1, vaa))
                    #         ka21 = np.concatenate((ka21, ka2))
                    #         ka2mark1 = np.concatenate((ka2mark1, ka2mark))
                    #         rad_mark1 = np.concatenate((rad_mark1, rad_mark))
                    #         time1 = np.concatenate((time1, time11))
                    #         raa0 = saa - vaa
                    #
                    # #         data_dict1 = {
                    # #             'rad': piL,
                    # #             'lat': lat,
                    # #             'lon': lon,
                    # #             'sif757': sif,
                    # #             'res_std': std,
                    # #             'radmean': radmean,
                    # #             'SoundingID': nn,
                    # #             'sif': sif740p,
                    # #             'sif_err': err,
                    # #             'sif757dc': sifdc,
                    # #             'sifdc': sif740pdc,
                    # #             'correction_factor': correction_factor,
                    # #             'sza': sza,
                    # #             'vza': vza,
                    # #             'raa': raa0,
                    # #             'ka2': ka2,
                    # #             'ka2mark': ka2mark,
                    # #             'radmark': rad_mark,
                    # #             'time': time11,
                    # #             'rad785': rad1,
                    # #             'rad665': rad2
                    # #         }
                    # #         outfname1 = rf'Z:\xuj\LT1\FSI\Goumang\data\DATA\output\SIF_output_Tracks\{yearr}\SIF-{file01}' + '.nc'
                    # #         try:
                    # #             with h5py.File(outfname1, 'w') as file:
                    # #                 for var_name, data in data_dict1.items():
                    # #                     dataset = file.create_dataset(var_name, data=data, compression='gzip',
                    # #                                                   compression_opts=5)
                    # #                 print(f"TRACKS-HDF5 file {outfname1} created successfully")
                    # #         except Exception as e:
                    # #             logging.error(f"Error saving HDF5 file {outfname1}: {e}")
                    # #
                    # except Exception as e:
                    #     logging.error(f"文件处理时出错: {e}")

            lat1 = np.concatenate(lat_lst, axis=0)
            lon1 = np.concatenate(lon_lst, axis=0)

            # sif1 = np.concatenate(sif_lst, axis=0)
            sif740p1 = np.concatenate(sif740p_lst, axis=0)

            # rad11 = np.concatenate(rad1_lst, axis=0)
            # rad21 = np.concatenate(rad2_lst, axis=0)

            radmean1 = np.concatenate(radmean_lst, axis=0)
            std1 = np.concatenate(std_lst, axis=0)
            nn1 = np.concatenate(nn_lst, axis=0)
            err1 = np.concatenate(err_lst, axis=0)

            # sifdc1 = np.concatenate(sifdc_lst, axis=0)
            sif740pdc1 = np.concatenate(sif740pdc_lst, axis=0)
            correction_factor1 = np.concatenate(corr_lst, axis=0)

            sza1 = np.concatenate(sza_lst, axis=0)
            vza1 = np.concatenate(vza_lst, axis=0)
            # saa1 = np.concatenate(saa_lst, axis=0)
            # vaa1 = np.concatenate(vaa_lst, axis=0)
            raa = np.concatenate(raa_lst, axis=0)

            ka21 = np.concatenate(ka2_lst, axis=0)
            ka2mark1 = np.concatenate(ka2mark_lst, axis=0)
            rad_mark1 = np.concatenate(radmark_lst, axis=0)

            time1 = np.concatenate(time_lst, axis=0)

            # raa = saa1 - vaa1
            lat1 = lat1.reshape(-1, 1500)
            lon1 = lon1.reshape(-1, 1500)
            std1 = std1.reshape(-1, 1500)
            radmean1 = radmean1.reshape(-1, 1500)
            nn1 = nn1.reshape(-1, 1500)
            sif740p1 = sif740p1.reshape(-1, 1500)
            err1 = err1.reshape(-1, 1500)
            sif740pdc1 = sif740pdc1.reshape(-1, 1500)
            correction_factor1 = correction_factor1.reshape(-1, 1500)
            sza1 = sza1.reshape(-1, 1500)
            vza1 = vza1.reshape(-1, 1500)
            raa = raa.reshape(-1, 1500)
            ka21 = ka21.reshape(-1, 1500)
            ka2mark1 = ka2mark1.reshape(-1, 1500)
            rad_mark1 = rad_mark1.reshape(-1, 1500)
            time1 = time1.reshape(-1, 1500)

            radians = np.arccos(sza1)
            sza1 = np.degrees(radians)
            radians = np.arccos(vza1)
            vza1 = np.degrees(radians)
            radians = np.arccos(raa)
            raa = np.degrees(radians)

            data_dict = {
                # 'rad': piL11,
                'lat': lat1,
                'lon': lon1,
                # 'sif757': sif1,
                'res_std': std1,
                'radmean': radmean1,
                'SoundingID': nn1,
                'sif': sif740p1,
                'sif_err': err1,
                # 'sif757dc': sifdc1,
                'sifdc': sif740pdc1,
                'correction_factor': correction_factor1,
                'sza': sza1,
                'vza': vza1,
                'raa': raa,
                'ka2': ka21,
                'ka2mark': ka2mark1,
                'radmark': rad_mark1,
                'time': time1,
                # 'rad773': rad11,
                # 'rad675': rad21
            }

            outfname = rf'{aux_and_output_path}\output\SIF_output\{o2}\{yearr}\TanSat-2_SIF_{band}_' + f"{yearr}_{i:02d}" + f"{j + 1:02d}" + '.nc'
            os.makedirs(rf'{aux_and_output_path}\output\SIF_output\{o2}\{yearr}', exist_ok=True)
            try:
                with h5py.File(outfname, 'w') as file:
                    for var_name, data in data_dict.items():
                        dataset = file.create_dataset(var_name, data=data, compression='gzip', compression_opts=5)
                    print(f"HDF5 file {outfname} created successfully")
                    logger.info(f'★ 轨道文件已保存到 {outfname}')

            except Exception as e:
                logging.error(f"Error saving HDF5 file {outfname}: {e}")

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    try:

        yearr=2022

        i_start = 6
        i_end = 12
        input_path = RF"F:\Tansat2_Tracks_Simulation\NEW\TanSat-2_O2_simulations"
        aux_and_output_path = RF"F:\Tansat2_Tracks_Simulation\NEW\TanSat-2-SIF-retrieval\data"
        o2 = "O2B"
        main(yearr,i_start, i_end,input_path,aux_and_output_path,o2)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
