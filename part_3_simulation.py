"""
PHẦN 3 - DỮ LIỆU MÔ PHỎNG VÀ ĐÁNH GIÁ ĐỘ CHÍNH XÁC

File này bao gồm:
  1. ve_qua()               : vẽ 1 quả cà chua (đổ bóng, vệt sáng, cuống, màu loang)
  2. tao_anh_tinh()         : ảnh tĩnh nền giấy trắng + vật chuẩn 20 mm + nhiều quả + ĐÁP ÁN
  3. tao_bang_tai(), khung_hinh_bang_tai(): video băng tải, quả chạy từ TRÁI sang PHẢI
  4. danh_gia_anh_tinh(), danh_gia_bang_tai(): so kết quả với đáp án
  5. Chạy riêng file này -> đánh giá hàng loạt 60 ảnh + 10 video, in bảng, lưu CSV:
         python part_3_simulation.py
"""
import csv
import math
import os
import time

import cv2
import numpy as np

from part_2_measuring_and_counting import CANH_VAT_CHUAN_MM, TEN_CO, phan_loai_co
# Màu vỏ cho từng độ chín, viết theo HSV của OpenCV (H 0-179). Đỏ ở quanh H = 0.


# Màu vỏ theo độ chín (HSV của OpenCV, H 0-179)
# Chín: đỏ | Ương: cam | Xanh: xanh lá
MAU_VO = {
    2: {"H": (-3, 4), "S": 215, "V": 225},     # Chín : đỏ, vài mảng cam
    1: {"H": (13, 18), "S": 228, "V": 240},    # Ương : cam, khoảng 1/3 loang đỏ
    0: {"H": (38, 52), "S": 185, "V": 200},    # Xanh : xanh lá, loang vàng
}
HUONG_ANH_SANG = np.array([-0.45, -0.55, 0.70]) / np.linalg.norm([-0.45, -0.55, 0.70])  # đèn chiếu từ trên-trái


# 1. Vẽ 1 quả cà chua
# Hình hơi bầu dục | màu loang | đổ bóng hình cầu | vệt phản sáng trắng | cuống 5 cánh
# Cuống và vệt sáng: để thử khả năng loại bỏ của Phần 1
def ve_qua(do_chin, r, rng):
    """Vẽ 1 quả bán kính r pixel. Trả về dict:
         anh  : ảnh màu (số thực) của quả
         alpha: độ phủ 0..1 (1 = trong quả, 0 = ngoài, mép = số lẻ để mép mịn)
         bong : bóng đổ (alpha làm mờ)
         r    : bán kính"""
    n = 2 * int(1.45 * r) + 4                      # ảnh vuông, rộng hơn quả để có chỗ cho bóng
    tam = (n - 1) / 2.0
    Y, X = np.mgrid[0:n, 0:n].astype(np.float32)
    dx = (X - tam) / r                             # tọa độ chuẩn hóa: mép quả ~ 1
    dy = (Y - tam) / r

    # Quả không tròn tuyệt đối: bán kính thay đổi nhẹ theo góc (hơi bầu dục, có 5 múi)
    goc = np.arctan2(dy, dx)
    ban_kinh = 1 + 0.012 * np.cos(5 * goc + rng.uniform(0, 6.3)) + 0.03 * np.cos(2 * goc + rng.uniform(0, 6.3))
    nx = dx / ban_kinh
    ny = dy / ban_kinh
    rn = np.sqrt(nx ** 2 + ny ** 2)                # < 1 là bên trong quả
    alpha = np.clip((1 - rn) * r + 0.5, 0, 1)

    # --- Màu sắc (kênh H) có vết loang: tạo ảnh nhiễu 5x5 rồi phóng to cho mượt
    mau = MAU_VO[do_chin]
    loang = cv2.resize(rng.random((5, 5)).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
    loang = (loang - loang.min()) / (loang.max() - loang.min() + 1e-6)
    h0 = rng.uniform(*mau["H"])
    if do_chin == 2:
        t = np.clip((loang - 0.80) / 0.08, 0, 1)   # 20% diện tích ngả sang cam (H = 12)
        H = h0 + t * (12 - h0)
    elif do_chin == 1:
        t = np.clip((loang - 0.60) / 0.08, 0, 1)   # khoảng 1/3 diện tích ngả sang đỏ (H = 3)
        H = h0 + t * (3 - h0)
    else:
        t = np.clip((loang - 0.72) / 0.08, 0, 1)   # một ít vết vàng (H = 30)
        H = h0 + t * (30 - h0)
    S = mau["S"] * (0.9 + 0.1 * rn)

    # --- Độ sáng V: đổ bóng hình cầu (chỗ quay về phía đèn sáng hơn)
    nz = np.sqrt(np.clip(1 - rn ** 2, 0, 1))                      # pháp tuyến mặt cầu
    chieu_sang = np.clip(nx * HUONG_ANH_SANG[0] + ny * HUONG_ANH_SANG[1] + nz * HUONG_ANH_SANG[2], 0, 1)
    V = mau["V"] * (0.5 + 0.5 * chieu_sang) + rng.normal(0, 3, H.shape)

    # --- Vệt phản sáng trắng (S giảm, V tăng)
    vet_sang = np.exp(-((nx + 0.38) ** 2 + (ny + 0.42) ** 2) / (2 * 0.12 ** 2))
    S = S * (1 - 0.85 * vet_sang)
    V = V + (255 - V) * vet_sang

    # --- Cuống: ngôi sao 5 cánh màu xanh đậm ở giữa (nhìn từ trên xuống)
    cuong = np.zeros((n, n), np.uint8)
    xoay = rng.uniform(0, 6.3)
    dinh = []
    for i in range(10):
        do_dai = (0.28 if i % 2 == 0 else 0.09) * r                # đỉnh nhọn / đỉnh lõm xen kẽ
        a = xoay + i * math.pi / 5
        dinh.append((tam + 0.05 * r + do_dai * math.cos(a), tam + 0.05 * r + do_dai * math.sin(a)))
    cv2.fillPoly(cuong, [np.array(dinh, np.int32)], 255, cv2.LINE_AA)
    w = cuong / 255.0
    H = H * (1 - w) + 45 * w
    S = S * (1 - w) + 170 * w
    V = V * (1 - w) + 0.55 * V * w

    # --- Ghép H, S, V rồi đổi sang BGR
    hsv = np.dstack([np.mod(np.round(H), 180), np.clip(S, 0, 255), np.clip(V, 0, 255)]).astype(np.uint8)
    anh = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    bong = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 0.18 * r)   # bóng = hình quả làm mờ
    return {"anh": anh, "alpha": alpha.astype(np.float32), "bong": bong, "r": r}


# Dán quả lên nền
# Kết quả = nền x (1 - alpha) + quả x alpha | vẽ bóng trước, quả sau
def dan_qua(nen, qua, cx, cy):
    """Dán quả lên ảnh nền (số thực) tại tâm (cx, cy): vẽ bóng trước (lệch xuống-phải), rồi tới quả.
    Công thức trộn: kết_quả = nền * (1 - alpha) + quả * alpha"""
    n, r = qua["anh"].shape[0], qua["r"]
    cao, rong = nen.shape[:2]
    for lop in ("bong", "qua"):
        lech_x, lech_y = (0.10 * r, 0.14 * r) if lop == "bong" else (0, 0)
        x0 = int(round(cx - n / 2 + lech_x))
        y0 = int(round(cy - n / 2 + lech_y))
        # Phần nằm trong ảnh (quả sát mép thì chỉ dán phần còn thấy)
        ax0, ay0, ax1, ay1 = max(x0, 0), max(y0, 0), min(x0 + n, rong), min(y0 + n, cao)
        if ax0 >= ax1 or ay0 >= ay1:
            continue
        vung_nen = nen[ay0:ay1, ax0:ax1]
        if lop == "bong":
            a = qua["bong"][ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0, None] * 0.30   # bóng tối 30%
            vung_nen *= (1 - a)
        else:
            a = qua["alpha"][ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0, None]
            vung_nen[:] = vung_nen * (1 - a) + qua["anh"][ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] * a


# Đáp án đường kính
# D = 2 x căn(diện tích / pi) / k  (cùng công thức với bộ đo)
def duong_kinh_that(qua, k):
    """Đáp án đường kính (mm) = đường kính hình tròn có cùng diện tích với quả đã vẽ."""
    return 2 * math.sqrt(qua["alpha"].sum() / math.pi) / k


MAU_NHIEU = {}     # nhiễu tạo sẵn cho video (tạo nhiễu mới mỗi khung thì chậm)


# Thử độ sáng và nhiễu
# Độ sáng: nhân hệ số (0,6 = tối 40%) | nhiễu: Gaussian độ lệch chuẩn sigma
def them_do_sang_va_nhieu(anh, do_sang, nhieu, dung_mau_co_san=False):
    """Đổi độ sáng (nhân hệ số, ví dụ 0.6 = tối 40%) và cộng nhiễu Gaussian độ lệch chuẩn = nhieu."""
    anh = anh * do_sang
    if nhieu > 0:
        if dung_mau_co_san:
            if anh.shape not in MAU_NHIEU:
                MAU_NHIEU[anh.shape] = [np.random.standard_normal(anh.shape).astype(np.float32) for _ in range(8)]
            anh = anh + MAU_NHIEU[anh.shape][np.random.randint(8)] * nhieu
        else:
            ma_tran_nhieu = np.empty(anh.shape, np.float32)
            cv2.randn(ma_tran_nhieu, 0, float(nhieu))
            anh = anh + ma_tran_nhieu
    return np.clip(anh, 0, 255).astype(np.uint8)


# 2. Ảnh tĩnh + đáp án
# k ngẫu nhiên 2-3 px/mm | đường kính quả 35-75 mm | nền giấy + vật chuẩn 20 mm
# Seed cố định -> vẽ lại đúng ảnh cũ để so sánh | cho_cham: đặt quả sát nhau
def tao_anh_tinh(so_qua=9, seed=None, do_sang=1.0, nhieu=0.0, cho_cham=False, rong=960, cao=640):
    """Trả về (ảnh, đáp án, k).  Đáp án = danh sách dict {do_chin, co, d_mm, tam} của từng quả.
    seed: cùng 1 số seed -> vẽ ra đúng cùng 1 ảnh (để so sánh khi đổi độ sáng, nhiễu)."""
    rng = np.random.default_rng(seed)

    # Mỗi ảnh có k khác nhau (2 - 3 px/mm), giống như camera đặt ở độ cao khác nhau
    canh = int(round(CANH_VAT_CHUAN_MM * rng.uniform(2.0, 3.0)))
    k = canh / CANH_VAT_CHUAN_MM

    # Nền giấy trắng hơi tối ở 4 góc + vân giấy
    Y, X = np.mgrid[0:cao, 0:rong].astype(np.float32)
    toi_goc = 1 - 0.24 * (((X - rong / 2) / rong) ** 2 + ((Y - cao / 2) / cao) ** 2)
    van_giay = cv2.GaussianBlur(rng.normal(0, 4, (cao, rong)).astype(np.float32), (0, 0), 1.2)
    nen = np.dstack([232 * toi_goc + van_giay, 236 * toi_goc + van_giay, 239 * toi_goc + van_giay])

    # Vật chuẩn: ô vuông đen ở góc dưới bên trái + chữ "20 mm" màu xám
    vx, vy = 24, cao - 24 - canh
    nen[vy:vy + canh, vx:vx + canh] = 22
    chu = np.zeros((cao, rong), np.uint8)
    cv2.putText(chu, "20 mm", (vx + canh + 8, vy + canh - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
    nen[chu > 0] = 150

    # Đặt từng quả ở vị trí ngẫu nhiên, không chồng lên nhau (hoặc chạm nhau nếu cho_cham)
    da_dat = []          # (cx, cy, r) các quả đã đặt
    dap_an = []
    so_lan_thu = 0
    while len(da_dat) < so_qua and so_lan_thu < 400 * so_qua:
        so_lan_thu += 1
        r = int(rng.uniform(35, 75) * k / 2)                    # đường kính 35 - 75 mm
        if cho_cham and da_dat and rng.random() < 0.6:          # đặt sát cạnh 1 quả có sẵn
            ox, oy, orr = da_dat[int(rng.integers(len(da_dat)))]
            a = rng.uniform(0, 2 * math.pi)
            cx, cy = ox + (orr + r) * math.cos(a), oy + (orr + r) * math.sin(a)
            khe_ho = -2
        else:
            cx, cy = rng.uniform(r + 12, rong - r - 12), rng.uniform(r + 12, cao - r - 12)
            khe_ho = 14
        if not (r + 8 < cx < rong - r - 8 and r + 8 < cy < cao - r - 8):   # ra ngoài ảnh
            continue
        if cx - r < vx + canh + 70 and cy + r > vy - 14:                  # đè lên vật chuẩn
            continue
        bi_chong = False
        for ox, oy, orr in da_dat:
            if math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2) < r + orr + khe_ho:
                bi_chong = True
        if bi_chong:
            continue

        do_chin = int(rng.integers(3))
        qua = ve_qua(do_chin, r, rng)
        dan_qua(nen, qua, cx, cy)
        d_mm = duong_kinh_that(qua, k)
        da_dat.append((cx, cy, r))
        dap_an.append({"do_chin": do_chin, "co": phan_loai_co(d_mm), "d_mm": d_mm, "tam": (cx, cy)})

    anh = them_do_sang_va_nhieu(nen.astype(np.float32), do_sang, nhieu)
    return anh, dap_an, k

# 3. BĂNG TẢI (video)
# 3. Băng tải mô phỏng
# Quả chạy trái -> phải | dap_an: bảng 3 x 3 thật để so với bảng đếm
def tao_bang_tai(so_qua=10, seed=None, rong=960, cao=420, k=2.2):
    """Chuẩn bị băng tải. Trả về dict lưu trạng thái. Quả xếp hàng bên TRÁI ngoài khung,
    mỗi khung hình cả hàng dịch sang phải 'toc_do' pixel."""
    rng = np.random.default_rng(seed)
    bt = {"rong": rong, "cao": cao, "k": k, "ray": 62, "chu_ky_vach": 80,
          "qua": [], "quang_duong": 0.0, "dap_an": np.zeros((3, 3), int)}
    x = 0.0
    for _ in range(so_qua):
        r = int(rng.uniform(35, 75) * k / 2)
        x -= r + rng.uniform(30, 150)                     # khoảng cách ngẫu nhiên giữa 2 quả
        do_chin = int(rng.integers(3))
        qua = ve_qua(do_chin, r, rng)                     # vẽ 1 lần, các khung sau chỉ việc dán
        qua["x_bat_dau"] = x
        qua["y"] = rng.uniform(bt["ray"] + r + 6, cao - bt["ray"] - r - 6)
        co = phan_loai_co(duong_kinh_that(qua, k))
        bt["dap_an"][do_chin, TEN_CO.index(co)] += 1       # đáp án: bảng 3 x 3 giống bảng đếm
        bt["qua"].append(qua)
        x -= r
    bt["quang_duong_het"] = -x + rong + 40                # đi đủ xa thì quả cuối đã ra khỏi khung

    # Vẽ sẵn nền băng tải (rộng hơn khung 1 chu kỳ vạch để cắt ra tạo hiệu ứng chuyển động)
    ray, w = bt["ray"], rong + bt["chu_ky_vach"]
    nen = np.zeros((cao, w, 3), np.float32)
    nen[:] = (200, 203, 204)                                            # mặt băng màu xám nhạt
    for xv in range(0, w, bt["chu_ky_vach"]):
        nen[ray:cao - ray, xv:xv + 3] = (186, 189, 190)                 # vạch kẻ trên mặt băng
    nen += cv2.GaussianBlur(rng.normal(0, 3, (cao, w)).astype(np.float32), (0, 0), 1)[:, :, None]
    for y0 in (0, cao - ray):                                           # 2 thanh ray kim loại
        do_dam = np.linspace(1.0, 0.85, ray, dtype=np.float32)[:, None, None]
        nen[y0:y0 + ray] = np.array([178, 182, 186], np.float32) * (do_dam if y0 == 0 else do_dam[::-1])
        for xd in range(30, w, 120):                                    # đinh tán
            nen[y0 + ray // 2 - 4:y0 + ray // 2 + 5, xd - 4:xd + 5] = (150, 152, 155)
    bt["nen"] = nen
    return bt

def bang_tai_da_xong(bt):
    return bt["quang_duong"] >= bt["quang_duong_het"]

# Vẽ 1 khung video
# Dịch nền tạo chuyển động | chỉ vẽ quả đang trong khung
def khung_hinh_bang_tai(bt, do_sang=1.0, nhieu=0.0):
    """Vẽ khung hình hiện tại. Gọi xong thì cộng bt['quang_duong'] += tốc độ để sang khung sau."""
    lech = int(bt["quang_duong"]) % bt["chu_ky_vach"]
    anh = bt["nen"][:, bt["chu_ky_vach"] - lech:bt["chu_ky_vach"] - lech + bt["rong"]].copy()
    for qua in bt["qua"]:
        cx = qua["x_bat_dau"] + bt["quang_duong"]
        if -2 * qua["r"] < cx < bt["rong"] + 2 * qua["r"]:             # chỉ vẽ quả đang trong khung
            dan_qua(anh, qua, cx, qua["y"])
    canh = int(round(CANH_VAT_CHUAN_MM * bt["k"]))                      # vật chuẩn gắn trên ray
    y = (bt["ray"] - canh) // 2
    anh[y:y + canh, 20:20 + canh] = 22
    return them_do_sang_va_nhieu(anh, do_sang, nhieu, dung_mau_co_san=True)

# 4. Đánh giá ảnh tĩnh
# Ghép quả nhận dạng với đáp án gần nhất (tâm cách < nửa đường kính)
# Chỉ số: đúng độ chín | đúng cỡ | sai số đường kính (mm)
def danh_gia_anh_tinh(ds_qua, dap_an):
    """Ghép mỗi quả nhận dạng với đáp án gần nhất (tâm cách nhau < nửa đường kính) rồi so sánh."""
    cac_cap = []
    for i, q in enumerate(ds_qua):
        for j, g in enumerate(dap_an):
            kc = math.sqrt((q["tam"][0] - g["tam"][0]) ** 2 + (q["tam"][1] - g["tam"][1]) ** 2)
            cac_cap.append((kc, i, j))
    cac_cap.sort()
    da_dung_i, da_dung_j = [], []
    dung_do_chin, dung_co, sai_so_mm = 0, 0, []
    for kc, i, j in cac_cap:
        if i in da_dung_i or j in da_dung_j or kc > 0.5 * ds_qua[i]["d_px"]:
            continue
        da_dung_i.append(i)
        da_dung_j.append(j)
        q, g = ds_qua[i], dap_an[j]
        dung_do_chin += q["do_chin"] == g["do_chin"]
        dung_co += q["co"] == g["co"]
        if q["d_mm"] is not None:
            sai_so_mm.append(abs(q["d_mm"] - g["d_mm"]))
    return {"so_qua_that": len(dap_an), "so_qua_dem": len(ds_qua),
            "dung_do_chin": dung_do_chin, "dung_co": dung_co, "sai_so_mm": sai_so_mm}

# Đánh giá băng tải
# Mỗi ô bảng 3 x 3 lấy số nhỏ hơn của bảng đếm và bảng đáp án = số quả đúng loại
def danh_gia_bang_tai(bang_dem, bang_dap_an):
    """So 2 bảng 3 x 3. Ô nào đếm đúng bao nhiêu quả thì lấy số nhỏ hơn của 2 bảng."""
    return {"so_qua_that": int(bang_dap_an.sum()), "so_qua_dem": int(bang_dem.sum()),
            "dung_loai": int(np.minimum(bang_dem, bang_dap_an).sum())}

# 5. Đánh giá hàng loạt: python part_3_simulation.py
# 4 điều kiện: chuẩn | tối 60% | nhiễu sigma = 20 | quả chạm nhau
# 15 ảnh x 10 quả x 4 điều kiện = 600 quả
# Băng tải: 10 video, tốc độ 5 -> 20 pixel/khung
# Lưu ket_qua_danh_gia.csv
if __name__ == "__main__":
    from part_2_measuring_and_counting import phan_tich_anh, tao_bo_dem, cap_nhat_bo_dem

    DIEU_KIEN = [  # (tên, độ sáng, nhiễu σ, cho chạm nhau)
        ("Chuan", 1.0, 0, False),
        ("Toi 60%", 0.6, 0, False),
        ("Nhieu sigma=20", 1.0, 20, False),
        ("Qua cham nhau", 1.0, 0, True),
    ]
    SO_ANH = 15                                # 15 ảnh x 10 quả x 4 điều kiện = 600 quả
    cac_dong_csv = []
    thoi_gian_xu_ly = []
    print(f"{'Dieu kien':<16}{'So qua':>7}{'Dem dung':>10}{'Do chin':>9}{'Co S/M/L':>10}{'Sai so TB':>11}{'Sai so max':>11}")
    for ten, do_sang, nhieu, cho_cham in DIEU_KIEN:
        tong_qua = sai_dem = dung_do_chin = dung_co = 0
        sai_so = []
        for s in range(SO_ANH):
            anh, dap_an, k = tao_anh_tinh(so_qua=10, seed=1000 + s, do_sang=do_sang, nhieu=nhieu, cho_cham=cho_cham)
            t0 = time.perf_counter()
            kq = phan_tich_anh(anh)
            thoi_gian_xu_ly.append(time.perf_counter() - t0)
            dg = danh_gia_anh_tinh(kq["qua"], dap_an)
            tong_qua += dg["so_qua_that"]
            sai_dem += abs(dg["so_qua_dem"] - dg["so_qua_that"])
            dung_do_chin += dg["dung_do_chin"]
            dung_co += dg["dung_co"]
            sai_so += dg["sai_so_mm"]
        dong = [ten, tong_qua, 100 * (1 - sai_dem / tong_qua), 100 * dung_do_chin / tong_qua,
                100 * dung_co / tong_qua, np.mean(sai_so), np.max(sai_so)]
        cac_dong_csv.append(["anh_tinh"] + dong)
        print(f"{ten:<16}{tong_qua:>7}{dong[2]:>9.1f}%{dong[3]:>8.1f}%{dong[4]:>9.1f}%{dong[5]:>9.2f}mm{dong[6]:>9.2f}mm")

    # Băng tải: 10 video, tốc độ tăng dần từ 5 đến 20 pixel/khung
    tong_that = tong_dem = tong_dung = so_khung = 0
    t0 = time.perf_counter()
    for s in range(10):
        bt = tao_bang_tai(so_qua=10, seed=2000 + s)
        bo_dem = tao_bo_dem()
        toc_do = 5 + s * 15 / 9
        while not bang_tai_da_xong(bt):
            kq = phan_tich_anh(khung_hinh_bang_tai(bt, nhieu=5))
            cap_nhat_bo_dem(bo_dem, kq["qua"], bt["rong"] / 2, kq["k"])
            bt["quang_duong"] += toc_do
            so_khung += 1
        dg = danh_gia_bang_tai(bo_dem["bang"], bt["dap_an"])
        tong_that += dg["so_qua_that"]
        tong_dem += dg["so_qua_dem"]
        tong_dung += dg["dung_loai"]
    fps = so_khung / (time.perf_counter() - t0)
    sai_dem = 100 * abs(tong_dem - tong_that) / tong_that
    print(f"\nBang tai (10 video): that {tong_that} qua, dem {tong_dem} qua -> sai so dem {sai_dem:.1f}%, "
          f"dung loai (do chin x co) {100 * tong_dung / tong_that:.1f}%")
    print(f"Toc do: anh tinh {1 / np.mean(thoi_gian_xu_ly):.0f} anh/s; bang tai (tao + xu ly) {fps:.0f} khung/s")
    cac_dong_csv.append(["bang_tai", "10 video", tong_that, 100 - sai_dem, 100 * tong_dung / tong_that, "", "", ""])

    duong_dan = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ket_qua_danh_gia.csv")
    with open(duong_dan, "w", newline="", encoding="utf-8-sig") as f:
        ghi = csv.writer(f)
        ghi.writerow(["loai", "dieu_kien", "so_qua", "dem_dung_%", "do_chin_dung_%", "co_dung_%",
                      "sai_so_TB_mm", "sai_so_max_mm"])
        ghi.writerows(cac_dong_csv)
    print("Da luu", duong_dan)
