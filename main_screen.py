"""
MÀN HÌNH GIAO DIỆN KHỞI ĐỘNG
PHÂN LOẠI ĐỘ CHÍN, ĐO KÍCH THƯỚC VÀ ĐẾM CÀ CHUA BẰNG XỬ LÝ ẢNH
"""
import base64
import os
import tkinter as tk
from tkinter import font as tkfont

import cv2
import numpy as np

THU_MUC = os.path.dirname(os.path.abspath(__file__))
# Tự động nhận thư mục 'picture' hoặc 'hinh_anh'
THU_MUC_ANH = os.path.join(THU_MUC, "picture") if os.path.isdir(os.path.join(THU_MUC, "picture")) else os.path.join(THU_MUC, "hinh_anh")

RONG_TK, CAO_TK = 1750, 899          # kích thước khung thiết kế (pixel)
FONT = "Arial"                        # Arial có đủ dấu tiếng Việt trên Windows

NEN = "#ffffff"
DEN = "#111111"
DO = "#b91c1c"
XANH_DAM = "#1c1c9c"
NUT_NEN = "#7fd3f7"
NUT_CHU = "#dc2626"

# Chữ: (nội dung, tâm x, tâm y, cỡ chữ, đậm, màu, neo, rộng tối đa)   -- tọa độ trên khung thiết kế
CHU = [
    ("BỘ GIÁO DỤC VÀ ĐÀO TẠO",                              875,  50, 40, False, DEN, "center", 1150),
    ("TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP. HỒ CHÍ MINH",    890, 100, 40, True,  DEN, "center", 1300),
    ("KHOA ĐIỆN – ĐIỆN TỬ",                                  880, 153, 40, True,  DEN, "center", 1150),
    ("BÁO CÁO ĐỀ TÀI CUỐI KỲ",                               875, 232, 40, True,  DO,  "center", 1150),
    ("MÔN HỌC: XỬ LÝ ẢNH",                                   880, 288, 40, True,  DO,  "center", 1150),
    ("ĐỀ TÀI: PHÂN LOẠI ĐỘ CHÍN, ĐO KÍCH THƯỚC VÀ ĐẾM SỐ LƯỢNG", 885, 348, 50, True, XANH_DAM, "center", 1660),
    ("CÀ CHUA BẰNG XỬ LÝ ẢNH TRÊN DỮ LIỆU MÔ PHỎNG",          895, 410, 50, True,  XANH_DAM, "center", 1660),
    ("ẢNH MẪU THỰC NGHIỆM",                                  280, 812, 34, True,  DEN, "center", 470),
    ("Tp. Hồ Chí Minh, năm 2026",                            885, 846, 36, True,  DEN, "center", 700),
    ("• Phân loại độ chín (HSI Color)",                     1280, 548, 28, True,  DEN, "w", 470),
    ("• Đo kích thước & Phân cỡ (S/M/L)",                   1280, 591, 28, True,  DEN, "w", 470),
    ("• Đếm trên ảnh tĩnh & Băng tải",                      1280, 634, 28, True,  DEN, "w", 470),
]

# Ảnh: (tên file, tâm x, tâm y, rộng, cao)   -- tọa độ trên khung thiết kế
ANH = [
    ("logo_hcmute.png",     88, 105, 165, 185),
    ("logo_ute.png",      1650, 100, 200, 180),
    ("picture_test_1.jpg", 252, 618, 325, 335),
    ("banh_xe_mau.png",    876, 627, 328, 337),
]

# Nút START: (tâm x, tâm y, rộng, cao)
NUT_START = (1502, 739, 262, 112)


def doc_anh(ten_file):
    """Đọc ảnh bằng imdecode (đường dẫn có dấu tiếng Việt vẫn đọc được). Lỗi -> None."""
    duong_dan = os.path.join(THU_MUC_ANH, ten_file)
    if not os.path.exists(duong_dan):
        return None
    return cv2.imdecode(np.fromfile(duong_dan, np.uint8), cv2.IMREAD_COLOR)


def sang_anh_tkinter(anh_bgr):
    """Đổi ảnh OpenCV (BGR) -> ảnh Tkinter. Dùng PPM (nhanh), lỗi thì dùng PNG."""
    rgb = cv2.cvtColor(anh_bgr, cv2.COLOR_BGR2RGB)
    cao, rong = rgb.shape[:2]
    try:
        return tk.PhotoImage(width=rong, height=cao, data=b"P6 %d %d 255 " % (rong, cao) + rgb.tobytes(),
                             format="PPM")
    except tk.TclError:
        _, buf = cv2.imencode(".png", anh_bgr)
        return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))


class ManHinhDau:
    """Trang bìa. khi_bat_dau: hàm được gọi khi bấm START (main.py truyền vào để mở giao diện chính)."""

    def __init__(self, root, khi_bat_dau):
        self.root = root
        self.khi_bat_dau = khi_bat_dau
        self.hen_gio = None
        self.anh_tk = []                             # giữ tham chiếu ảnh (nếu không Python xóa mất ảnh)
        self.anh_goc = {t[0]: doc_anh(t[0]) for t in ANH}

        root.title("Phân loại độ chín, đo kích thước và đếm cà chua")
        root.configure(bg=NEN)
        root.minsize(900, 480)
        try:
            root.state("zoomed")                     # phóng to toàn màn hình (Windows)
        except tk.TclError:
            root.geometry("1440x800")

        self.khung = tk.Frame(root, bg=NEN)          # khung ngoài: phủ kín cửa sổ
        self.khung.pack(fill="both", expand=True)
        self.tho = tk.Frame(self.khung, bg=NEN)      # khung thiết kế: giữ đúng tỉ lệ 1750:899, đặt giữa
        self.tho.place(relx=0.5, rely=0.5, anchor="center", width=RONG_TK, height=CAO_TK)
        self.khung.bind("<Configure>", self.khi_doi_kich_thuoc)

        root.bind("<Return>", lambda e: self.bat_dau())
        root.bind("<space>", lambda e: self.bat_dau())

    # Vẽ lại khi đổi kích thước (hẹn 60 ms để kéo cửa sổ không bị giật)
    def khi_doi_kich_thuoc(self, _=None):
        if self.hen_gio:
            self.root.after_cancel(self.hen_gio)
        self.hen_gio = self.root.after(60, self.ve)

    def ve(self):
        self.hen_gio = None
        rong, cao = self.khung.winfo_width(), self.khung.winfo_height()
        if rong < 50 or cao < 50:
            return
        s = min(rong / RONG_TK, cao / CAO_TK)        # hệ số thu / phóng
        self.tho.place_configure(width=int(RONG_TK * s), height=int(CAO_TK * s))
        for w in self.tho.winfo_children():          # xóa hết rồi vẽ lại theo hệ số mới
            w.destroy()
        self.anh_tk = []

        for ten_file, cx, cy, w, h in ANH:           # ---- ảnh
            anh = self.anh_goc[ten_file]
            if anh is None:
                continue
            kich_thuoc = (max(int(w * s), 1), max(int(h * s), 1))
            anh = cv2.resize(anh, kich_thuoc, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            self.anh_tk.append(sang_anh_tkinter(anh))
            tk.Label(self.tho, image=self.anh_tk[-1], bg=NEN, bd=0).place(
                x=int(cx * s), y=int(cy * s), anchor="center")

        for noi_dung, cx, cy, co, dam, mau, neo, rong_max in CHU:      # ---- chữ
            co_chu = self.vua_chu(noi_dung, int(co * s), dam, int(rong_max * s))
            tk.Label(self.tho, text=noi_dung, bg=NEN, fg=mau,
                     font=(FONT, -co_chu, "bold" if dam else "normal")).place(
                x=int(cx * s), y=int(cy * s), anchor=neo if neo == "center" else "w")

        cx, cy, w, h = NUT_START                      # ---- nút START
        nut = tk.Button(self.tho, text="START", command=self.bat_dau, bg=NUT_NEN, fg=NUT_CHU,
                        activebackground="#5cc4f0", activeforeground=NUT_CHU, relief="raised",
                        bd=max(int(4 * s), 2), font=(FONT, -int(62 * s), "bold"), cursor="hand2")
        nut.place(x=int((cx - w / 2) * s), y=int((cy - h / 2) * s), width=int(w * s), height=int(h * s))
        nut.focus_set()

    def vua_chu(self, noi_dung, co, dam, rong_max):
        """Giảm cỡ chữ (pixel) cho đến khi chữ vừa trong rong_max -> không bao giờ bị cắt."""
        co = max(co, 8)
        while co > 8:
            f = tkfont.Font(family=FONT, size=-co, weight="bold" if dam else "normal")
            if f.measure(noi_dung) <= rong_max:
                break
            co -= 1
        return co

    def bat_dau(self):
        """START: bỏ trang bìa, mở giao diện chương trình chính."""
        if self.hen_gio:
            self.root.after_cancel(self.hen_gio)
            self.hen_gio = None
        self.root.unbind("<Return>")
        self.root.unbind("<space>")                  # nhường phím Space cho chức năng tạm dừng ở giao diện chính
        self.khung.destroy()
        self.khi_bat_dau()


# Chạy riêng để xem thử trang bìa:  python man_hinh_dau.py
if __name__ == "__main__":
    gui = tk.Tk()
    ManHinhDau(gui, lambda: print("START -> mở giao diện chính"))
    gui.mainloop()
