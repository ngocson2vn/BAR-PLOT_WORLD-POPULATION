#%% Nhập thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import cv2


#%% Hàm tạo thứ bậc
def ranking_data(data, var, by):
    # data: là bộ dữ liệu
    # var: là cột số dùng để xếp hạng trong mỗi giá trị của cột by (ví dụ: xếp hạng theo cột var của từng năm)
    # by: là cột thể hiện từng giai đoạn, từng phần của bộ dữ liệu
    # ==> Tạo thêm cột rank xếp hạng theo giảm dần (lớn nhất là hạng 1)
     rank = data.groupby(by)[var].rank(method = 'dense', ascending = False).astype(int)
     data['rank'] = rank
     return data

#%% funtion làm mượt dữ liệu
def smoothing_data(data, var, by, fps, ascending=True):
    # data: là bộ dữ liệu
    # var: là cột số cần làm mượt. 1 cột hoặc 1 list các cột
    # by: là cột cố định trong khi làm mượt
    # fps: số  lần chia nhỏ dữ liệu (1 lần là 1 khung hình)
    # ==> Hàm này sẽ làm mượt dữ liệu cột bar theo từng bước thay đổi của cột by,
    # ==> mỗi bước thay đổi của cột by sẽ chia nhỏ cột var thành fps lần thay đổi (VD: từ 1-5 chia nhỏ thành: 1-2-3-4-5)
    # ==> data sẽ được bổ sung thêm cột fps - sẽ là cột để làm giá trị chạy cho biểu đồ.
    # ==> cột var được tạo thêm các giá trị trung gian để làm mượt giữ liệu

    # Tạo list rỗng để chứa các giá trị
    list_of_dic = []
    # danh sách các cột không thay đổi
    column_names = list(data.columns.values)
    if type(var) != list:
        var = [var]
    for i in var:
        column_names.remove(i)
    # sắp xếp dữ liệu theo cột by cố định
    data = data.sort_values(by=by, ascending=ascending).reset_index(drop=True)

    # Vòng lặp đầu tiên đi từ đầu đến cuối bộ dữ liệu
    # Lưu ý: dữ liệu cần được sắp
    for i in range(data.shape[0]):
        if i == 0:
            # Tại hàng đầu tiên của dữ liệu
            dic = {}
            for c in column_names:
                dic[c] = data[c][i]
            for v in var:
                if v == 'rank':
                    dic[v] = data[v][i]
                else:
                    dic[v] = int(data[v][i])
            dic['fps'] = str(data[by][i]) + '-' + str(fps)
            list_of_dic.append(dic)
        else:
            # Tại các dòng tiếp theo
            # Lặp để tạo thêm các số liệu trung gian nhằm tăng độ mượt
            # Với số khung hình (fps) được chọn
            for j in range(fps):
                dic = {}
                for c in column_names:
                    dic[c] = data[c][i]

                for v in var:
                    phan_du = data[v][i] - data[v][i - 1]
                    value = data[v][i - 1] + ((j + 1) / fps) * phan_du
                    if v == 'rank':
                        value = value
                    else:
                        value = int(value)
                    dic[v] = value
                dic['fps'] = str(data[by][i]) + '-' + '0' * (len(str(fps)) - len(str(j + 1))) + str(j + 1)
                list_of_dic.append(dic)
    return pd.DataFrame(list_of_dic)

#%% Funtin vẽ biểu đồ
def create_hbar(data_country, data_world, i, ytick, x , y = 'rank', top = 20):
    # Lọc chỉ giữ lại data theo năm
    data = data_country[data_country['fps'] == i]
    # Giữ lại các hàng có rank lớn hơn top để đưa lân biểu đồ (có thể số lượng nhiều hơn rank bởi trong trường hợp thay đổi vị trí)
    data = data[data[y] <= top]
    # Sắp xếp dữ liệu theo dân số từ lớn đến nhỏ
    data = data.sort_values(by = y)
    data = data.reset_index(drop = True)
    data['rank'] = data['rank'] * -1
    # Tạo trang giấy có độ dài chiều rộng 9 inch - chiều dài 6 inch
    plt.style.use('dark_background')
    fig = plt.figure(figsize  = (9.6, 5.4), dpi = 100)
    # Set slyle có sẵn trong thư viện seaborn
    # sb.set_style("dark")
    # Vẽ biểu đồ hình thanh với trục x là dân số còn trục y là quốc gia
    plt.barh(y = data[y], width = data[x], height = 0.7, color = sb.color_palette("YlOrRd_r", 20),
             tick_label = data[ytick])
    # Tạo và định dạng tiêu đề của biểu đồ
    # title = 'TOP 20 Quốc gia dân số lớn nhất'.upper()
    # title_obj = plt.title(title)
    # plt.setp(title_obj, color='orangered', fontsize = 22, fontweight="bold", fontname='Times New Roman Bold')
    # Tạo và định dạng nhãn của các trục
    plt.tick_params(axis="y", labelsize=8, labelrotation=15, labelcolor= "w")
    # plt.tick_params(axis="x", labelsize=6, labelrotation=0, labelcolor="w", bottom= True, top = True, labeltop = True, labelbottom = True)

    x_max = data[x].max()
    if x_max >= 840000000:
        xlim = (1.2 * x_max) # cố định trục x
    else:
        xlim = 1000000000 # cố định trục x

    plt.xlim(right = xlim) # cố định trục x
    plt.ylim(bottom = (-1 * top)-1) # cố định trục y

    # xoá nhãn trục x
    plt.xlabel('')
    # Xóa nhãn trục y
    plt.ylabel('')
    # Tạo và thêm các giá trị số chạy trên biểu đồ
    for t in range(data.shape[0]):
        # Thêm giá trị của dân số vào biểu đồ
        value = f'{int(data.iloc[t][x]):,}' # Định dạng kiểu số: 123,456,000
        plt.text(data.iloc[t][x] + 20000000, data.iloc[t][y], # Tọa độ
                value, # Giá trị
                color='g', va="center", fontsize = 7) # Định dạng
        # Thêm tên của quốc gia
        plt.text(-0.015 * xlim, data.iloc[t][y],  # Tọa độ
                 data['country'][t],  # Giá trị
                 color='w', va="center", ha='right', fontsize= 7, rotation= 15)  # Định dạng

    # Tạo chữ trung tâm
    plt.text(0.70 * xlim, -9,
             "TỔNG DÂN SỐ\n THẾ GIỚI",
            color='firebrick', va="center", ha="right", fontsize = 20, fontweight="bold")
    plt.text(0.72 * xlim, -9, # Tọa độ
             i.split('-')[0], # Giá trị
            color='orangered', va="center", ha="left", fontsize = 40, fontweight="bold")    # Định dạng
    # Tạo tổng dân số thế giới ở trung tâm
    world_population = f'{int(data_world[data_world["fps"] == i][x]):,}' # Dân số thế giới
    plt.text(0.67 * xlim, -13, # Tọa độ
             world_population, # Giá trị
            color='g', va="center", ha="center", fontsize = 40, fontweight="bold")    # Định dạng
    # Tạo chữ ký
    plt.text(0.99 * xlim, (-1 * top),
             'Trình bày: Võ Văn Thương\nNguồn dữ liệu: https://data.worldbank.org',
            color='w', va="center", ha="right", fontsize = 5)
    plt.axis('off')
    # save dạng numpy array
    fig.canvas.draw ()
    image_from_plot = fig.canvas.tostring_rgb()
    image_from_plot = np.frombuffer(image_from_plot, dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    image_from_plot = image_from_plot.reshape(h, w, 3)
    # Conver qua mảng BGR theo định dạng của OpenCV
    image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return image_from_plot


