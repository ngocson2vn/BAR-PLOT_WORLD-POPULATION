# %% Nhập thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
from funtion import *


# %% xử lý dữ liệu
population = pd.read_excel('/Users/vovanthuong/Documents/BAR-PLOT_WORLD-POPULATION/data/population by country.xls',
                           sep=' ', encoding='utf-8')
world = pd.read_excel('/Users/vovanthuong/Documents/BAR-PLOT_WORLD-POPULATION/data/population - world.xls',
                      sep=' ', encoding='utf-8')
# %% Xóa đi các cột dư trong dataframe population
population = population.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'])
# Chuyển dữ liệu từ cột sang hàng
population = population.melt(id_vars='Country Name')
population = population.rename(columns={'Country Name': 'country', 'variable': 'year', 'value': 'population'})
population = population[~ population.population.isna()]
population = population.astype({'population': int})

#%%
# Xếp hạng theo từng năm
population = ranking_data(data=population, var='population', by='year')

#%%
# Chỉ giữ lại các nước có hạng >= 21 để làm nhẹ cho việc làm mượt dữ liệu
population = population[population['rank'] <= 21]

#%%
population['rank2'] =  population['rank']

#%%
# Làm mượt dữ liệu của dân số thế giới
world2 = smoothing_data(data=world, var='population', by='year', fps=45)
# Làm mượt giữ liệu theo từng quốc gia và ghép lại với nhau

#%%
population2 = pd.DataFrame()
for i in population.country.unique():
    data = population[population.country == i]
    data = smoothing_data(data=data, var=['population', 'rank'], by='year', fps= 45)
    population2 = population2.append(data)

# %% VẼ BIỂU ĐỒ
# Một vòng lặp tương ứng -> 1 biểu đồ
fps = population2.fps.unique()
fps = np.sort(fps)

#%%
list_plot_array = []
i = 0
for f in fps:
    plot_data = create_hbar(data_country=population2, data_world=world2,
                            i=f, ytick='country', x='population', y='rank', top=20)
    list_plot_array.append(plot_data)
    i += 1
    print('Hình thứ:', i, '/', len(fps))

# %% RENDER VIDEO
height, width, layers = list_plot_array[0].shape
size = (width, height)
video_name = '/content/gdrive/My Drive/Colab Notebooks/DAN_SO_THE_GIOI/top_20_dan_so_tg.avi'
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 14, size)
for i in range(len(list_plot_array)):
    video.write(list_plot_array[i])
video.release()
