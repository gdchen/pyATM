#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:29:36 2022

@author: yaoyichen
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset,num2date,date2num


from scipy.interpolate import interpn    


def interpolate_2d(x_ref, y_ref, value_ref, x_query, y_query):
    """
    经度，纬度，值，待差值的经度，待差值的纬度
    所有都是numpy 类型的 
    
    """ 
    points_ref = (x_ref, y_ref)
    points_query = (x_query, y_query)
    
    value_query = interpn(points_ref, value_ref, points_query, method = "linear")
    return value_query

lon_vector_360 = np.asarray([-179.5, -178.5, -177.5, -176.5, -175.5, -174.5, -173.5, -172.5,
       -171.5, -170.5, -169.5, -168.5, -167.5, -166.5, -165.5, -164.5,
       -163.5, -162.5, -161.5, -160.5, -159.5, -158.5, -157.5, -156.5,
       -155.5, -154.5, -153.5, -152.5, -151.5, -150.5, -149.5, -148.5,
       -147.5, -146.5, -145.5, -144.5, -143.5, -142.5, -141.5, -140.5,
       -139.5, -138.5, -137.5, -136.5, -135.5, -134.5, -133.5, -132.5,
       -131.5, -130.5, -129.5, -128.5, -127.5, -126.5, -125.5, -124.5,
       -123.5, -122.5, -121.5, -120.5, -119.5, -118.5, -117.5, -116.5,
       -115.5, -114.5, -113.5, -112.5, -111.5, -110.5, -109.5, -108.5,
       -107.5, -106.5, -105.5, -104.5, -103.5, -102.5, -101.5, -100.5,
        -99.5,  -98.5,  -97.5,  -96.5,  -95.5,  -94.5,  -93.5,  -92.5,
        -91.5,  -90.5,  -89.5,  -88.5,  -87.5,  -86.5,  -85.5,  -84.5,
        -83.5,  -82.5,  -81.5,  -80.5,  -79.5,  -78.5,  -77.5,  -76.5,
        -75.5,  -74.5,  -73.5,  -72.5,  -71.5,  -70.5,  -69.5,  -68.5,
        -67.5,  -66.5,  -65.5,  -64.5,  -63.5,  -62.5,  -61.5,  -60.5,
        -59.5,  -58.5,  -57.5,  -56.5,  -55.5,  -54.5,  -53.5,  -52.5,
        -51.5,  -50.5,  -49.5,  -48.5,  -47.5,  -46.5,  -45.5,  -44.5,
        -43.5,  -42.5,  -41.5,  -40.5,  -39.5,  -38.5,  -37.5,  -36.5,
        -35.5,  -34.5,  -33.5,  -32.5,  -31.5,  -30.5,  -29.5,  -28.5,
        -27.5,  -26.5,  -25.5,  -24.5,  -23.5,  -22.5,  -21.5,  -20.5,
        -19.5,  -18.5,  -17.5,  -16.5,  -15.5,  -14.5,  -13.5,  -12.5,
        -11.5,  -10.5,   -9.5,   -8.5,   -7.5,   -6.5,   -5.5,   -4.5,
         -3.5,   -2.5,   -1.5,   -0.5,    0.5,    1.5,    2.5,    3.5,
          4.5,    5.5,    6.5,    7.5,    8.5,    9.5,   10.5,   11.5,
         12.5,   13.5,   14.5,   15.5,   16.5,   17.5,   18.5,   19.5,
         20.5,   21.5,   22.5,   23.5,   24.5,   25.5,   26.5,   27.5,
         28.5,   29.5,   30.5,   31.5,   32.5,   33.5,   34.5,   35.5,
         36.5,   37.5,   38.5,   39.5,   40.5,   41.5,   42.5,   43.5,
         44.5,   45.5,   46.5,   47.5,   48.5,   49.5,   50.5,   51.5,
         52.5,   53.5,   54.5,   55.5,   56.5,   57.5,   58.5,   59.5,
         60.5,   61.5,   62.5,   63.5,   64.5,   65.5,   66.5,   67.5,
         68.5,   69.5,   70.5,   71.5,   72.5,   73.5,   74.5,   75.5,
         76.5,   77.5,   78.5,   79.5,   80.5,   81.5,   82.5,   83.5,
         84.5,   85.5,   86.5,   87.5,   88.5,   89.5,   90.5,   91.5,
         92.5,   93.5,   94.5,   95.5,   96.5,   97.5,   98.5,   99.5,
        100.5,  101.5,  102.5,  103.5,  104.5,  105.5,  106.5,  107.5,
        108.5,  109.5,  110.5,  111.5,  112.5,  113.5,  114.5,  115.5,
        116.5,  117.5,  118.5,  119.5,  120.5,  121.5,  122.5,  123.5,
        124.5,  125.5,  126.5,  127.5,  128.5,  129.5,  130.5,  131.5,
        132.5,  133.5,  134.5,  135.5,  136.5,  137.5,  138.5,  139.5,
        140.5,  141.5,  142.5,  143.5,  144.5,  145.5,  146.5,  147.5,
        148.5,  149.5,  150.5,  151.5,  152.5,  153.5,  154.5,  155.5,
        156.5,  157.5,  158.5,  159.5,  160.5,  161.5,  162.5,  163.5,
        164.5,  165.5,  166.5,  167.5,  168.5,  169.5,  170.5,  171.5,
        172.5,  173.5,  174.5,  175.5,  176.5,  177.5,  178.5,  179.5])
lat_vector_180 = np.asarray([-89.5, -88.5, -87.5, -86.5, -85.5, -84.5, -83.5, -82.5, -81.5,
       -80.5, -79.5, -78.5, -77.5, -76.5, -75.5, -74.5, -73.5, -72.5,
       -71.5, -70.5, -69.5, -68.5, -67.5, -66.5, -65.5, -64.5, -63.5,
       -62.5, -61.5, -60.5, -59.5, -58.5, -57.5, -56.5, -55.5, -54.5,
       -53.5, -52.5, -51.5, -50.5, -49.5, -48.5, -47.5, -46.5, -45.5,
       -44.5, -43.5, -42.5, -41.5, -40.5, -39.5, -38.5, -37.5, -36.5,
       -35.5, -34.5, -33.5, -32.5, -31.5, -30.5, -29.5, -28.5, -27.5,
       -26.5, -25.5, -24.5, -23.5, -22.5, -21.5, -20.5, -19.5, -18.5,
       -17.5, -16.5, -15.5, -14.5, -13.5, -12.5, -11.5, -10.5,  -9.5,
        -8.5,  -7.5,  -6.5,  -5.5,  -4.5,  -3.5,  -2.5,  -1.5,  -0.5,
         0.5,   1.5,   2.5,   3.5,   4.5,   5.5,   6.5,   7.5,   8.5,
         9.5,  10.5,  11.5,  12.5,  13.5,  14.5,  15.5,  16.5,  17.5,
        18.5,  19.5,  20.5,  21.5,  22.5,  23.5,  24.5,  25.5,  26.5,
        27.5,  28.5,  29.5,  30.5,  31.5,  32.5,  33.5,  34.5,  35.5,
        36.5,  37.5,  38.5,  39.5,  40.5,  41.5,  42.5,  43.5,  44.5,
        45.5,  46.5,  47.5,  48.5,  49.5,  50.5,  51.5,  52.5,  53.5,
        54.5,  55.5,  56.5,  57.5,  58.5,  59.5,  60.5,  61.5,  62.5,
        63.5,  64.5,  65.5,  66.5,  67.5,  68.5,  69.5,  70.5,  71.5,
        72.5,  73.5,  74.5,  75.5,  76.5,  77.5,  78.5,  79.5,  80.5,
        81.5,  82.5,  83.5,  84.5,  85.5,  86.5,  87.5,  88.5,  89.5])

lon_vector_72 = np.arange(-180, 180, 5)
lat_vector_46 = np.arange(-90,  94,  4)

def merge_inversion_result():
    """
    将不同月份的数据汇总后打印
    """
    date_list = ["20180101","20180116","20180201","20180220","20180301","20180316",
                 "20180401","20180416","20180501","20180516","20180601","20180616",
                 "20180701","20180716","20180801","20180816","20180901","20180916",
                 "20181001","20181016","20181101","20181116","20181201","20181216"]
    
    if_merge2 = True
    if(if_merge2 == True):
        f_var_np = np.empty([12, 46, 72])
    else:
        f_var_np = np.empty([24, 46, 72])
        
    t_var_np = []
    sum_list = []
    
    frame_index = 0
    for index, date_str in enumerate(date_list):
        input_file = f"/Users/yaoyichen/dataset/auto_experiment/experiment_0/full2018/real_inversion_satellite_nc/real_inversion_satellite_{date_str}/inversion_result_004.nc"
        nc_in =  nc.Dataset(input_file)
        tt = nc_in["f"][0,:,:,0]
        
        # sum_value = tt.sum()
        # sum_list.append(sum_value) 
        
        # print(date_str, sum_value)
        
        if(if_merge2 == True):
            if(index%2 ==0):
                tt_temp = tt
            else:
                t_var_np.append(frame_index*31*1440)
                f_var_np[frame_index,:,:] = (tt + tt_temp)/2.0
                frame_index += 1
        else:
            t_var_np.append(frame_index*15.5*1440)
            f_var_np[frame_index,:,:] = tt
            frame_index += 1
        
    
    plt.plot(sum_list)
    
    # f_var_np[-1,:,:] = np.mean(f_var_np[0:24,:,:],axis = 0)
    # t_var_np.append(10000*1440)
    
    #做全球的结果, 将年度结果合并一份


    return f_var_np, t_var_np



def reshape_4672_180360(input_data, lat_vector_46, lon_vector_72, lat_vector_180, lon_vector_360):
    """
    4672 将数据差值到 180360的网格上
    """
    x_ref = lat_vector_46
    y_ref = np.concatenate([lon_vector_72, np.asarray([180,]) ])
    data_ref = np.concatenate([input_data, input_data[:, 0:1]], axis = 1)
    
    tt = np.meshgrid(lon_vector_360, lat_vector_180 )
    
    # 将反演的坐标 flatten到一维
    result = interpolate_2d(x_ref, y_ref, data_ref, 
                            x_query = tt[1].flatten(), y_query = tt[0].flatten() )
    result_reshape = np.reshape(result, [180,360] )
    
    return result_reshape



def write_global_data(output_file, flux_data, time_data, lon_vector, lat_vector  ):
    """
    写nc文件
    """
    ncout = Dataset(output_file, 'w', 'NETCDF4')
    ncout.createDimension('longitude', len(lon_vector))
    ncout.createDimension('latitude', len(lat_vector))
    ncout.createDimension('time', None)
    
    lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
    latvar = ncout.createVariable('latitude', 'float32', ('latitude'))
    timevar = ncout.createVariable('time', 'int32', ('time'))
    timevar.setncattr('units', "minutes since 2000-01-01 00:00:00")
    
    fluxvar = ncout.createVariable("f", 'float32',
                                ('time', 'latitude', 'longitude'))
        
    lonvar[:] = lon_vector
    latvar[:] = lat_vector
    fluxvar[:] = flux_data
    timevar[:] = time_data
    ncout.close()
    pass



f_var_np, t_var_np = merge_inversion_result()

f_var_np_full = np.zeros([12,180,360])

for i in range(12):
    f_var_np_full[i,:,:] = reshape_4672_180360(f_var_np[i,:,:],  lat_vector_46, lon_vector_72, lat_vector_180, lon_vector_360)


output_file = "CT2019B.flux1x1.2018_inversion_merge79.nc"
write_global_data(output_file, f_var_np, t_var_np, lon_vector_72, lat_vector_46)

output_file2 = "CT2019B.flux1x1.2018_inversion_merge79_full.nc"
write_global_data(output_file2, f_var_np_full, t_var_np, lon_vector_360, lat_vector_180)


output_file3 = "CT2019B.flux1x1.2018_inversion_ct.nc"
write_global_data(output_file3, carbontracker_data_full, t_var_np, lon_vector_360, lat_vector_180)

#%%

def get_carbontracker_data(input_file):
    """
    读取 carbontracker数据
    """
    df =  nc.Dataset(input_file)
    bio = df["bio_flux_opt"][:]
    fossil = df["fossil_flux_imp"][:]
    ocean = df["ocn_flux_opt"][:]
    fire = df["fire_flux_imp"][:]
    total = bio + fossil + ocean 
    total = total.data[0,:,:]
    return total

carbontracker_data_full = np.zeros([12,180,360])

for i in range(1,13):
    input_file = "/Users/yaoyichen/dataset/carbon/carbonTracker/monthly_flux/CT2019B.flux1x1.2018" + str(i).zfill(2) + ".nc"
    carbontracker_data = get_carbontracker_data(input_file)
    carbontracker_data_full[i-1,:, : ] = carbontracker_data



def average_on_transcom_regions(value_data, transcom_regions_data):
    """
    在 transcom_region 上做平均
    """
    flux_transcom_regions = np.zeros(transcom_regions_data.shape)
    result_mapping = {}
    for i in range(1,12,1):
        tt_mean = value_data[transcom_regions_data == i].mean()
        flux_transcom_regions[transcom_regions_data == i] = tt_mean
        result_mapping[i] = tt_mean
    return flux_transcom_regions, result_mapping

flux_transcom_regions, result_mapping = average_on_transcom_regions(total, transcom_regions_data)
# plt.imshow(flux_transcom_regions)

#%%
input_file = "/Users/yaoyichen/dataset/carbon/regions.nc"
nc_in =  nc.Dataset(input_file)
transcom_regions_data = nc_in["transcom_regions"][:]
grid_cell_area =  nc_in["grid_cell_area"][:]
grid_cell_area_norm = grid_cell_area/np.max(grid_cell_area)

def generate_transcom_namemapping(transcom_names):
    result_mapping = {}
    for i in range(12):
        tt = list(nc_in["transcom_names"][:].data[i])
        tt2 = b"".join(tt).decode('utf-8')
        result_mapping[i+1] = tt2.strip()
    return result_mapping

transcom_names = nc_in["transcom_names"]
transcom_names_mapping = generate_transcom_namemapping(transcom_names)


#%%
# output_file = "regional_result.nc"
# ncout = Dataset(output_file, 'w', 'NETCDF4')

# ncout.createDimension('longitude', 360)
# ncout.createDimension('latitude', 180)
# ncout.createDimension('time', None)


# lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
# latvar = ncout.createVariable('latitude', 'float32', ('latitude'))
# timevar = ncout.createVariable('time', 'int32', ('time'))
# timevar.setncattr('units', "minutes since 2000-01-01 00:00:00")
# fluxvar = ncout.createVariable("f", 'float32',
#                             ('time', 'latitude', 'longitude'))

lonvar[:] = lon_vector_360
latvar[:] = lat_vector_180
fluxvar[:] = flux_transcom_regions
timevar[:]= np.asarray([0])
ncout.close()


#%%
# input_data = f_var_np[0,:,:]
# tt = reshape_4672_180360(input_data,  lat_vector_46, lon_vector_72, lat_vector_180, lon_vector_360)
# plt.imshow(tt)
# f_var_np_full
# carbontracker_data_full
for i in range(12):
    result = np.corrcoef(f_var_np_full[i,20:-20,:].flatten(), carbontracker_data_full[i,20:-20,:].flatten())[1,0]
    print(f"{result:.3f}")
    



# data = np.mean(f_var_np_full[:,:],axis = 0)
data1 = f_var_np_full[7,:,:]
flux_transcom_regions, result_mapping = average_on_transcom_regions(data1, transcom_regions_data)
output_file4 = "CT2019B.flux1x1.2018_inversion_inversion_transcom.nc"
write_global_data(output_file4, flux_transcom_regions, t_var_np, lon_vector_360, lat_vector_180)


# np.mean([:,:],axis = 0)
data2 = carbontracker_data_full[7,:,:]
flux_transcom_regions, result_mapping = average_on_transcom_regions(data2, transcom_regions_data)
output_file5 = "CT2019B.flux1x1.2018_inversion_ct_transcom.nc"
write_global_data(output_file5, flux_transcom_regions, t_var_np, lon_vector_360, lat_vector_180)




tt1 = average_on_transcom_regions(data1, transcom_regions_data)
tt2 = average_on_transcom_regions(data2, transcom_regions_data)


plt.plot(carbontracker_data_full[:,30:-30,:].mean(axis = (1,2)), label = "CarbonTracker")
plt.plot(f_var_np_full[:,30:-30,:].mean(axis = (1,2)), label = "autoInverse")
plt.xlabel("month")
plt.ylabel("flux (mol.m-2s-1)")
plt.legend()

#%%
data = [0.436,
0.504,
0.420,
0.331,
0.341,
0.601,
0.643,
0.364,
0.379,
0.525,
0.540,
0.401]

plt.plot(data)
plt.xlabel("month")
plt.ylabel("correlation")
plt.ylim([0,1])
plt.legend()
