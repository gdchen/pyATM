import netCDF4 as nc
from netCDF4 import Dataset, num2date, date2num
import datetime
import numpy as np
import os


def get_variable_carbon_2d(foldername, datestr):
    file_name = "../data/nc_file/myfile_carbon.nc"
    df = nc.Dataset(file_name)
    longitude = df.variables["longitude"][:]
    latitude = df.variables["latitude"][:]

    longitude = np.asarray(df.variables["longitude"][:])
    latitude = np.asarray(df.variables["latitude"][:])
    print(f"longitude:{longitude}")
    print(f"latitude:{latitude}")
    u = np.transpose(np.asarray(df.variables["u"][:]), (0, 2, 1))
    v = np.transpose(np.asarray(df.variables["v"][:]), (0, 2, 1))
    c = np.transpose(np.asarray(df.variables["t"][:]), (0, 2, 1))/273.0

    return longitude, latitude, u, v, c



def get_variable_carbon_3d(folder_name, file_name, layers = 10):
    # file_name = "../data/nc_file/myfile_carbon.nc"
    folder_file_name = os.path.join(folder_name, file_name)

    df = nc.Dataset(folder_file_name)
    longitude = df.variables["longitude"][:]
    latitude = df.variables["latitude"][:]

    longitude = np.asarray(df.variables["longitude"][:])
    latitude = np.asarray(df.variables["latitude"][:])
    print(f"longitude:{longitude}")
    print(f"latitude:{latitude}")

    u = np.transpose(np.asarray(df.variables["u"][:]), (0, 2, 1))
    v = np.transpose(np.asarray(df.variables["v"][:]), (0, 2, 1))
    w = u/1000.0
    c = np.transpose(np.asarray(df.variables["t"][:]), (0, 2, 1))/273.0

    print(u.shape)
    u = np.tile(np.expand_dims(u,-1),[1,1,1,layers]) 
    u = u + np.random.randn(*u.shape).astype(np.float32)
    v = np.tile(np.expand_dims(v,-1),[1,1,1,layers]) + np.random.randn(*u.shape).astype(np.float32)
    w = np.tile(np.expand_dims(w,-1),[1,1,1,layers]) + 0.1*np.random.randn(*u.shape).astype(np.float32)
    c = np.tile(np.expand_dims(c,-1),[1,1,1,layers]) + 0.02*np.random.randn(*u.shape).astype(np.float32)
    
    print()
    return longitude, latitude, u, v, w, c

def write_2d_carbon_netcdf(data_, ref_nc_name, output_nc_name, time_string, plot_interval):

    df = nc.Dataset(ref_nc_name)

    original_nx, original_ny, original_nt = df.dimensions[
        "longitude"].size, df.dimensions["latitude"].size, df.dimensions["time"].size
    new_nx, new_ny = original_nx, original_ny

    ncout = Dataset(output_nc_name, 'w', 'NETCDF4')
    ncout.createDimension('longitude', new_nx)

    ncout.createDimension('latitude', new_ny)
    ncout.createDimension('time', None)

    lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
    latvar = ncout.createVariable('latitude', 'float32', ('latitude'))
    timevar = ncout.createVariable('time', 'int32', ('time'))

    lonvar.setncattr('units', df.variables["longitude"].units)
    latvar.setncattr('units', df.variables["latitude"].units)
    timevar.setncattr('units', df.variables["time"].units)

    total_len = len(time_string)

    lonvar[:] = df.variables["longitude"][:]

    latvar[:] = df.variables["latitude"][:]

    calendar = 'standard'
    timevar[:] = nc.date2num(time_string[
        0:total_len:plot_interval], units=df.variables["time"].units, calendar=calendar)


    f = ncout.createVariable("f", 'float32',
                             ('time', 'latitude', 'longitude'))
    c = ncout.createVariable("c", 'float32',
                             ('time', 'latitude', 'longitude'))
    u = ncout.createVariable("u", 'float32',
                             ('time', 'latitude', 'longitude'))
    v = ncout.createVariable("v", 'float32',
                             ('time', 'latitude', 'longitude'))
    

    f[:] = np.transpose(data_[0:total_len:plot_interval, 0, :, :], (0, 2, 1))
    c[:] = np.transpose(data_[0:total_len:plot_interval, 1, :, :], (0, 2, 1))
    u[:] = np.transpose(data_[0:total_len:plot_interval, 2, :, :], (0, 2, 1))
    v[:] = np.transpose(data_[0:total_len:plot_interval, 3, :, :], (0, 2, 1))

    ncout.close()
    return None





def write_carbon_netcdf_3d(data_, ref_nc_name, output_nc_name, time_string, plot_interval, layers, vector_z):

    df = nc.Dataset(ref_nc_name)

    original_nx, original_ny, original_nt = df.dimensions[
        "longitude"].size, df.dimensions["latitude"].size, df.dimensions["time"].size
    new_nx, new_ny = original_nx, original_ny

    ncout = Dataset(output_nc_name, 'w', 'NETCDF4')
    ncout.createDimension('longitude', new_nx)
    ncout.createDimension('latitude', new_ny)
    ncout.createDimension('height', layers)
    ncout.createDimension('time', None)

    lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
    latvar = ncout.createVariable('latitude', 'float32', ('latitude'))
    heightvar = ncout.createVariable('height', 'float32', ('height'))
    timevar = ncout.createVariable('time', 'int32', ('time'))

    lonvar.setncattr('units', df.variables["longitude"].units)
    latvar.setncattr('units', df.variables["latitude"].units)
    heightvar.setncattr('units', "")
    timevar.setncattr('units', df.variables["time"].units)

    total_len = len(time_string)

    lonvar[:] = df.variables["longitude"][:]
    latvar[:] = df.variables["latitude"][:]
    heightvar[:] = vector_z

    calendar = 'standard'
    timevar[:] = nc.date2num(time_string[
        0:total_len:plot_interval], units=df.variables["time"].units, calendar=calendar)


    f = ncout.createVariable("f", 'float32',
                             ('time', 'latitude', 'longitude','height'))
    c = ncout.createVariable("c", 'float32',
                             ('time', 'latitude', 'longitude','height')) 
    u = ncout.createVariable("u", 'float32',
                             ('time', 'latitude', 'longitude','height'))
    v = ncout.createVariable("v", 'float32',
                             ('time', 'latitude', 'longitude','height'))
    w = ncout.createVariable("w", 'float32',
                             ('time', 'latitude', 'longitude','height'))
    

    f[:] = np.transpose(data_[0:total_len:plot_interval, 0, :, :, :], (0, 2, 1,3))
    c[:] = np.transpose(data_[0:total_len:plot_interval, 1, :, :, :], (0, 2, 1,3))
    u[:] = np.transpose(data_[0:total_len:plot_interval, 2, :, :, :], (0, 2, 1,3))
    v[:] = np.transpose(data_[0:total_len:plot_interval, 3, :, :, :], (0, 2, 1,3))
    w[:] = np.transpose(data_[0:total_len:plot_interval, 4, :, :, :], (0, 2, 1,3))

    ncout.close()
    return None





# def write_carbon_netcdf_3d2(data_, ref_nc_name, output_nc_name, time_string, plot_interval, layers, vector_z):

#     df = nc.Dataset(ref_nc_name)

#     original_nx, original_ny, original_nt = df.dimensions[
#         "longitude"].size, df.dimensions["latitude"].size, df.dimensions["time"].size
#     new_nx, new_ny = original_nx, original_ny

#     ncout = Dataset(output_nc_name, 'w', 'NETCDF4')
#     ncout.createDimension('longitude', new_nx)
#     ncout.createDimension('latitude', new_ny)
#     ncout.createDimension('height', layers)
#     ncout.createDimension('time', None)

#     lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
#     latvar = ncout.createVariable('latitude', 'float32', ('latitude'))
#     heightvar = ncout.createVariable('height', 'float32', ('height'))
#     timevar = ncout.createVariable('time', 'int32', ('time'))

#     lonvar.setncattr('units', df.variables["longitude"].units)
#     latvar.setncattr('units', df.variables["latitude"].units)
#     heightvar.setncattr('units', "")
#     timevar.setncattr('units', df.variables["time"].units)

#     total_len = len(time_string)

#     lonvar[:] = df.variables["longitude"][:]
#     latvar[:] = df.variables["latitude"][:]
#     heightvar[:] = vector_z

#     calendar = 'standard'
#     timevar[:] = nc.date2num(time_string[
#         0:total_len:plot_interval], units=df.variables["time"].units, calendar=calendar)


#     f = ncout.createVariable("f", 'float32',
#                              ('time', 'latitude', 'longitude','height'))
#     c = ncout.createVariable("c", 'float32',
#                              ('time', 'latitude', 'longitude','height')) 
#     u = ncout.createVariable("u", 'float32',
#                              ('time', 'latitude', 'longitude','height'))
#     v = ncout.createVariable("v", 'float32',
#                              ('time', 'latitude', 'longitude','height'))
#     w = ncout.createVariable("w", 'float32',
#                              ('time', 'latitude', 'longitude','height'))
    

#     f[:] = np.transpose(data_[0:total_len:plot_interval, 0, :, :, :], (0, 2, 1,3))
#     c[:] = np.transpose(data_[0:total_len:plot_interval, 1, :, :, :], (0, 2, 1,3))
#     u[:] = np.transpose(data_[0:total_len:plot_interval, 2, :, :, :], (0, 2, 1,3))
#     v[:] = np.transpose(data_[0:total_len:plot_interval, 3, :, :, :], (0, 2, 1,3))
#     w[:] = np.transpose(data_[0:total_len:plot_interval, 4, :, :, :], (0, 2, 1,3))

#     ncout.close()
#     return None



#%%
"""
用 nc文件可视化 matrix， 
"""




def write_netcdf_single_variable_single_time_2d(data_, output_nc_name, longitude_vector, latitude_vector):
    """
    netcdf 写单时间单变量的文件
    """
    ncout = Dataset(output_nc_name, 'w', 'NETCDF4')
    ncout.createDimension('longitude', len(longitude_vector))
    ncout.createDimension('latitude', len(latitude_vector))

    lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
    latvar = ncout.createVariable('latitude', 'float32', ('latitude'))

    lonvar[:] = longitude_vector
    latvar[:] = latitude_vector

    matrix = ncout.createVariable("matrix", 'float32',
                             ('latitude', 'longitude'))

    matrix[:] = np.transpose(data_, (1,0))

    ncout.close()
    return None




def write_netcdf_single_variable_multi_time_2d(data_, time_vector,output_nc_name, longitude_vector, latitude_vector):
    """
    netcdf 写多时间变量的文件
    """
    ncout = Dataset(output_nc_name, 'w', 'NETCDF4')
    ncout.createDimension('longitude', len(longitude_vector))
    ncout.createDimension('latitude', len(latitude_vector))
    ncout.createDimension('time', None)

    lonvar = ncout.createVariable('longitude', 'float32', ('longitude'))
    latvar = ncout.createVariable('latitude', 'float32', ('latitude'))
    timevar = ncout.createVariable('time', 'int32', ('time'))

    lonvar[:] = longitude_vector
    latvar[:] = latitude_vector
    timevar[:] = time_vector

    matrix = ncout.createVariable("matrix", 'float32',
                             ('time','latitude', 'longitude'))

    matrix[:] = np.transpose(data_, (0,2,1))

    ncout.close()
    return None

