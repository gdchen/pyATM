
import numpy as np
from scipy.interpolate import interpn        




def initialize_points(long_block_number, lati_block_number, long_spacing, lati_spacing, particle_inblock):
    """
    初始化颗粒坐标
    long_block_number: 经度方向 划分block的个数
    lati_block_number：纬度方向 划分block的个数
    particle_inblock: 每个block个播撒的点的个数
    
    """
    x_coordinate_random = np.random.uniform(low=0.0, high=1.0, size=[long_block_number,lati_block_number, particle_inblock])
    y_coordinate_random = np.random.uniform(low=0.0, high=1.0, size=[long_block_number,lati_block_number, particle_inblock])
    x_coordinate =  np.empty(x_coordinate_random.shape)
    y_coordinate =  np.empty(y_coordinate_random.shape)
    for long_index in range(long_block_number):
        for lati_index in range(lati_block_number):
            x_coordinate[long_index, lati_index,:] = (x_coordinate_random[long_index, lati_index,:] + long_index)*long_spacing - 180 - 2.5
            y_coordinate[long_index, lati_index,:] = (y_coordinate_random[long_index, lati_index,:] + lati_index)*lati_spacing - 86 -  2.0
   
    return x_coordinate, y_coordinate



def initialize_points_singlegrid(long_index, lati_index, long_spacing, lati_spacing, particle_inblock):
    """
    初始化颗粒坐标, 单个坐标  long_index, lati_index
    long_index: 经度方向 划分block的个数
    lati_index: 划分block的个数
    particle_inblock: 每个block个播撒的点的个数
    
    """
    x_coordinate_random = np.random.uniform(low=0.0, high=1.0, size=[1,1, particle_inblock])
    y_coordinate_random = np.random.uniform(low=0.0, high=1.0, size=[1,1, particle_inblock])
    x_coordinate =  np.empty(x_coordinate_random.shape)
    y_coordinate =  np.empty(y_coordinate_random.shape)
    x_coordinate[0, 0,:] = (x_coordinate_random[0, 0,:] + long_index)*long_spacing - 180 - 2.5
    y_coordinate[0, 0,:] = (y_coordinate_random[0, 0,:] + lati_index)*lati_spacing - 86 -  2.0
   
    return x_coordinate, y_coordinate






def lagragian_padding_2d_x(x_grid, u_use, v_use):
    """
    在经度方向pad多个网格，防止插值有问题
    """
    x_use_pad = np.concatenate([-360 + x_grid[-3::], x_grid, 360+ x_grid[0:3]])
    u_use_pad = np.concatenate([ u_use[-3::], u_use,  u_use[0:3]])
    v_use_pad = np.concatenate([ v_use[-3::], v_use,  v_use[0:3]])
    return x_use_pad, u_use_pad, v_use_pad


def lagragian_padding_2d_y(y_grid, u_use, v_use):
    """
    在经度方向pad多个网格，防止插值有问题
    """
    half_x_lenght = u_use.shape[0]
    y_use_pad = np.concatenate([-180 - np.flip(y_grid[0:3]), y_grid, 180 - np.flip(y_grid[-3::])])
    
    u_pad_left =  np.flip(np.roll(u_use[:,0:3], half_x_lenght, axis = 0),axis =1)
    u_pad_right = np.flip(np.roll(u_use[:,-3::],half_x_lenght, axis = 0),axis =1)
    
    v_pad_left =  -1.0*np.flip(np.roll(v_use[:,0:3], half_x_lenght, axis = 0),axis =1)
    v_pad_right=  -1.0*np.flip(np.roll(v_use[:,-3::], half_x_lenght, axis = 0),axis =1)
    
    
    u_use_pad = np.concatenate([ u_pad_left, u_use,  u_pad_right], axis = 1)
    v_use_pad = np.concatenate([ v_pad_left, v_use,  v_pad_right], axis = 1)
    return y_use_pad, u_use_pad, v_use_pad


def interpolate_2d(x_ref, y_ref, value_ref, x_query, y_query):
    
    """
    经度，纬度，值，待差值的经度，待差值的纬度
    所有都是numpy 类型的 
    
    """
    # points_query_list = []
    # for x_val,y_val  in zip(x_query, y_query):
    #     points_query_list.append([x_val ,y_val])
    # points_query = np.asarray(points_query_list)
        
    points_ref = (x_ref, y_ref)
    points_query = (x_query, y_query)
    
    value_query = interpn(points_ref, value_ref, points_query, method = "linear")
    # value_query = interpn(points_ref, value_ref, points_query, method = "splinef2d")
    
    
    return value_query


def interpolate_3d(x_ref, y_ref, z_ref, value_ref, x_query, y_query, z_query):
    
    """
    经度，纬度，值，待差值的经度，待差值的纬度
    所有都是numpy 类型的 
    
    """
    # points_query_list = []
    # for x_val, y_val, z_val in zip(x_query, y_query, z_query):
    #     points_query_list.append([x_val ,y_val, z_val])
    # points_query = np.asarray(points_query_list)
        
    points_ref = (x_ref, y_ref, z_ref)
    points_query = (x_query, y_query, z_query)
    
    # value_query = interpn(points_ref, value_ref, points_query, method = "linear")
    value_query = interpn(points_ref, value_ref, points_query, method = "linear")
    
    
    return value_query





def get_map_factor(long_vector):
    map_factor = 1.0 / (np.cos(2 * np.pi / 360.0 *long_vector))
    return map_factor

def uv2_radius_velocity(u_query, v_query, y_query):
    meter_per_radius = (6378)*1000 * 2.0*np.pi / 360.0
    
    map_factor = get_map_factor(y_query)
    map_factor[map_factor > 5.0] = 5.0
    
    u_long = u_query/meter_per_radius*3600*24 * map_factor
    v_lati = v_query/meter_per_radius*3600*24
    return u_long,v_lati


def position_update(x_point , y_point, u_long, v_lati, day_number = 1.0 ):
    """
    由坐标 + 速度 ， 更新坐标场。
    更新后，需要对于移出边界的坐标进行更新
    """
    x_point_new = x_point + u_long*day_number
    y_point_new = y_point + v_lati*day_number

    
    y_mod_index = y_point_new > 90
    y_point_new[y_mod_index] = 180 - y_point_new[y_mod_index]
    x_point_new[y_mod_index] = x_point_new[y_mod_index] + 180
    v_lati[y_mod_index] = -v_lati[y_mod_index]
                                
    
    y_mod_index = y_point_new < -90
    y_point_new[y_mod_index] = -180 - y_point_new[y_mod_index]
    x_point_new[y_mod_index] = x_point_new[y_mod_index] + 180
    v_lati[y_mod_index] = -v_lati[y_mod_index]
    
    x_point_new = (x_point_new + 182.5)%360 - 182.5
    
    return x_point_new, y_point_new





"""
需要增加拼接模块, 使得边界不溢出。
"""

def interpolate_particle_velocity_2d(x_point, y_point, u_use, v_use, x_grid_1d, y_grid_1d):
    """
    x_grid_1d, y_grid_1d: 一维的网格序列
    u,v 2维场
    x_point, y_point 带差值的一维网格坐标
    """
    
    y_grid_1d_pad, u_use_pad_y, v_use_pad_y = lagragian_padding_2d_y(y_grid_1d, u_use, v_use) 
    
    x_grid_1d_pad, u_use_pad_yx, v_use_pad_yx = lagragian_padding_2d_x(x_grid_1d, u_use_pad_y, v_use_pad_y ) 
    # print(y_grid_1d_pad, x_grid_1d_pad, u_use_pad_yx.shape , v_use_pad_yx.shape, len(y_grid_1d_pad), len(x_grid_1d_pad))
    
    # import matplotlib.pyplot as plt
    # plt.imshow(u_use_pad_yx)
    
    # plt.imshow(v_use_pad_yx)
    # step1:差值速度
    u_query = interpolate_2d(x_grid_1d_pad, y_grid_1d_pad, u_use_pad_yx, x_point, y_point)
    v_query = interpolate_2d(x_grid_1d_pad, y_grid_1d_pad, v_use_pad_yx, x_point, y_point)

    # step2: 速度转到经纬度速度
    u_long, v_lati = uv2_radius_velocity(u_query, v_query, y_point)
    return u_long, v_lati
    

def particle_forward_order1(x_point, y_point, u_use, v_use, x_grid_1d, y_grid_1d,day_number = 1.0):
    """
    一阶精度的颗粒推进，
    x_point, y_point 之前的坐标
    x_point_new, y_point_new 更新后的坐标
    """
    u_long, v_lati = interpolate_particle_velocity_2d(x_point, y_point, u_use, v_use, x_grid_1d, y_grid_1d)
    # step3: 更新坐标
    
    x_point_new, y_point_new  = position_update(x_point, y_point, 
                                       u_long, v_lati,
                                       day_number = day_number)
    return x_point_new, y_point_new


def particle_forward_order2(x_point, y_point, u_use, v_use, x_grid_1d, y_grid_1d,day_number = 1.0):
    """
    一阶精度的颗粒推进，
    x_point, y_point 之前的坐标
    x_point_new, y_point_new 更新后的坐标
    
    其中 u_long1, v_lati1 都是经纬度速度
    """
    u_long1, v_lati1 = interpolate_particle_velocity_2d(x_point, y_point, u_use, v_use, x_grid_1d, y_grid_1d)
    
    x_point_new, y_point_new  = position_update(x_point, y_point, 
                                       u_long1, v_lati1,
                                       day_number = day_number)
    
    u_long2, v_lati2 = interpolate_particle_velocity_2d(x_point_new, y_point_new, u_use, v_use, x_grid_1d, y_grid_1d)
    
    u_long_mean = (u_long1 + u_long2)/2.0
    v_long_mean = (v_lati1 + v_lati2)/2.0
    
    # print(torch.sum(torch.abs(u_long_mean) > 40)/ u_long_mean.shape.numel())
    # print(torch.sum(torch.abs(u_long_mean) > 40)/ u_long_mean.shape.numel())
    
    x_point_new, y_point_new  = position_update(x_point, y_point, 
                                       u_long_mean, v_long_mean,
                                       day_number = day_number)
    
    return x_point_new,y_point_new
    



def generate_loc_matrix_target(x_point, y_point,long_spacing, lati_spacing):
    
    """
    针对单个时刻， x_point， x_point 的统计
    """
    
    long_block_number = int(360/long_spacing)
    lati_block_number = int(176/lati_spacing)
    
    
    loc_matrix = np.zeros([long_block_number, lati_block_number])
    
    """
    x_coordinate[long_index, lati_index,:] = (x_coordinate_random[long_index, lati_index,:] + long_index)*long_spacing - 180 - 2.5
    y_coordinate[long_index, lati_index,:] = (y_coordinate_random[long_index, lati_index,:] + lati_index)*lati_spacing - 86 -  2.0
       
    """
    out_domain_number = 0
    for particle_index in range(len(x_point)):
        x_coordinate = x_point[particle_index]
        y_coordinate = y_point[particle_index]
        long_index = int(np.floor((x_coordinate + (180 + 2.5))//long_spacing))
        lati_index = int(np.floor((y_coordinate + 86 +  2.0)//lati_spacing))
        
        if((lati_index >=lati_block_number ) or (lati_index < 0)):
            out_domain_number = out_domain_number + 1
        else:
            loc_matrix[long_index, lati_index] = loc_matrix[long_index, lati_index] + 1.0
    print(f"out domain percentage:{1.0*out_domain_number/len(x_point)} ")
    return loc_matrix





def generate_loc_matrix_source_target(x_point, y_point,long_spacing, lati_spacing):
    
    long_block_number = int(360/long_spacing)
    lati_block_number = int(176/lati_spacing)

    loc_matrix = np.zeros([long_block_number, lati_block_number,long_block_number, lati_block_number])
    x_point_reshape = x_point.reshape(long_block_number, lati_block_number, -1 ) 
    y_point_reshape = y_point.reshape(long_block_number, lati_block_number, -1 )

    particle_inblock = x_point_reshape.shape[2]
    
    out_domain_number = 0
    for source_long_index in range(long_block_number):
        for source_lati_index in range(lati_block_number):
            for particle_index in range(particle_inblock):
                x_coordinate = x_point_reshape[source_long_index, source_lati_index, particle_index]
                y_coordinate = y_point_reshape[source_long_index, source_lati_index, particle_index]
                long_index = int(np.floor((x_coordinate + (180 + 2.5))//long_spacing))
                lati_index = int(np.floor((y_coordinate + 86 +  2.0)//lati_spacing))
                
                if((lati_index >=lati_block_number ) or (lati_index < 0)):
                    out_domain_number = out_domain_number + 1
                else:
                    loc_matrix[source_long_index, source_lati_index,long_index,lati_index] = \
                        loc_matrix[source_long_index, source_lati_index,long_index,lati_index] + 1.0
    
    print(f"out domain percentage:{1.0*out_domain_number/len(x_point)} ")
    return loc_matrix




#%%
def generate_forward_matrix_from_locmaxtrix(loc_matrix_source_target):
    time_vector_list = []
    flag = 0
    index = 0
    for long_index in np.arange(0,72,1):
        for lati_index in np.arange(0,44,1):
            time_vector_list.append( int(str(1) + str(long_index).zfill(3) + str(lati_index).zfill(3)))
            index = index + 1
            write_matrix = loc_matrix_source_target[long_index,lati_index,:,:]
            write_matrix = write_matrix[np.newaxis,:,:]
            if(np.sum(write_matrix) > 1.0):
                write_matrix = 1.0*write_matrix/np.sum(write_matrix)
            
            if(flag == 0):
                write_matrix_batch = write_matrix
                flag = 1
            else:
                write_matrix_batch = np.concatenate([write_matrix_batch, write_matrix], axis = 0)
    return write_matrix_batch, time_vector_list




def generate_jacobian_matrix_from_locmaxtrix(loc_matrix_source_target):
    query_mapping = {}
    time_vector_list = []
    flag = 0
    index = 0
    
    for long_index in np.arange(0,72,1):
        for lati_index in np.arange(0,44,1):
            
            time_vector_list.append( int(str(1) + str(long_index).zfill(3) + str(lati_index).zfill(3)))
            query_mapping[index ] = f"{long_index}_{lati_index}" 
            index = index + 1
            
            write_matrix = loc_matrix_source_target[:,:,long_index,lati_index]
            write_matrix = write_matrix[np.newaxis,:,:]
            
            if(np.sum(write_matrix) > 1.0):
                write_matrix = 1.0*write_matrix/np.sum(write_matrix)
            
            if(flag == 0):
                write_matrix_batch = write_matrix
                flag = 1
            else:
                write_matrix_batch = np.concatenate([write_matrix_batch, write_matrix], axis = 0)
    
    return write_matrix_batch, time_vector_list





def lagragian_save(file_name, x_point_all, y_point_all,loc_matrix ):
    """
    将产生的轨迹保存， x_point_all， y_point_all， 以及 源汇矩阵 loc_matrix
    
    """
    with open(file_name, 'wb') as f:
        np.save(f, x_point_all)
        np.save(f, y_point_all)
        np.save(f, loc_matrix)
    
#%%
def lagragian_load(file_name):
    """
    加载轨迹矩阵
    """
    with open(file_name, 'rb') as f:
        x_point_all = np.load(f)
        y_point_all = np.load(f)
        loc_matrix = np.load(f)
    return x_point_all, y_point_all,loc_matrix 



def plot_lagrangian_snapshot(x_point, y_point, filename):
    """
    根据lagrangian 散点绘制图
    """
    import matplotlib.pyplot as plt
    plt.figure(0)
    plt.scatter(x_point, y_point,s = 2, alpha = 0.1)
    
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.cla()



def plot_lagrangian_snapshot_basemap(x_point, y_point, filename, dot_size = 1):

    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    fig = plt.figure(2)
    fig.set_size_inches(8, 6.5)
    
    m = Basemap(projection='merc', \
                llcrnrlat=-70, urcrnrlat=70, \
                llcrnrlon=-180, urcrnrlon=180, \
                lat_ts=20, \
                resolution='i' )
    """        
    m = Basemap(llcrnrlon=-100.,llcrnrlat=0.,urcrnrlon=-30.,urcrnrlat=57.,
                 projection='lcc',lat_1=40.,lat_2=60.,lon_0=-60.,
    #             resolution ='l',area_thresh=1000.)
    """
                            
    x,y = m(x_point, 
            y_point)
    m.scatter(x,y,
              s = dot_size, alpha = 0.1) 
    
    m.drawcoastlines(color='white', linewidth=0.2 )  # add coastlines
    m.drawcoastlines()
    m.drawcountries()
    
    
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.cla()





def plot_lagrangian_trace_basemap(x_point_all, y_point_all, filename, sample_number):
    
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    fig = plt.figure(2)
    fig.set_size_inches(8, 6.5)
    
    m = Basemap(projection='merc', \
                llcrnrlat=-70, urcrnrlat=70, \
                llcrnrlon=-180, urcrnrlon=180, \
                lat_ts=20, \
                resolution='i' )
    """        
    m = Basemap(llcrnrlon=-100.,llcrnrlat=0.,urcrnrlon=-30.,urcrnrlat=57.,
                 projection='lcc',lat_1=40.,lat_2=60.,lon_0=-60.,
    #             resolution ='l',area_thresh=1000.)
    """
    
    for i in range(sample_number):
        if(True):
            time_index = np.random.choice(np.arange(x_point_all.shape[1]))
        
        """
        # if(True):
        #     time_index = 33 + i*44 
        
        # time_index = i
        # x_grid = i//44
        # y_grid = i%44
        
        # if(x_grid < 21) & (x_grid > 16) & (y_grid < 32) & (y_grid > 26):
        # time_index = i
        """
    
        x, y = m(x_point_all[:,time_index], y_point_all[:,time_index])
        m.scatter(x,y, s = 0.5, alpha = 0.3) 
    
    m.drawcoastlines(color='white', linewidth=0.2 )  # add coastlines
    m.drawcoastlines()
    m.drawcountries()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.cla()
    
    

def plot_grid_lagrangian_trace(x_point_all, y_point_all, long_index, lati_index, 
                               file_name, grid_shape, sample_number):
    x_point_all_reshape = x_point_all.reshape(grid_shape)
    y_point_all_reshape = y_point_all.reshape(grid_shape)
    
    [time_len,long_block_number,lati_block_number, particle_inblock ] = grid_shape 
    
    in_block_index = np.random.choice(np.arange(particle_inblock),sample_number)
    
    plot_lagrangian_snapshot_basemap(x_point_all_reshape[:, long_index, lati_index,:][:,in_block_index].flatten(), 
                             y_point_all_reshape[:, long_index,  lati_index,:][:,in_block_index].flatten(), 
                             file_name, dot_size = 0.2)
    return 0
