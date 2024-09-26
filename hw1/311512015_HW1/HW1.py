import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
import argparse

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N, object):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')
    plt.savefig(object + '_Normal.png')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D, object):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.savefig(object + '_Depth.png')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    ######### outlier removal ##########
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    # cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    inlier_cloud = pcd.select_by_index(ind)
    o3d.io.write_point_cloud(filepath, inlier_cloud,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    image_row , image_col = image.shape
    return image


if __name__ == '__main__':

    # input object
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str)
    args = parser.parse_args()
    object = args.object
    path = "./test/" + object + "/"

    IMG1 = cv2.imread(path + 'pic1.bmp')
    IMG2 = cv2.imread(path + 'pic2.bmp')
    IMG3 = cv2.imread(path + 'pic3.bmp')
    IMG4 = cv2.imread(path + 'pic4.bmp')
    IMG5 = cv2.imread(path + 'pic5.bmp')
    IMG6 = cv2.imread(path + 'pic6.bmp')

    # read poses of light source, calculate their vectors
    S_lst = []
    with open(path + 'LightSource.txt', 'r', encoding='utf8') as shadow_file:
        for x in shadow_file.readlines():
            temp = x[7:-2].split(',')
            temp = [float(i) for i in temp]
            S_lst.append(temp)
    S_lst = np.array(S_lst)
    print("lightsource vector:\n",S_lst)

    # get lightsource vector (m, 3)
    lst_light = []
    for i in S_lst:
        lst_light.append(i/np.linalg.norm(i))
    S_lst = np.array(lst_light)
    print("lightsource unit vector :\n", S_lst)

    # size of image
    albedo_lst = np.zeros(IMG1.shape)
    N_lst = np.zeros(IMG1.shape)

    global image_row 
    global image_col
    image_row , image_col, _ = IMG1.shape

    # each pixel
    for x in range(IMG1.shape[0]) :
        for y in range(IMG1.shape[1]) :
            I = np.array([
                IMG1[x][y][0],
                IMG2[x][y][0],
                IMG3[x][y][0],
                IMG4[x][y][0],
                IMG5[x][y][0],
                IMG6[x][y][0]
            ])

            # SVD
            u,s,v=np.linalg.svd(S_lst, full_matrices = False)
            S_lst_inv=np.dot(v.transpose(),np.dot(np.diag(s**-1),u.transpose()))
            N = np.dot(S_lst_inv, I)
            G = N.T
            
            # Normal n = N/|N|
            G_gray = G
            Gnorm = np.linalg.norm(G_gray)
            # make a mask
            if Gnorm==0:
                continue
            N_lst[x][y] = G_gray/Gnorm
            
            # Albedo |N|
            rho = np.linalg.norm(G)
            albedo_lst[x][y] = rho

    # change 0~255
    normal = N_lst
    N_lst = ((N_lst[:, :, [2, 1, 0]]*0.5 + 0.5)*255).astype(np.uint8)
    albedo_lst = (albedo_lst/np.max(albedo_lst)*255).astype(np.uint8)
    normal_visualization(normal, object)

    ############################### reconstruct ##################################
    mask = albedo_lst[:,:,0]
    objectPixels = np.asarray(np.where((mask != 0)))
    S = objectPixels.shape[1] # number of pixels having depth
    print('number of pixel having depth: ', S)

    M = np.zeros((2*S, S))
    V = np.zeros((2*S, 1))

    # z indices that have depths
    z_index = np.zeros((IMG1.shape[0], IMG1.shape[1]))
    for i in range(S):
        z_index[objectPixels[0][i]][objectPixels[1][i]] = i
    z_index = z_index.astype(int)

    for i in range(S):
        row = objectPixels[0][i] # y
        col = objectPixels[1][i] # x
        nx = normal[row][col][0]
        ny = normal[row][col][1]
        nz = normal[row][col][2]

        if (z_index[row, col+1] > 0) and (z_index[row+1, col] > 0): # x-dir ok & y-dir ok
            # y-dir
            M[2*i, z_index[row+1, col]] = -1
            M[2*i, z_index[row, col]] = 1
            V[2*i] = -ny/nz

            # x-dir
            M[2*i+1, z_index[row][col+1]] = 1
            M[2*i+1, z_index[row][col]] = -1
            V[2*i+1] = -nx/nz
        elif (z_index[row][col+1] > 0): # x-dir ok & y-dir not ok
            # y-dir
            if z_index[row-1, col] > 0:
                M[2*i, z_index[row-1][col]] = 1
                M[2*i, z_index[row][col]] = -1
                V[2*i] = -ny/nz

            # x-dir
            M[2*i+1, z_index[row][col+1]] = 1
            M[2*i+1, z_index[row][col]] = -1
            V[2*i+1] = -nx/nz
        elif (z_index[row+1][col] > 0): # x-dir not ok & y-dir ok
            # y-dir
            M[2*i, z_index[row+1][col]] = -1
            M[2*i, z_index[row][col]] = 1
            V[2*i] = -ny/nz

            # x-dir
            if z_index[row, col-1] > 0:
                M[2*i+1, z_index[row][col-1]] = -1
                M[2*i+1, z_index[row][col]] = 1
                V[2*i+1] = -nx/nz
        else: # x-dir not ok & y-dir not ok
            # y-dir
            if z_index[row-1, col] > 0:
                M[2*i, z_index[row-1][col]] = 1
                M[2*i, z_index[row][col]] = -1
                V[2*i] = -ny/nz
            
            # x-dir
            if z_index[row, col-1] > 0:
                M[2*i+1, z_index[row][col-1]] = -1
                M[2*i+1, z_index[row][col]] = 1
                V[2*i+1] = -nx/nz

    print('M shape: ', M.shape)
    print('V shape: ', V.shape)

    ###### Mz=V ######
    Ms = lil_matrix(M)
    MtM = Ms.T @ Ms
    Mtv = Ms.T @ V
    z = spsolve(MtM, Mtv)

    # make image size
    z_final = np.zeros(((IMG1.shape[0], IMG1.shape[1])))
    for i in range(S):
        z_final[objectPixels[0][i]][objectPixels[1][i]] = z[i]

    depth_visualization(z_final, object)
    plt.show()

    save_ply(z_final, object + ".ply")
    show_ply(object + ".ply")