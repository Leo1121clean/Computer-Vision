import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.neighbors import KDTree

plt.rcParams['figure.figsize'] = [8, 6]

# Read image and convert them to gray!!
def read_image(path):
    img = cv2.imread(path)
    img_gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb

def SIFT(img):
    siftDetector = cv2.SIFT_create() # limit 1000 points

    #kp表示輸入的關鍵點，dst表示輸出的sift特徵向量(descripter)，通常是128維的
    kp, des = siftDetector.detectAndCompute(img, None)

    return kp, des

def plot_sift(gray, rgb, kp):
    tmp = rgb.copy()
    img = cv2.drawKeypoints(gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

# kp2 match kp1
def matcher(kp1, des1, kp2, des2, threshold):

    # Use kdtree to search 128-D space of descripter
    tree = KDTree(des2, leaf_size = 30)
    matches = []
    for i, x in enumerate(des1):
        distances, indices = tree.query([x], k = 2)
        matches.append([distances[0], indices[0][0], i])

    # Apply ratio test
    good_index = []
    for m in matches:
        if m[0][0] < threshold * m[0][1]:
            good_index.append([m[1], m[2]])

    #key point pair
    matches = []
    for pair in good_index:
        matches.append(list(kp1[pair[1]].pt + kp2[pair[0]].pt))

    matches = np.array(matches)
    return matches


def plot_matches(matches, total_img):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)

    plt.show()


def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p2[0], p2[1], p2[2], -p1[1]*p2[0], -p1[1]*p2[1], -p1[1]*p2[2]]
        row2 = [p2[0], p2[1], p2[2], 0, 0, 0, -p1[0]*p2[0], -p1[0]*p2[1], -p1[0]*p2[2]]
        # row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p1[0]*p2[1], -p1[1]*p2[1], -p1[2]*p2[1]]
        # row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p1[0]*p2[0], -p1[1]*p2[0], -p1[2]*p2[0]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1

    return H

def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)

def get_error(points, H):
    num_points = len(points)
    # all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    # all_p2 = points[:, 2:4]
    all_p2 = np.concatenate((points[:, 2:4], np.ones((num_points, 1))), axis=1)
    all_p1 = points[:, 0:2]
    estimate_p1 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p2[i])
        estimate_p1[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p1 - estimate_p1 , axis=1) ** 2

    return errors

def ransac(matches, threshold, iters):
    num_best_inliers = 0
    
    for i in range(iters):
        points = random_point(matches)
        H = homography(points) # img2 to img1
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H



def stitch_img(left, right, H):

    print("stiching image ...")

    # Convert to double and normalize. Avoid noise.
    left = cv2.normalize(left.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    # Convert to double and normalize.
    right = cv2.normalize(right.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   

    # left image
    # height_l, width_l, channel_l = left.shape
    # corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    # corners_new = [np.dot(H, corner) for corner in corners]
    # corners_new = np.array(corners_new).T 
    # x_news = corners_new[0] / corners_new[2]
    # y_news = corners_new[1] / corners_new[2]
    # y_min = min(y_news)
    # x_min = min(x_news)

    # right image
    height_l, width_l, channel_l = left.shape
    height_r, width_r, channel_r = right.shape
    corners = [[0, 0, 1], [width_r, 0, 1], [width_r, height_r, 1], [0, height_r, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
        
    y_max = max(y_news)
    x_max = max(x_news)
    # y_min = min(y_news)
    # x_min = min(x_news)

    # y_max = max(y_max, height_l)
    # x_max = max(x_max, width_l)
    # y_min = min(y_min, 0)
    # x_min = min(x_min, 0)

    # translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    # H = np.dot(translation_mat, H)

    # Get height, width
    # height_l, width_l, channel_l = left.shape
    # height_new = int(round(abs(y_max) + height_l))
    # width_new = int(round(abs(x_max) + width_l))
    height_new = round(abs(y_max))
    width_new = round(abs(x_max))
    # height_new = round(abs(y_max) - y_min)
    # width_new = round(abs(x_max) - x_min)
    size = (width_new, height_new)

    # warped image
    # warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)
    # warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)
    warped_l = cv2.warpPerspective(src=left, M=np.identity(3), dsize=size)
    warped_r = cv2.warpPerspective(src=right, M=H, dsize=size)
    # warped_l = cv2.warpPerspective(src=left, M=translation_mat, dsize=size)
    # warped_r = cv2.warpPerspective(src=right, M=H, dsize=size)
     
    black = np.zeros(3)  # Black pixel.


    print("warp shape: ", warped_r.shape)
    # Stitching procedure, store results in warped_l.
    beta = 0.15
    for i in range(warped_r.shape[0]):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]
       
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black): 
                # warped_l[i, j, :] = pixel_l * 0.5 + pixel_r * 0.5
                alpha = np.linalg.norm(pixel_l - pixel_r)
                if(np.linalg.norm(pixel_l) >= np.linalg.norm(pixel_r)):
                    if (alpha > beta):
                        warped_l[i, j, :] = pixel_l * 0.98 + pixel_r*0.02
                    else:
                        warped_l[i, j, :] = pixel_l*0.5 + pixel_r*0.5
                else:
                    if (alpha > beta):
                        warped_l[i, j, :] = pixel_l * 0.02 + pixel_r*0.98
                    else:
                        warped_l[i, j, :] = pixel_l*0.5 + pixel_r*0.5
                # if(np.linalg.norm(pixel_l) >= np.linalg.norm(pixel_r)):
                #     if (alpha > beta):
                #         warped_l[i, j, :] = pixel_l * 0.98 + pixel_r*0.02
                #     else:
                #         warped_l[i, j, :] = pixel_l * (1 - alpha) + pixel_r * alpha
                # else:
                #     if (alpha > beta):
                #         warped_l[i, j, :] = pixel_l * 0.02 + pixel_r*0.98
                #     else:
                #         warped_l[i, j, :] = pixel_l * alpha + pixel_r * (1 - alpha)
            else:
                pass
    
    stitch_image = warped_l
    # stitch_image = cv2.GaussianBlur(stitch_image, (11, 11), 0)
    return stitch_image



if __name__=="__main__":
    
    # # baseline
    img1_gray, img1_origin, img1_rgb = read_image('baseline/m1.jpg')
    img2_gray, img2_origin, img2_rgb = read_image('baseline/m2.jpg')
    img3_gray, img3_origin, img3_rgb = read_image('baseline/m3.jpg')
    img4_gray, img4_origin, img4_rgb = read_image('baseline/m4.jpg')
    img5_gray, img5_origin, img5_rgb = read_image('baseline/m5.jpg')
    img6_gray, img6_origin, img6_rgb = read_image('baseline/m6.jpg')

    # bonus
    # img1_gray, img1_origin, img1_rgb = read_image('bonus/m1.jpg')
    # img2_gray, img2_origin, img2_rgb = read_image('bonus/m2.jpg')
    # img3_gray, img3_origin, img3_rgb = read_image('bonus/m3.jpg')
    # img4_gray, img4_origin, img4_rgb = read_image('bonus/m4.jpg')
    
    # SIFT only can use gray
    kp_img1, des_img1 = SIFT(img1_gray)
    kp_img2, des_img2 = SIFT(img2_gray)
    kp_img3, des_img3 = SIFT(img3_gray)
    kp_img4, des_img4 = SIFT(img4_gray)
    kp_img5, des_img5 = SIFT(img5_gray)
    kp_img6, des_img6 = SIFT(img6_gray)

    # kp_left_img = plot_sift(img1_gray, img1_rgb, kp_left)
    # kp_right_img = plot_sift(img2_gray, img2_rgb, kp_right)
    # total_kp = np.concatenate((kp_left_img, kp_right_img), axis=1)
    # plt.imshow(total_kp)
    
    # for baseline: 0.6
    # for bonus: 0.4
    matches = matcher(kp_img1, des_img1, kp_img2, des_img2, 0.6)
    matches2 = matcher(kp_img2, des_img2, kp_img3, des_img3, 0.6)
    matches3 = matcher(kp_img3, des_img3, kp_img4, des_img4, 0.6)
    matches4 = matcher(kp_img4, des_img4, kp_img5, des_img5, 0.6)
    matches5 = matcher(kp_img5, des_img5, kp_img6, des_img6, 0.6)
    
    # total_img = np.concatenate((img1_rgb, img2_rgb), axis=1)
    # plot_matches(matches, total_img) # Good mathces
    
    inliers, H = ransac(matches, 0.5, 1000)
    inliers2, H2 = ransac(matches2, 0.5, 1000)
    inliers3, H3 = ransac(matches3, 0.5, 1000)
    inliers4, H4 = ransac(matches4, 0.5, 1000)
    inliers5, H5 = ransac(matches5, 0.5, 1000)
    # plot_matches(inliers, total_img) # show inliers matches
    
    merge_img1 = stitch_img(img1_rgb, img2_rgb, H)
    merge_img2 = stitch_img(merge_img1, img3_rgb, H@H2)
    merge_img3 = stitch_img(merge_img2, img4_rgb, H@H2@H3)
    merge_img4 = stitch_img(merge_img3, img5_rgb, H@H2@H3@H4)
    merge_img5 = stitch_img(merge_img4, img6_rgb, H@H2@H3@H4@H5)
    # merge_img5 = stitch_img(img5_rgb, img6_rgb, H5)


    # plt.imshow(stitch_img(img1_rgb, img2_rgb, H))
    plt.imshow(merge_img5)
    # plt.imshow(merge_img3)
    plt.show()