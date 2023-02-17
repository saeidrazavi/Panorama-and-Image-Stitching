from keyword import iskeyword
from logging import critical
from operator import ge
from sys import warnoptions
from turtle import width
from winreg import KEY_WOW64_32KEY
import numpy as np
import cv2
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
import time


# ------------------------------
# -------- functions -----------
# ------------------------------
def laplacian_pyramid(src1, src2, mask):

    # generate Gaussian pyramid for mask
    src1 = src1.astype(np.float32)
    src2 = src2.astype(np.float32)
    mask = (mask/255).astype(np.float32)

    M = mask.copy()
    gpm = [M]
    for i in range(9):
        M = cv2.pyrDown(M).astype(np.float32)
        gpm.append(M)

    # generate Gaussian pyramid for A
    G = src1.copy()
    gpA = [G]
    for i in range(9):
        G = cv2.pyrDown(G).astype(np.float32)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = src2.copy()
    gpB = [G]
    for i in range(9):
        G = cv2.pyrDown(G).astype(np.float32)
        gpB.append(G)
    # generate Laplacian Pyramid for A
    lpA = [gpA[8]]
    for i in range(8, 0, -1):
        GE = cv2.pyrUp(gpA[i]).astype(np.float32)
        L = (gpA[i-1] - GE).astype(np.float32)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[8]]
    for i in range(8, 0, -1):
        GE = cv2.pyrUp(gpB[i]).astype(np.float32)
        L = (gpB[i-1] - GE).astype(np.float32)
        lpB.append(L)
# Now add left and right halves of images in each level
    LS = []
    i = 0
    for la, lb in zip(lpA, lpB):
        i += 1
        ls = ((la[:, :]*(1-gpm[9-i]) + lb[:, :]*gpm[9-i])).astype(np.float32)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]

    for i in range(1, 9):
        ls_ = cv2.pyrUp(ls_)
        ls_ = (ls_+LS[i]).astype(np.float32)
        ls_[ls_ > 255] = 255
        ls_[ls_ < 0] = 0

    return ls_
# ---------------------------
# ---------------------------
# define a function to give a line between two specific points (p1,p2)


def line(p1, p2):

    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

# -------------------------
# define a function to find intersection points between two specific lines (l1,l2)


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False

# ---------------------------------
# find minimum error


def minimum_eror(frame1, frame2, boundaries, type, i):

    gray_frame1 = cv2.cvtColor(frame1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_frame2 = cv2.cvtColor(frame2.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    if(type == 'vertical'):
        x_min = int(boundaries[0][1])
        x_max = int(boundaries[1][1])
        y_min = min(int(boundaries[0][0]), int(boundaries[1][0]))
        y_max = max(int(boundaries[0][0]), int(boundaries[1][0]))

    if(type == 'horizental_up'):
        x_min = max(int(boundaries[0][1]), int(boundaries[3][1]))
        x_max = min(int(boundaries[0][1]), int(boundaries[3][1]))
        y_min = int(boundaries[0][0])
        y_max = int(boundaries[3][0])

    if(type == 'horizental_down'):
        x_min = min(int(boundaries[1][1]), int(boundaries[2][1]))
        x_max = max(int(boundaries[1][1]), int(boundaries[2][1]))
        y_min = int(boundaries[1][0])
        y_max = int(boundaries[2][0])

    if(type == 'vertical'):
        intersection_area_in_frame1 = gray_frame1[x_min:x_max,
                                                  y_min:y_max+1500]
        intersection_area_in_frame2 = gray_frame2[x_min:x_max,
                                                  y_min:y_max+1500]
    if(type == 'horizental_up'):
        intersection_area_in_frame1 = gray_frame1[x_min:x_min +
                                                  400+50*i, y_min:y_max]
        intersection_area_in_frame2 = gray_frame2[x_min:x_min +
                                                  400+50*i, y_min:y_max]
    if(type == 'horizental_down'):
        intersection_area_in_frame1 = gray_frame1[x_min -
                                                  100:x_max, y_min:y_max]
        intersection_area_in_frame2 = gray_frame2[x_min -
                                                  100:x_max, y_min:y_max]
    common_area = np.abs(intersection_area_in_frame1 -
                         intersection_area_in_frame2)

    energy_function = np.zeros_like(common_area).astype(np.uint64)
    index_matrix = np.zeros_like(common_area).astype(np.uint8)
    if(type == 'vertical'):
        height, width = np.array(common_area).shape
        energy_function[:, 0] = 1000000000
        energy_function[:, width-1] = 1000000000
    else:
        width, height = np.array(common_area).shape
        energy_function[0, :] = 1000000000
        energy_function[width-1, :] = 1000000000

    # ---dp optimization
    for i in range(height):

        if(i == 0):
            if(type == 'vertical'):
                energy_function[0, :] = common_area[0, :]
            else:
                energy_function[:, 0] = common_area[:, 0]
        else:
            for j in range(1, width-1):
                if(type == 'vertical'):
                    distance = common_area[i, j]+energy_function[i-1, j-1:j+2]
                    index = np.where(distance == np.min(distance))[0][0]
                    index_matrix[i, j] = index
                    energy_function[i, j] = np.min(distance)
                else:
                    distance = common_area[j, i]+energy_function[j-1:j+2, i-1]
                    index = np.where(distance == np.min(distance))[0][0]
                    index_matrix[j, i] = index
                    energy_function[j, i] = np.min(distance)

    return energy_function, index_matrix, [y_min, y_max], [x_min, x_max], [height, width]

# ---------------------------
# dp optimization to find a path with minimum eror


def dp_opt(frame1, frame2, boundaries, mask, i):

    energy_function, index_matrix, y_list, x_list, size = minimum_eror(
        frame1, frame2, boundaries, "vertical", i)
    # make mask and blending :
    height, width = size[0], size[1]
    y_min1, y_max1 = y_list[0], y_list[1]
    x_min1, x_max1 = x_list[0], x_list[1]

    for k in range(height-1, 0, -1):
        if k == height-1:
            y = np.where(
                np.min(energy_function[k, :]) == energy_function[k, :])[0][0]
        # image = cv2.circle(frame1, (y+y_min, k+x_min), 2, (255, 0, 0), 2)
        mask[k+x_min1, : y_min1+y+1, :] = 0
        # frame1[k+x_min, y_min+y:, :] = frame2[k+x_min, y_min+y:, :]
        y_new = np.copy(y+index_matrix[k, y]-1)
        y = np.copy(y_new)

    # ------------------
    # ------------------

    energy_function, index_matrix, y_list, x_list, size = minimum_eror(
        frame1, frame2, boundaries, "horizental_up", i)
    # make mask and blending :
    height, width = size[0], size[1]
    y_min2, y_max2 = y_list[0], y_list[1]
    x_min2, x_max2 = x_list[0], x_list[1]
    for k in range(height-1, 0, -1):
        if k == height-1:
            y = np.where(
                np.min(energy_function[:, k]) == energy_function[:, k])[0][0]
        # image = cv2.circle(frame1, (k+y_min2, y+x_min2), 2, (255, 0, 0), 2)
        mask[:y+x_min2+1, y_min2+k, :] = 0
        # frame1[k+x_min, y_min+y:, :] = frame2[k+x_min, y_min+y:, :]
        y_new = np.copy(y+index_matrix[y, k]-1)
        y = np.copy(y_new)

    # ------------------------
    # ------------------------

    energy_function, index_matrix, y_list, x_list, size = minimum_eror(
        frame1, frame2, boundaries, "horizental_down", i)
    # make mask and blending :
    height, width = size[0], size[1]
    y_min3, y_max3 = y_list[0], y_list[1]
    x_min3, x_max3 = x_list[0], x_list[1]
    for k in range(height-1, 0, -1):
        if k == height-1:
            y = np.where(
                np.min(energy_function[:, k]) == energy_function[:, k])[0][0]
        # image = cv2.circle(
        #     frame1, (k+y_min3, y+x_min3-1000), 2, (255, 0, 0), 2)
        mask[y+x_min3-500+1:, y_min3+k, :] = 0
        # frame1[k+x_min, y_min+y:, :] = frame2[k+x_min, y_min+y:, :]
        y_new = np.copy(y+index_matrix[y, k]-1)
        y = np.copy(y_new)

    frame1 = (frame1*(1-mask)+frame2*(mask)).astype(np.uint8)
    frame1[frame1 > 255] = 255
    frame1[frame1 < 0] = 0
    frame1[:, :, :] = frame1

    x_min = max(x_min1, x_min2)
    x_max = min(x_max1, x_max3)
    y_min = y_min1
    y_max = max(y_max2, y_max3)
    n1 = (x_max-x_min) % 512
    n2 = (y_max-y_min) % 512

    blend = laplacian_pyramid(
        frame1[x_min:x_max-n1+512, y_min:y_max-n2+512, :], frame2[x_min:x_max-n1+512, y_min:y_max-n2+512, :], 255*mask[x_min:x_max-n1+512, y_min:y_max-n2+512, :])
    frame1[x_min:x_max-n1+512, y_min:y_max-n2+512, :] = blend
    return frame1

# -------------------------------------------
# ---find intersecting areas between two frame


def intersecting_areas(img1, homo1, homo2):
    height, width = np.array(img1).shape[:2]

    corners_of_img1 = np.float32(
        [[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]]).reshape(-1, 1, 2)
    new_corners1 = cv2.perspectiveTransform(
        corners_of_img1, homo1)

    corners_of_img2 = np.float32(
        [[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]]).reshape(-1, 1, 2)
    new_corners2 = cv2.perspectiveTransform(
        corners_of_img2, homo2)

    a1, a2, a3, a4 = new_corners1[0], new_corners1[1], new_corners1[2], new_corners1[3]
    b1, b2, b3, b4 = new_corners2[0], new_corners2[1], new_corners2[2], new_corners2[3]

    line1 = line(b1[0], b4[0])
    line2 = line(b2[0], b3[0])
    line3 = line(a3[0], a4[0])
    line4 = line(a1[0], a4[0])
    line5 = line(a2[0], a3[0])

    boundries = []
    boundries.append(b1[0])
    boundries.append(b2[0])
    if(intersection(line2, line3)[0] < intersection(line2, line5)[0]):
        boundries.append(intersection(line2, line3))
    else:
        boundries.append(intersection(line2, line5))

    if(intersection(line1, line3)[0] < intersection(line1, line4)[0]):
        boundries.append(intersection(line1, line3))
    else:
        boundries.append(intersection(line1, line4))
    return boundries
# -----------------
# -----------------


def stich(list_of_frame, list_of_homographies, dimmension, blending=True):

    for i in range(0, len(list_of_frame)):
        if(i == 0):
            homo, t = norimilize_matrix(
                list_of_frame[i], list_of_homographies[i])
            last_homo = np.copy(homo)

        else:
            homo = np.copy(list_of_homographies[i])
            homo = np.dot(t, homo)

        warped = cv2.warpPerspective(
            list_of_frame[i], homo, dimmension, flags=cv2.INTER_LINEAR)

        if(i == 0):
            merged_image = np.copy(warped)
        else:
            boundries = (intersecting_areas(list_of_frame[i-1],
                                            last_homo, homo))
            borders = find_boarder(list_of_frame[i], homo)
            output_cors = np.array([borders[0][0][:], borders[1][0]
                                    [:], borders[2][0][:], borders[3][0][:]], dtype=np.int32)
            mask = np.zeros([dsz[1], dsz[0], 3])
            cv2.fillConvexPoly(mask, output_cors, (255, 255, 255))
            mask /= 255
            if(blending):
                merged_image = dp_opt(
                    merged_image, warped, boundries, mask, i-1)
            else:
                merged_image = (merged_image*(1-mask) +
                                warped*(mask)).astype(np.uint8)

            last_homo = homo

    return merged_image

# ----------------
# ----------------


def find_homograpgy(src1, src2):

    # find key points between two specific frames
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(src1.astype(np.uint8), None)
    kp2, des2 = sift.detectAndCompute(src2.astype(np.uint8), None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []

    # Apply ratio test to find matches between two frames
    for m, n in matches:
        if m.distance < 0.95*n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    # find homography between list of corresponding points
    homography, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5, maxIters=7000, confidence=0.995)

    homography[:2, 2] *= 4
    homography[2, :2] /= 4

    return homography

# -----------------------
# -----------------------


def norimilize_matrix(src, hom):
    final = np.copy(hom)
    h, w = src.shape[:2]
    p = [[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]]
    p_prime = np.array(np.dot(hom, p))
    p_zegond = p_prime/p_prime[2, :]
    x_min = np.min(p_zegond[0, :])
    y_min = np.min(p_zegond[1, :])
    t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    if(x_min < 0):
        t[0, 2] = x_min*-1

    t[0, 2] += 20
    if(y_min < 0):
        t[1, 2] = y_min*-1

    t[1, 2] += 20

    return np.dot(t, final), t

# -----------------------------------
# -----------------------------------


def movment_detection_video(list_of_original_frames, list_of_homographies, background_image, name, dsz):

    number_of_frames = 900
    # ----------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{str(name)}.mp4", fourcc, 30, dsz)
    # ---------------------------
    Threshold = 140
    for i in range(number_of_frames):
        final_mask = np.zeros([dsz[1], dsz[0], 3])
        homography = list_of_homographies[i]
        warped_frame = cv2.warpPerspective(
            background_image, np.linalg.inv(homography), dsz, flags=cv2.INTER_LINEAR).astype(np.int16)

        difference = (list_of_frames[i]-warped_frame).astype(np.int16)
        mask = np.abs(
            difference[:, :, 0]) + np.abs(difference[:, :, 1])+np.abs(difference[:, :, 2])
        # ----------------------------------------------
        mask = mask > Threshold
        kernel = np.ones((3, 3), np.uint8)
        final_mask[:, :, 0] = mask
        final_mask[0:int(1/10*c1), :, :] = 0
        final_mask[int(9/10*c1):, :, :] = 0

        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((25, 25), np.uint8)
        closing = cv2.morphologyEx(
            opening, cv2.MORPH_CLOSE, kernel).astype(np.uint16)

        # -----------------------------------------------

        frame = np.copy(list_of_frames[i])
        frame = frame.astype(np.uint16)
        frame[:, :, :] += closing*100
        frame[frame > 255] = 255

        out.write(cv2.cvtColor(frame.astype(
            np.uint8), cv2.COLOR_BGR2RGB))
        del warped_frame, difference, mask, frame, final_mask

# -----------------------------------


def make_wider_video(number_of_frames, original_width, list_of_homos, dsz, name, subtracted_image):
    background_image = subtracted_image
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{str(name)}.mp4", fourcc, 30, dsz)

    for i in range(number_of_frames):

        homography = list_of_homos[i, :, :]
        translation_matrix = np.identity(3)
        translation_matrix[0, 2] = +1*(dsz[0]-original_width)/900*i
        warped_frame = cv2.warpPerspective(
            background_image, np.dot(translation_matrix, np.linalg.inv(homography)), dsz, flags=cv2.INTER_LINEAR)
        out.write(cv2.cvtColor(warped_frame.astype(
            np.uint8), cv2.COLOR_BGR2RGB))

    out.release()
# ----------------------------------


def make_video(frames, list_of_homos, dsz, name, background=False, smoothnes=False, **kwargs):

    background_image = kwargs.get('c', None)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    size_0f_window = dsz
    out = cv2.VideoWriter(f"{str(name)}.mp4", fourcc, 30, dsz)

    for i, frame in enumerate(frames):

        homography = list_of_homos[i, :, :]

        if(background == False):
            if(smoothnes == False):
                warped_frame = cv2.warpPerspective(
                    frame, homography, size_0f_window, flags=cv2.INTER_LINEAR)
                out.write(cv2.cvtColor(warped_frame, cv2.COLOR_BGR2RGB))
            else:
                warped_frame = cv2.warpPerspective(
                    frame, np.dot(np.linalg.inv(homography), list_of_all_homographies[i]), size_0f_window, flags=cv2.INTER_LINEAR)
                out.write(cv2.cvtColor(warped_frame, cv2.COLOR_BGR2RGB))

        else:
            warped_frame = cv2.warpPerspective(
                background_image, np.linalg.inv(homography), dsz, flags=cv2.INTER_LINEAR)

            out.write(cv2.cvtColor(warped_frame.astype(
                np.uint8), cv2.COLOR_BGR2RGB))

    out.release()


# -----------------------------------------------
# ----find homography between two arbitrary frame
# resize frames by factor of 4 in order to reduce run time

def find_homo_between_two_frame(frame1, num1, frame2, key1, key2):

    height, width = np.array(frame1).shape[:2]
    dim = (int(width/4), int(height/4))

    if(num1 < 270):
        homo_frame1_middle = find_homograpgy(cv2.resize(
            frame1, dim, interpolation=cv2.INTER_AREA), cv2.resize(key1, dim, interpolation=cv2.INTER_AREA))
        homo_middle_frame2 = find_homograpgy(cv2.resize(
            key1, dim, interpolation=cv2.INTER_AREA), cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA))
        final_homo = np.dot(homo_frame1_middle, homo_middle_frame2)

    elif(num1 > 630):

        homo_frame1_middle = find_homograpgy(cv2.resize(
            frame1, dim, interpolation=cv2.INTER_AREA), cv2.resize(key2, dim, interpolation=cv2.INTER_AREA))
        homo_middle_frame2 = find_homograpgy(cv2.resize(
            key2, dim, interpolation=cv2.INTER_AREA), cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA))
        final_homo = np.dot(homo_frame1_middle, homo_middle_frame2)

    else:

        final_homo = find_homograpgy(cv2.resize(frame1, dim, interpolation=cv2.INTER_AREA), cv2.resize(
            frame2, dim, interpolation=cv2.INTER_AREA))

    return final_homo

# -------------------------
# ---------------find border


def find_boarder(frame, homo):

    height, width = np.array(frame).shape[:2]

    corners_of_frame = np.float32(
        [[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]]).reshape(-1, 1, 2)
    new_corners = cv2.perspectiveTransform(corners_of_frame, homo)

    return new_corners

# ------------------------------------
# ---------------background subtration


def background_subtraction(frames, corners, homographies):

    width = 6000
    height = 2560
    final_subtracted_frame = np.zeros([2560, 6000, 3])
    number_of_small_window = 230
    width_of_each_small_window = 6000//number_of_small_window

    for i in range(number_of_small_window):

        x_min_valid = i*width_of_each_small_window
        x_max_valid = (i+1)*width_of_each_small_window
        list_of_valid_windows = []

        # ----------------------------------------------
        c = 0
        for j in range(len(frames)):
            x_min = max(corners[j][0][0][0], corners[j][1][0][0])
            x_max = min(corners[j][2][0][0], corners[j][3][0][0])
            homography = homographies[j]
            if ((x_min <= x_max_valid and x_min_valid <= x_max)):
                c += 1
                translation_matrix = np.identity(3)
                translation_matrix[0, 2] = -1*x_min_valid
                warped = cv2.warpPerspective(frames[j].astype(
                    np.uint8), np.dot(translation_matrix, homography), (width_of_each_small_window, 2560), flags=cv2.INTER_LINEAR)
                list_of_valid_windows.append(warped)

                del warped
        # -------------------------------------------------------------

        if(c != 0):

            final_subtracted_frame[:, x_min_valid:x_max_valid,
                                   :] = np.median(list_of_valid_windows, axis=0).astype(np.uint8)
            del list_of_valid_windows
    return final_subtracted_frame

# ------------------------------


def smooth_homos(list_of_all_homos):
    smooth_list = np.zeros_like(list_of_all_homos)
    for i in range(3):
        for j in range(3):
            smooth_list[:, i, j] = savgol_filter(
                list_of_all_homos[:, i, j], 351, 7)
    return smooth_list

# ------------------------------
# -------- Main code -----------
# ------------------------------


# ------Opens the Video file - read frames
cap = cv2.VideoCapture('original_video.mp4')
list_of_frames = []
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False or i == 900:  # read first 99 frames of video
        break
    list_of_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    i += 1
cap.release()
cv2.destroyAllWindows()


c1, c2 = np.array(list_of_frames[0]).shape[:2]
dsz = (5120, 2560)

# --------part1
# -------------------------------

# show a square in both frames
start_time = time.time()
res1 = np.copy(list_of_frames[450])
res2 = np.copy(list_of_frames[270])
pts = np.array([[500, 400], [1000, 400],
                [1000, 1000], [500, 1000]],
               np.int32)

color = (255, 0, 0)
isClosed = True
cv2.polylines(res1, [pts], isClosed, color, thickness=3)
plt.imsave("res01-450-rect.jpg", res1)

homo_270_to_450 = find_homo_between_two_frame(
    list_of_frames[270], 270, list_of_frames[450], list_of_frames[270], list_of_frames[630])
new_pts = cv2.perspectiveTransform(
    np.array([pts.astype(np.float32)]), np.linalg.inv(homo_270_to_450))
cv2.polylines(res2, new_pts.astype(np.int32), isClosed, color, thickness=3)
plt.imsave("res02-270-rect.jpg", res2)

# ----combine two frames
numbers = [270, 450]
dsz = (2650, 1500)
list_of_homo_5key_frame = []
list_of_image_5key_frame = []
for i in range(len(numbers)):
    index = int(numbers[i])
    homography = find_homo_between_two_frame(
        list_of_frames[index], index, list_of_frames[450], list_of_frames[270], list_of_frames[630])
    list_of_homo_5key_frame.append(homography)
    list_of_image_5key_frame.append(list_of_frames[index])

final_output = stich(list_of_image_5key_frame,
                     list_of_homo_5key_frame, dsz, False)
plt.imsave("res03-270-450-panorama.jpg", final_output)
print("part1 done :)")
print(f"execution time : {(time.time() - start_time)} seconds\n")

# ---------part2
# ----------combine 5 key frames
start_time = time.time()
numbers = [90, 270, 450, 630, 810]
dsz = (5300, 2400)
list_of_homo_5key_frame = []
list_of_image_5key_frame = []
for i in range(len(numbers)):
    index = int(numbers[i])
    homography = find_homo_between_two_frame(
        list_of_frames[index], index, list_of_frames[450], list_of_frames[270], list_of_frames[630])
    list_of_homo_5key_frame.append(homography)
    list_of_image_5key_frame.append(list_of_frames[index])

final_output = stich(list_of_image_5key_frame,
                     list_of_homo_5key_frame, dsz, True)
plt.imsave("res04-key-frames-panorama.jpg", final_output)
print("part2 done :)")
print(f"execution time : {(time.time() - start_time)} seconds\n")
# ---------------------------------------------
start_time = time.time()
# ---make list of all homographies and corners
list_of_corners = []
list_of_all_homographies = []
for i, frame in enumerate(list_of_frames):
    if(i == 0):
        homography = find_homo_between_two_frame(
            list_of_frames[i], i, list_of_frames[450], list_of_frames[270], list_of_frames[610])
        homography, t = norimilize_matrix(
            frame, homography)
    elif(i != 0):
        homography = find_homo_between_two_frame(
            list_of_frames[i], i, list_of_frames[450], list_of_frames[270], list_of_frames[610])
        homography = np.dot(t, homography)
    list_of_all_homographies.append(homography)
    list_of_corners.append(find_boarder(frame, homography))


list_of_all_homographies = np.array(list_of_all_homographies)
list_of_corners = np.array(list_of_corners)
smooth_homogrphies = smooth_homos(list_of_all_homographies)
# --------------------------------------------------------------

# ---------part3
dsz = (6000, 2560)
make_video(list_of_frames, list_of_all_homographies,
           dsz, "res05-reference-plane", False, False)
print("part3 done :)")
print(f"execution time : {(time.time() - start_time)} seconds\n")

# ----------part4
start_time = time.time()
subtracted_image = background_subtraction(
    list_of_frames, list_of_corners, list_of_all_homographies)
plt.imsave("res06-background-panorama.jpg",
           subtracted_image.astype(np.uint8))
print("part4 done :)")
print(f"execution time : {(time.time() - start_time)} seconds\n")

# --------part5
start_time = time.time()
dsz = (int(c2), int(c1))
make_video(list_of_frames, list_of_all_homographies,
           dsz, "res07-background-video", True, False, c=subtracted_image)
print("part5 done :)")
print(f"execution time : {(time.time() - start_time)} seconds\n")

# -------part6
start_time = time.time()
dsz = (int(c2), int(c1))
movment_detection_video(list_of_frames, list_of_all_homographies,
                        subtracted_image, "res08-foreground-video", dsz)
print("part6 done :)")
print(f"execution time : {(time.time() - start_time)} seconds\n")

# -------part 7
start_time = time.time()
dsz = (int(1.5*c2), int(c1))
subtracted_image = plt.imread("res06-background-panorama.jpg")
make_wider_video(len(list_of_frames), c2, list_of_all_homographies,
                 dsz, "res09-background-video-wider", subtracted_image)
print("part7 done :)")
print(f"execution time : {(time.time() - start_time)} seconds\n")

# ------part8
start_time = time.time()
dsz = (int(c2), int(c1))
make_video(list_of_frames, smooth_homogrphies, dsz, "res10-video-shakeless",
           False, True, c=subtracted_image)
print("part8 done :)")
print(f"execution time : {(time.time() - start_time)} seconds\n")
