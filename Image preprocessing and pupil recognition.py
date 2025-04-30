import os
import cv2
import cv2 as cv
import numpy as np
import glob
import pickle
import pandas as pd
import time


# ==========------------************ 更改内容 ************-------------================
# 首先要考虑二值化的阈值是否合理
name = 'Zhaozihe_164800_1'
path = 'C:\\Users\\86138\\Desktop\\' + name 
# 眼角初筛框选范围
boundary_threshold1 = 0.8  # 左眼
boundary_threshold2 = 0.8  # 右眼
boundary_thresholdupdown = 0.1
# 瞳孔二值化阈值
global Glasses_Binarization_threshold, Binarization_threshold1 , kl
kl = (10,10)#闭操作的核，核越小，椭圆算法求出的椭圆边界越精确，但是可能出现曲线不闭合导致读取椭圆失败
Glasses_Binarization_threshold_left = 145   #戴眼镜时候左眼阈值
Glasses_Binarization_threshold_right = 145   #戴眼镜时候右眼阈值
Binarization_threshold_left1 = 120   #不戴眼镜时候左眼阈值
Binarization_threshold_right1 = 120   #不戴眼镜时候右眼阈值
# 是否戴眼镜
if path[-12:-9] == '_wg':
    WithGlassesOrNot = 'Yes'
else:
    WithGlassesOrNot = 'No'

# ==========------------************ 更改内容 ************-------------================

'''
数据处理过程：
1 首张图片
  1.1 调整PupilCenterEstimated中参数scaleFactor和minNeighbors，框选合适的眼睛初步计算 眼角位置
2 全部图片
  2.1.提取video_object文件夹所有图片的质心，确定观察点
  2.2.利用函数PupilCenterEstimated，初步提取眼睛图片文件夹所有图片的瞳孔位置
  2.3.利用PupilEllipse计算所有眼睛图片的瞳孔椭圆
  2.4.根据初步计算的眼角位置计算所有眼角位置
'''


def ExtractionCenterOfMass(image):  # 提取质心

    # 提取符合条件的像素
    mask = ((image[:, :, 0] <= 20) & (image[:, :, 1] <= 20) & (image[:, :, 2] >= 150) & (image[:, :, 2] <= 230)).astype(
        np.uint8) * 255

    # 创建一个黑色背景
    black_background = np.zeros_like(image)

    # 将符合条件的像素置为白色
    black_background[mask == 255] = [255, 255, 255]
    gray = cv2.cvtColor(black_background, cv2.COLOR_BGR2GRAY)

    cx = 639
    cy = 359

    # 计算质心
    m = cv2.moments(gray)
    if m['m10'] > 0:
        # 计算质心坐标
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
    return cx, cy


def PupilCenterEstimated(IM, InitialValue=None):  # 初步估计瞳孔位置及宽度
    # 输入为未经处理的图像
    # InitialValue表示框选的初值，格式为（x,y,w）
    IMleft = IM
    gray_image = cv.cvtColor(IMleft, cv.COLOR_BGR2GRAY)
    # -----------------------检测是否有眼睛------------------------
    # 初步估计瞳孔中心区域(x_eye,y_eye)表示左上角坐标
    # 初步估计眼睛上下宽度w_eye
    x_eye = None
    y_eye = None
    w_eye = None
    # 加载眼睛检测器
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    # 在灰度图像上检测眼睛
    eyes = eye_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=5)
    if np.any(eyes):  # 检测是否有眼睛被检测出来
        if eyes.shape[0] == 1:  # 被检测出一个眼睛
            x_eye, y_eye, w_eye, h_eye = eyes[0]
        if eyes.shape[0] > 1:  # 被检测出多个眼睛
            if InitialValue is not None:
                x, y, w = InitialValue
                min_diff = np.inf
                for i in range(eyes.shape[0]):
                    diff = ((x+w/2) - (eyes[i][0]+eyes[i][2]/2)) ** 2 + ((y+w/2) - (eyes[i][1]+eyes[i][2]/2)) ** 2
                    if diff < min_diff:
                        min_diff = diff
                        closest_index = i
            if InitialValue is None:
                '''
                # 方法一：
                # 横坐标最接近中心（320，240）的视为眼睛
                min_diff = np.inf
                for i in range(eyes.shape[0]):
                    diff = (eyes[i][0] - 320) ** 2 + (eyes[i][1] - 240) ** 2
                    if diff < min_diff:
                        min_diff = diff
                        closest_index = i
                '''
                # 方法二：
                # 宽度最大的视为眼睛
                max_diff = 0
                for i in range(eyes.shape[0]):
                    diff = eyes[i][2]
                    if diff > max_diff:
                        max_diff = diff
                        closest_index = i

            x_eye, y_eye, w_eye, h_eye = eyes[closest_index]

        if w_eye < 80:
            x_eye = round(x_eye - w_eye / 2 + 75)
            y_eye = round(y_eye - w_eye / 2 + 75)
            w_eye = 150
    return x_eye, y_eye, w_eye


def PupilEllipse(IM, PupilCenterEstimated=None, WithGlassesOrNot='No', threshold_leftright=None):
    # 计算瞳孔椭圆
    if PupilCenterEstimated is not None:
        x_eye, y_eye, w_eye = PupilCenterEstimated
        if x_eye is not None:
            IMleft = IM[y_eye:(w_eye + y_eye), x_eye:(w_eye + x_eye)]
        else:
            IMleft = IM
    else:
        IMleft = IM

    if WithGlassesOrNot == 'Yes':
        if threshold_leftright == 'right':
            Binarization_threshold = Glasses_Binarization_threshold_right
        else:
            Binarization_threshold = Glasses_Binarization_threshold_left
    else:
        if threshold_leftright == 'right':
            Binarization_threshold = Binarization_threshold_right1
        else:
            Binarization_threshold = Binarization_threshold_left1

    # 求瞳孔中心
    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, kl)  # 闭操作的核
    gray_image = cv.cvtColor(IMleft, cv.COLOR_BGR2GRAY)

    gray_IMleft = cv.GaussianBlur(gray_image, (5, 5), 0)  # 滤波
    # IM_gray_GaussianBlur_equalizeHist = cv.equalizeHist(gray_IMleft)  # 直方图拉伸
    _, binary_IMleft = cv.threshold(gray_IMleft, Binarization_threshold, 255, cv.THRESH_BINARY_INV)  # 二值化
    closing_IMleft = cv.morphologyEx(binary_IMleft, cv.MORPH_OPEN, kernel1)  # 开操作
    # opening = cv.morphologyEx(binary_IMleft, cv.MORPH_OPEN, kernel1, 1)
    closing_IMleft = cv.Canny(closing_IMleft, 45, 150)
    contours_IMleft, _ = cv.findContours(closing_IMleft, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    best_ellipse = None
    best_similarity = 10000.0
    n = len(contours_IMleft)
    if n > 0:
        for contour in contours_IMleft:
            S = cv.contourArea(contour, True)
            #print(abs(S))
            if 300 < abs(S) < 20000:
                # print(abs(S))
                ellipse = cv.fitEllipse(contour)
                # cv.ellipse(image, ellipse, (255, 0, 0), 2)
                center, axes, angle = ellipse
                # print(axes)
                start_angle = 0
                end_angle = 360
                step = 1
                ellipse_points = cv.ellipse2Poly((int(center[0]), int(center[1])), (int(axes[0] / 2), int(axes[1] / 2)),
                                                 int(angle),
                                                 start_angle, end_angle, step)
                similarity = cv.matchShapes(contour, ellipse_points, cv.CONTOURS_MATCH_I1, 0.0)
                # 判断边界点是否像椭圆，挑出最像的
                if similarity < best_similarity:
                    if abs((axes[0] - axes[1]) / axes[0]) < 0.6:
                        best_similarity = similarity
                        best_ellipse = ellipse
                        selected = contour
    if best_ellipse is not None:
        best_ellipse = list(best_ellipse)
        [(k1_s, k2_s), (k3_s, k4_s), k5_s] = best_ellipse
        # print(best_ellipse)
        if abs((k3_s - k4_s) / k3_s) > 0.6:
            k1_s = k2_s = k3_s = k4_s = k5_s = None
        # print(k1_s)
    else:
        k1_s = k2_s = k3_s = k4_s = k5_s = None
    # print(k1_s)
    if k1_s is not None:
        pupil_left_x = k1_s
        pupil_left_y = k2_s
        # 平滑处理
        selected_contour = cv.blur(selected, (5, 5))
        # 降维
        new_contour = np.squeeze(selected_contour)
        satisfy_contour = []  # 目标点
        for i in range(len(selected_contour)):
            prev10 = selected_contour[i - 10] if i >= 10 else selected_contour[i - 10 + len(selected_contour)]
            prev5 = selected_contour[i - 5] if i >= 5 else selected_contour[i - 5 + len(selected_contour)]
            prev3 = selected_contour[i - 3] if i >= 3 else selected_contour[i - 3 + len(selected_contour)]
            prev2 = selected_contour[i - 2] if i >= 2 else selected_contour[i - 2 + len(selected_contour)]
            prev1 = selected_contour[i - 1] if i >= 1 else selected_contour[i - 1 + len(selected_contour)]
            curr = selected_contour[i]
            next1 = selected_contour[(i + 1) % len(selected_contour)]
            next2 = selected_contour[(i + 2) % len(selected_contour)]
            next3 = selected_contour[(i + 3) % len(selected_contour)]
            next5 = selected_contour[(i + 5) % len(selected_contour)]
            next10 = selected_contour[(i + 10) % len(selected_contour)]
            # ----------------- 1.对插值后的边界点计算曲率方向，与，边界点与拟合椭圆心切线方向，作比较
            # 关系应该是接近垂直，所以余弦值接近0
            # 计算方向向量
            vec_tangent = np.array(next3 - prev3)
            # 计算半径的法相向量
            vec_normal = np.array(curr - (pupil_left_x, pupil_left_y))
            # 计算向量的模
            norm_tangent = np.linalg.norm(vec_tangent)
            norm_normal = np.linalg.norm(vec_normal)
            # 计算向量的点积
            dot_product = np.dot(vec_tangent, vec_normal.T)
            # 计算夹角的余弦值
            if norm_tangent != 0 and norm_normal != 0:
                cos_angle = dot_product / (norm_tangent * norm_normal)
            else:
                cos_angle = 0  # 或者设置为其他合适的值
            angle_cos = np.abs(cos_angle)
            # ------------------------------------------------------------------------
            # ----------------- 2. 计算凹凸性 ------------------------------------------
            # 向量next-curr 与 向量curr-prev 的差应该与 向量curr-圆心 成锐角，也就是和法向向量成钝角，余弦值小于0
            # 计算差向量
            vec_differ = (next5 - curr) - (curr - prev5)
            norm_differ = np.linalg.norm(vec_differ)
            # 计算向量的点积
            dot_product_AT = np.dot(vec_differ, vec_normal.T)
            # 计算夹角的余弦值
            if norm_differ != 0 and norm_normal != 0:
                cos_angle_AT = dot_product_AT / (norm_differ * norm_normal)
            else:
                cos_angle_AT = 0  # 或者设置为其他合适的值
            angle_cos_AT = abs(cos_angle_AT)
            # print(i,vec_differ,vec_normal, angle_cos_AT, norm_normal)
            # -------------------------------------------------------------------------
            # ----------------- 根据阈值选择点------------------------------------------
            # 设定阈值
            if angle_cos < 0.5:  # 设定阈值1： 余弦值接近0  <<<<<<<<------
                if angle_cos_AT > 0.5:  # 设定阈值2： a凹凸性，余弦值小于0  <<<<<<<<------
                    if norm_normal > k3_s / 2 or norm_normal > k4_s / 2:
                        # if True:
                        satisfy_contour.append(curr)
                    # print(i, curr, angle_cos, angle_cos_AT)
            #     ---------------------------------------------------------------------
        satisfy_contour = np.array(satisfy_contour)
        if len(satisfy_contour) > 5:
            ellipsel_satisfy = cv.fitEllipse(satisfy_contour)
            ellipsel_satisfy = list(ellipsel_satisfy)
            [(k1, k2), (k3, k4), k5] = ellipsel_satisfy
            # print(ellipsel_satisfy)
        else:
            k1 = k2 = k3 = k4 = k5 = None
    else:
        k1 = k2 = k3 = k4 = k5 = None

    if x_eye is not None and k1 is not None:
        k1 = k1 + x_eye
        k2 = k2 + y_eye

    return k1, k2, k3, k4, k5


def EyeCorner(IM, ty, Boundary=None, Location=None):
    '''     找眼角
    ty=1表示左眼，ty=0表示右眼
    Boundary 表示是否有眼睛初定位数据，经过PupilCenterEstimated求出的Boundary，
              格式（boundary_up, boundary_down, boundary_left，boundary_right）,用在文件夹第一张图片眼角定位
    Location 表示是否有精确定位的眼角位置，格式（x, y），用在已有第一张定位眼角后，精确定位每一张眼角
    '''
    if Boundary is not None:
        boundary_up, boundary_down, boundary_left, boundary_right = Boundary
        IM_eye = IM[boundary_up:boundary_down, boundary_left:boundary_right]
        Location = None
    else:
        IM_eye = IM

    gray_image = cv.cvtColor(IM_eye, cv.COLOR_BGR2GRAY)
    gray_image = cv.GaussianBlur(gray_image, (5, 5), 0)  # 滤波
    # 定义20x20的卷积核
    kernel_size = (20, 20)
    kernel = np.ones(kernel_size, dtype=np.float32) / (kernel_size[0] * kernel_size[1])

    if Location is not None:
        x, y = Location
        # ======================眼角精提取================================
        # 获取图像尺寸
        height, width, = gray_image.shape
        # 计算截取区域的上下 左右 边界
        length = 5
        top = max(0, y - length)
        bottom = min(height - 1, y + length)
        left = max(0, x - length)
        right = min(width - 1, x + length)
        # 截取图像区域
        cropped_image = gray_image[top:bottom, left:right]
        cropped_image = cv.equalizeHist(cropped_image)  # 直方图拉伸
        # 定义5x5的卷积核
        kernel_size1 = (5, 5)
        kernel1 = np.ones(kernel_size1, dtype=np.float32) / (kernel_size1[0] * kernel_size1[1])
        # 对图像进行卷积操作
        cropped_image5 = cv.filter2D(cropped_image, -1, kernel1)
        ret, binaryC = cv.threshold(cropped_image5, 100, 255, cv.THRESH_BINARY_INV)  # 二值化
        heightC, widthC = binaryC.shape
        if ty == 0:
            binaryC[:, int(widthC - 0.5 * (right - x)):] = 255
        elif ty == 1:
            binaryC[:, :int(0.5 * (x - left))] = 255
        contours, _ = cv.findContours(binaryC, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        n = len(contours)
        Smax = 0
        corner_x, corner_y = x, y
        for i in range(n):
            S = cv.contourArea(contours[i], True)
            con = np.array(contours[i])
            if Smax < abs(S):
                Smax = abs(S)
                y_coordinates5 = con[:, 0, 1]
                x_coordinates5 = con[:, 0, 0]
                if len(y_coordinates5):
                    if ty == 0:
                        min_x_index5 = np.argmin(x_coordinates5)
                        corner_x = x_coordinates5[min_x_index5]
                        corner_y = y_coordinates5[min_x_index5]
                    if ty == 1:
                        max_x_index5 = np.argmax(x_coordinates5)
                        corner_x = x_coordinates5[max_x_index5]
                        corner_y = y_coordinates5[max_x_index5]
                else:
                    corner_x, corner_y = x, y
        eye_corner_x = corner_x + left
        eye_corner_y = corner_y + top
    else:
        # 对图像进行卷积操作
        convolved_image = cv.filter2D(gray_image, -1, kernel)
        # 应用Canny算子
        canny_edges = cv.Canny(convolved_image, 10, 15)
        indices = np.where(canny_edges == 255)
        y_coordinates = indices[0]
        x_coordinates = indices[1]
        if len(x_coordinates):
            if ty == 0:
                # 找到横坐标最小的点的索引
                min_x_index = np.argmin(x_coordinates)
                # 找到横坐标最小的点的坐标
                x = x_coordinates[min_x_index]
                y = y_coordinates[min_x_index]
            elif ty == 1:
                # 找到横坐标最大的点的索引
                max_x_index = np.argmax(x_coordinates)
                # 找到横坐标最大的点的坐标
                x = x_coordinates[max_x_index]
                y = y_coordinates[max_x_index]
            # ======================眼角精提取================================
            # 获取图像尺寸
            height, width, = gray_image.shape
            # 计算截取区域的上下 左右 边界
            length = 40
            top = max(0, y - length)
            bottom = min(height - 1, y + length)
            left = max(0, x - length)
            right = min(width - 1, x + length)
            # 截取图像区域
            cropped_image = gray_image[top:bottom, left:right]
            cropped_image = cv.equalizeHist(cropped_image)  # 直方图拉伸
            # 定义5x5的卷积核
            kernel_size1 = (5, 5)
            kernel1 = np.ones(kernel_size1, dtype=np.float32) / (kernel_size1[0] * kernel_size1[1])
            # 对图像进行卷积操作
            cropped_image5 = cv.filter2D(cropped_image, -1, kernel1)
            ret, binaryC = cv.threshold(cropped_image5, 100, 255, cv.THRESH_BINARY_INV)  # 二值化
            heightC, widthC = binaryC.shape
            if ty == 0:
                binaryC[:, int(widthC - 0.5 * (right - x)):] = 255
            elif ty == 1:
                binaryC[:, :int(0.5 * (x - left))] = 255
            contours, _ = cv.findContours(binaryC, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            n = len(contours)
            Smax = 0
            for i in range(n):
                S = cv.contourArea(contours[i], True)
                con = np.array(contours[i])
                if Smax < abs(S):
                    Smax = abs(S)
                    y_coordinates5 = con[:, 0, 1]
                    x_coordinates5 = con[:, 0, 0]
                    if len(y_coordinates5):
                        if ty == 0:
                            min_x_index5 = np.argmin(x_coordinates5)
                            corner_x = x_coordinates5[min_x_index5]
                            corner_y = y_coordinates5[min_x_index5]
                        if ty == 1:
                            max_x_index5 = np.argmax(x_coordinates5)
                            corner_x = x_coordinates5[max_x_index5]
                            corner_y = y_coordinates5[max_x_index5]
                    else:
                        corner_x, corner_y = x, y
            eye_corner_x = corner_x + left
            eye_corner_y = corner_y + top
        else:
            eye_corner_x = None
            eye_corner_y = None
    if Boundary is not None:
        eye_corner_x = eye_corner_x + boundary_left
        eye_corner_y = eye_corner_y + boundary_up

    return eye_corner_x, eye_corner_y


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global eye_corner_first_left_x, eye_corner_first_left_y, eye_corner_first_right_x, eye_corner_first_right_y
        global eye_corner_w_left_x, eye_corner_w_left_y, eye_corner_w_right_x, eye_corner_w_right_y
        if x < 639:
            if x > 320:
                eye_corner_first_left_x = x
                eye_corner_first_left_y = y
                print(f'左眼内眼角坐标: ({eye_corner_first_left_x}, {eye_corner_first_left_y})')
                cv2.circle(result, (x, y), 5, (0, 255, 255), -1)
                cv2.imshow("Result", result)
            else:
                eye_corner_w_left_x = x
                eye_corner_w_left_y = y
                print(f'左眼外眼角坐标: ({eye_corner_w_left_x}, {eye_corner_w_left_y})')
                cv2.circle(result, (x, y), 5, (255, 0, 255), -1)
                cv2.imshow("Result", result)

        else:
            if x < 960:
                eye_corner_first_right_x = x - 639
                eye_corner_first_right_y = y
                print(f'右眼内眼角坐标: ({eye_corner_first_right_x}, {eye_corner_first_right_y})')
                cv2.circle(result, (x, y), 5, (0, 255, 255), -1)
                cv2.imshow("Result", result)
            else:
                eye_corner_w_right_x = x - 639
                eye_corner_w_right_y = y
                print(f'右眼外眼角坐标: ({eye_corner_w_right_x}, {eye_corner_w_right_y})')
                cv2.circle(result, (x, y), 5, (255, 0, 255), -1)
                cv2.imshow("Result", result)



# ==================================================================================
# 创建子文件夹
Pic_eye_treated = os.path.join(path, 'Pic_eye_treated')
os.makedirs(Pic_eye_treated, exist_ok=True)

# 1 ==================================================================================
# 读取第first张眼睛图片，通过调整参数，初步估计 "眼角" 位置
# 读取第一张Pic
path_left = path + '\\original_left_eye'
path_right = path + '\\original_right_eye'
for filename in os.listdir(path_left):
    if filename.endswith(".jpg"):
        img_path = os.path.join(path_left, filename)
        IM_first_left = cv2.imread(img_path)
        PupilCenterEstimated_left_first = PupilCenterEstimated(IM_first_left)
        x_eye_left_first, y_eye_left_first, w_eye_left_first = PupilCenterEstimated_left_first
        # 观察瞳孔
        k1_left_first, k2_left_first, k3_left_first, k4_left_first, k5_left_first = PupilEllipse(IM_first_left,
                                                                   PupilCenterEstimated=PupilCenterEstimated_left_first,
                                                                   WithGlassesOrNot=WithGlassesOrNot,threshold_leftright='left')
        if x_eye_left_first is not None and k1_left_first is not None:
            break  # 找到一张图片后退出循环

for filename in os.listdir(path_right):
    if filename.endswith(".jpg"):
        img_path = os.path.join(path_right, filename)
        IM_first_right = cv2.imread(img_path)
        PupilCenterEstimated_right_first = PupilCenterEstimated(IM_first_right)
        x_eye_right_first, y_eye_right_first, w_eye_right_first = PupilCenterEstimated_right_first
        k1_right_first, k2_right_first, k3_right_first, k4_right_first, k5_right_first = PupilEllipse(IM_first_right,
                                                                        PupilCenterEstimated=PupilCenterEstimated_right_first,
                                                                        WithGlassesOrNot=WithGlassesOrNot,threshold_leftright='right')
        if x_eye_left_first is not None and k1_right_first is not None:
            break  # 找到一张图片后退出循环
# 找眼睛边界
boundary_up_left_first = round(max([y_eye_left_first+w_eye_left_first*boundary_thresholdupdown, 0]))
boundary_down_left_first = round(min([y_eye_left_first + w_eye_left_first*(1-boundary_thresholdupdown), 479]))
boundary_left_left_first = max([x_eye_left_first - round(w_eye_left_first * boundary_threshold1), 0])
boundary_right_left_first = min([x_eye_left_first + round(w_eye_left_first * (boundary_threshold1 + 1)), 639])
boundary_left_first = [boundary_up_left_first, boundary_down_left_first, boundary_left_left_first,
                       boundary_right_left_first]
boundary_up_right_first = round(max([y_eye_right_first+w_eye_right_first*boundary_thresholdupdown, 0]))
boundary_down_right_first = round(min([y_eye_right_first + w_eye_right_first*(1-boundary_thresholdupdown), 479]))
boundary_left_right_first = max([x_eye_right_first - round(w_eye_right_first * boundary_threshold2), 0])
boundary_right_right_first = min([x_eye_right_first + round(w_eye_right_first * (boundary_threshold2 + 1)), 639])
boundary_right_first = [boundary_up_right_first, boundary_down_right_first, boundary_left_right_first,
                        boundary_right_right_first]
# 用EyeCorner初步识别   内眼角
eye_corner_first_left_x, eye_corner_first_left_y = EyeCorner(IM_first_left, ty=1,
                                                             Boundary=boundary_left_first, Location=None)
eye_corner_first_right_x, eye_corner_first_right_y = EyeCorner(IM_first_right, ty=0,
                                                               Boundary=boundary_right_first, Location=None)

eye_corner_w_left_x, eye_corner_w_left_y = 0, 0 #外眼角
eye_corner_w_right_x, eye_corner_w_right_y = 0, 0

cv.ellipse(IM_first_left, [(int(k1_left_first), int(k2_left_first)), (k3_left_first, k4_left_first), k5_left_first], (0, 0, 255), 2)
cv.ellipse(IM_first_right, [(int(k1_right_first), int(k2_right_first)), (k3_right_first, k4_right_first), k5_right_first], (0, 0, 255), 2)
# 展示
cv2.rectangle(IM_first_left, (x_eye_left_first, y_eye_left_first),
              (x_eye_left_first + w_eye_left_first, y_eye_left_first + w_eye_left_first), (255, 0, 0), 3)
cv2.rectangle(IM_first_right, (x_eye_right_first, y_eye_right_first),
              (x_eye_right_first + w_eye_right_first, y_eye_right_first + w_eye_right_first), (255, 0, 0), 3)
cv2.rectangle(IM_first_left, (boundary_left_left_first, boundary_up_left_first),
              (boundary_right_left_first, boundary_down_left_first), (0, 255, 0), 1)
cv2.rectangle(IM_first_right, (boundary_left_right_first, boundary_up_right_first),
              (boundary_right_right_first, boundary_down_right_first), (0, 255, 0), 1)
cv2.circle(IM_first_left, (eye_corner_first_left_x, eye_corner_first_left_y), 5, (0, 0, 255), -1)
cv2.circle(IM_first_right, (eye_corner_first_right_x, eye_corner_first_right_y), 5, (0, 0, 255), -1)
result = cv2.hconcat([IM_first_left, IM_first_right])
cv2.imshow('Result', result)
# 设置鼠标回调函数
cv2.setMouseCallback('Result', mouse_callback)
cv2.waitKey(0)
print('左眼内眼角：',(eye_corner_first_left_x, eye_corner_first_left_y),'； 右眼内眼角：', (eye_corner_first_right_x, eye_corner_first_right_y))
# ==================================================================================

start_time = time.time()  # 记录程序开始时间
# ==================================================================================================================
# ==================================================================================================================
# ==================================================================================================================
print('数据处理开始')
# 2.1 提取目标点坐标
path_object = path + '\\video_object'
jpg_files = glob.glob(os.path.join(path_object, '*.jpg'))
num = len(jpg_files)
result_object = np.zeros((num, 2))
i_object = 0
for filename in os.listdir(path_object):
    if filename.endswith(".jpg"):
        img_path = os.path.join(path_object, filename)
        IM_first_object = cv2.imread(img_path)
        x, y = ExtractionCenterOfMass(IM_first_object)
        if x == 639 and y == 359:
            result_object[:, 0] = 639
            result_object[:, 1] = 359
    result_object[i_object, 0] = x
    result_object[i_object, 1] = y
    i_object += 1
# 将矩阵存储到pkl文件中
with open(path + '//' + name + '_result_object.pkl', 'wb') as f:
    pickle.dump(result_object, f)

# ==================================================================================
# 2.1 提取目标点坐标
#     ------------------------------左眼---------------------------------
LeftEye = np.zeros((num, 12))
result_left_eye = np.zeros((num, 6))
i_left = 0
RightEye = np.zeros((num, 12))
result_right_eye = np.zeros((num, 6))
i_right = 0
i_s = 0
eye_corner_left_x = eye_corner_first_left_x
eye_corner_left_y = eye_corner_first_left_y
eye_corner_right_x = eye_corner_first_right_x
eye_corner_right_y = eye_corner_first_right_y
x_eye_left, y_eye_left, w_eye_left = x_eye_left_first, y_eye_left_first, w_eye_left_first
x_eye_right, y_eye_right, w_eye_right = x_eye_right_first, y_eye_right_first, w_eye_right_first

k1_l, k2_l, k3_l, k4_l, k5_l = k1_left_first, k2_left_first, k3_left_first, k4_left_first, k5_left_first
k1_r, k2_r, k3_r, k4_r, k5_r = k1_right_first, k2_right_first, k3_right_first, k4_right_first, k5_right_first

for filename_left, filename_right in zip(os.listdir(path_left), os.listdir(path_right)):
    eye_c_left_x = None
    eye_c_left_y = None
    eye_c_right_x = None
    eye_c_right_y = None

    text_left = 'None'
    text_right = 'None'
    #     ------------------------------左眼---------------------------------
    if filename_left.endswith(".jpg"):
        img_path_left = os.path.join(path_left, filename_left)
        IM_left = cv2.imread(img_path_left)
        # 初步读取眼睛图像
        PupilCenterEstimated_left = PupilCenterEstimated(IM_left, (x_eye_left, y_eye_left, w_eye_left))
        if PupilCenterEstimated_left[0] is not None:
            x_eye_l, y_eye_l, w_eye_l = PupilCenterEstimated_left
            if ((x_eye_l+w_eye_l/2)-(x_eye_left+w_eye_left/2))**2+((y_eye_l+w_eye_l/2)-(y_eye_left+w_eye_left/2))**2 < 1000:
                x_eye_left, y_eye_left, w_eye_left = x_eye_l, y_eye_l, w_eye_l
            else:
                x_eye_left = round(k1_l - w_eye_left_first/2)
                y_eye_left = round(k2_l - w_eye_left_first/2)
                w_eye_left = w_eye_left_first

        # 读取瞳孔椭圆
        k1_left, k2_left, k3_left, k4_left, k5_left = PupilEllipse(IM_left,
                                                                   PupilCenterEstimated=(x_eye_left, y_eye_left, w_eye_left),
                                                                   WithGlassesOrNot=WithGlassesOrNot,
                                                                   threshold_leftright='left')
        if (eye_corner_left_x-eye_corner_first_left_x)**2+(eye_corner_left_y-eye_corner_first_left_y)**2 > 50:
            eye_corner_left_x = eye_corner_first_left_x
            eye_corner_left_y = eye_corner_first_left_y
        eye_c_left_x, eye_c_left_y = EyeCorner(IM_left, ty=1, Boundary=None,
                                                         Location=(eye_corner_left_x, eye_corner_left_y))
        if (eye_corner_left_x-eye_c_left_x)**2+(eye_corner_left_y-eye_c_left_y)**2 < 25:
            eye_corner_left_x = eye_c_left_x
            eye_corner_left_y = eye_c_left_y

    print('左眼：', k1_left)
    if k1_left is not None:
        if (k1_left-k1_l)**2 > 5000:
            k1_left = None
        else:
            k1_l, k2_l, k3_l, k4_l, k5_l = k1_left, k2_left, k3_left, k4_left, k5_left
    if k1_left is not None:
        LeftEye[i_left, 0:5] = k1_left, k2_left, k3_left, k4_left, k5_left
    if eye_corner_left_x is not None:
        LeftEye[i_left, 5:7] = eye_corner_left_x, eye_corner_left_y
    if x_eye_left is not None:
        LeftEye[i_left, 7:12] = x_eye_left, y_eye_left, w_eye_left, x_eye_left+w_eye_left/2, y_eye_left+w_eye_left/2

    #     ------------------------------右眼---------------------------------
    if filename_right.endswith(".jpg"):
        img_path_right = os.path.join(path_right, filename_right)
        IM_right = cv2.imread(img_path_right)
        # 初步读取眼睛图像
        PupilCenterEstimated_right = PupilCenterEstimated(IM_right, (x_eye_right, y_eye_right, w_eye_right))
        if PupilCenterEstimated_right[0] is not None:
            x_eye_r, y_eye_r, w_eye_r = PupilCenterEstimated_right
            if ((x_eye_r+w_eye_r/2)-(x_eye_right+w_eye_right/2))**2+((y_eye_r+w_eye_r/2)-(y_eye_right+w_eye_right/2))**2 < 1000:
                x_eye_right, y_eye_right, w_eye_right = x_eye_r, y_eye_r, w_eye_r
            else:
                x_eye_right = round(k1_r - w_eye_right_first / 2)
                y_eye_right = round(k2_r - w_eye_right_first / 2)
                w_eye_right = w_eye_right_first
        # 读取瞳孔椭圆
        k1_right, k2_right, k3_right, k4_right, k5_right = PupilEllipse(IM_right,
                                                                        PupilCenterEstimated=(x_eye_right, y_eye_right, w_eye_right),
                                                                        WithGlassesOrNot=WithGlassesOrNot,
                                                                        threshold_leftright='right')
        if (eye_corner_right_x-eye_corner_first_right_x)**2+(eye_corner_right_y-eye_corner_first_right_y)**2 > 50:
            eye_corner_right_x = eye_corner_first_right_x
            eye_corner_right_y = eye_corner_first_right_y
        eye_c_right_x, eye_c_right_y = EyeCorner(IM_right, ty=0, Boundary=None,
                                                           Location=(eye_corner_right_x, eye_corner_right_y))
        if (eye_corner_right_x-eye_c_right_x)**2+(eye_corner_right_y-eye_c_right_y)**2 < 25:
            eye_corner_right_x = eye_c_right_x
            eye_corner_right_y = eye_c_right_y
    print('右眼：', k1_right)
    if k1_right is not None:
        if (k1_right-k1_r)**2 > 5000:
            k1_right = None
        else:
            k1_r, k2_r, k3_r, k4_r, k5_r = k1_right, k2_right, k3_right, k4_right, k5_right
    if k1_right is not None:
        RightEye[i_right, 0:5] = k1_right, k2_right, k3_right, k4_right, k5_right
    if eye_corner_right_x is not None:
        RightEye[i_right, 5:7] = eye_corner_right_x, eye_corner_right_y
    if x_eye_right is not None:
        RightEye[i_right, 7:12] = x_eye_right, y_eye_right, w_eye_right, x_eye_right+w_eye_right/2, y_eye_right+w_eye_right/2

    #     ------------------------------双眼---------------------------------
    if k1_left is not None and eye_corner_left_x is not None and k1_right is not None and eye_corner_right_x is not None:
        i_s += 1
        result_left_eye[i_left, :] = eye_corner_left_x, eye_corner_left_y, k1_left, k2_left, eye_corner_w_left_x, eye_corner_w_left_y
        result_right_eye[i_right, :] = eye_corner_right_x, eye_corner_right_y, k1_right, k2_right, eye_corner_w_right_x, eye_corner_w_right_y
        cv.ellipse(IM_left, [(int(k1_left), int(k2_left)), (k3_left, k4_left), k5_left], (0, 0, 255), 2)
        cv.ellipse(IM_right, [(int(k1_right), int(k2_right)), (k3_right, k4_right), k5_right], (0, 0, 255), 2)
        cv.circle(IM_left, (eye_corner_left_x, eye_corner_left_y), 8, (255, 0, 0), -1)
        cv.circle(IM_right, (eye_corner_right_x, eye_corner_right_y), 8, (255, 0, 0), -1)
        cv.circle(IM_left, (eye_corner_w_left_x, eye_corner_w_left_y), 8, (255, 0, 255), -1)
        cv.circle(IM_right, (eye_corner_w_right_x, eye_corner_w_right_y), 8, (255, 0, 255), -1)
        cv.arrowedLine(IM_left, (eye_corner_left_x, eye_corner_left_y), (int(k1_left), int(k2_left)), (0, 255, 0),
                       2, 0, 0, 0.2)
        cv.arrowedLine(IM_right, (eye_corner_right_x, eye_corner_right_y), (int(k1_right), int(k2_right)), (0, 255, 0),
                       2, 0, 0, 0.2)
        cv.arrowedLine(IM_left, (eye_corner_w_left_x, eye_corner_w_left_y), (int(k1_left), int(k2_left)), (255, 0, 255),
                       2, 1, 0, 0.1)
        cv.arrowedLine(IM_right, (eye_corner_w_right_x, eye_corner_w_right_y), (int(k1_right), int(k2_right)), (255, 0, 255),
                       2, 1, 0, 0.1)
        text_left = str((int(k1_left) - eye_corner_left_x, int(k2_left) - eye_corner_left_y))
        text_right = str((int(k1_right) - eye_corner_right_x, int(k2_right) - eye_corner_right_y))

        if x_eye_left is not None:
            cv2.rectangle(IM_left, (x_eye_left, y_eye_left),
                          (x_eye_left + w_eye_left, y_eye_left + w_eye_left), (255, 0, 0), 3)
        if x_eye_right is not None:
            cv2.rectangle(IM_right, (x_eye_right, y_eye_right),
                          (x_eye_right + w_eye_right, y_eye_right + w_eye_right), (255, 0, 0), 3)

    cv.putText(IM_left, 'Left Eye: ' + text_left, (50, 50), cv.FONT_HERSHEY_COMPLEX, 0.8,
               (0, 255, 0), 2)
    cv.putText(IM_right, 'Right Eye: ' + text_right, (50, 50), cv.FONT_HERSHEY_COMPLEX, 0.8,
               (0, 255, 0), 2)
    i_left += 1
    i_right += 1
    IM_treated = np.hstack((IM_left, IM_right))

    formatted_i = "{:03d}".format(i_left)
    print(formatted_i,k1_left,k1_right,eye_corner_w_left_x)
    print('——————————————————————————')
    path_IM_treated = path + '\\Pic_eye_treated\\' + formatted_i + '.jpg'
    cv.imwrite(path_IM_treated, IM_treated)

with open(path + '//' + name + '_result_left_eye.pkl', 'wb') as l:
    pickle.dump(result_left_eye, l)
with open(path + '//' + name + '_result_right_eye.pkl', 'wb') as l:
    pickle.dump(result_right_eye, l)

df_LeftEye = pd.DataFrame(LeftEye)
df_LeftEye.to_excel(path + '//LeftEye.xlsx', index=False)
df_RightEye = pd.DataFrame(RightEye)
df_RightEye.to_excel(path + '//RightEye.xlsx', index=False)

print('成功率：', i_s/i_left)
end_time = time.time()  # 记录程序结束时间
print("程序运行时间：", end_time - start_time, "秒")
