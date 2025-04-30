import os
import glob
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.ndimage import median_filter

# 数据预处理相关函数
def extract_features_from_pkl(path):
    """
    从指定路径下的 pkl 文件中提取左瞳孔和右瞳孔顺时针旋转的数据，并进行处理和特征提取。
    :param path: pkl 文件所在的路径
    :return: 处理后的特征数组
    """
    left_movein_clockwise = []
    left_moveout_clockwise = []
    right_moveout_clockwise = []
    right_movein_clockwise = []
    pkl_files = glob.glob(os.path.join(path, '*.pkl'))

    for i, pkl_file in enumerate(pkl_files, start=1):
        name = os.path.basename(pkl_file)
        with open(pkl_file, 'rb') as file:
            data = pickle.load(file)
            # 左瞳孔顺时针旋转
            if i % 3 == 1 and name[-21:-20] == '1':
                for index, row in enumerate(data):
                    if 144 <= index <= 264:
                        left_movein_clockwise.append(row)
                    if 334 <= index <= 457:
                        left_moveout_clockwise.append(row)
            # 右瞳孔顺时针旋转
            elif i % 3 == 0 and name[-22:-21] == '1':
                for index, row in enumerate(data):
                    if 144 <= index <= 264:
                        right_moveout_clockwise.append(row)
                    if 334 <= index <= 457:
                        right_movein_clockwise.append(row)

    def extract_single_column_data(move_data):
        return [row[2] for row in move_data]

    left_movein_data = extract_single_column_data(left_movein_clockwise)
    left_moveout_data = extract_single_column_data(left_moveout_clockwise)
    right_moveout_data = extract_single_column_data(right_moveout_clockwise)
    right_movein_data = extract_single_column_data(right_movein_clockwise)

    speed_left_in = split_and_filter_data1(left_movein_data)
    speed_right_out = split_and_filter_data1(right_moveout_data)
    speed_left_out = split_and_filter_data3(left_moveout_data)
    speed_right_in = split_and_filter_data3(right_movein_data)

    combined_array = np.stack(
        (speed_left_in, speed_right_out, speed_left_out, speed_right_in),
        axis=-1)
    speed_all = combined_array.reshape(40, 32)
    return speed_all

def filter_and_calculate_stats1(subarray):
    """
    对输入的子数组进行数据平滑处理，并计算相关统计特征。
    :param subarray: 输入的子数组
    :return: 计算得到的统计特征
    """
    smoothed = median_filter(subarray, size=5).ravel()
    mean_val = np.mean(smoothed)
    std_val = np.std(smoothed)
    min_val = np.min(smoothed)
    max_val = np.max(smoothed)

    diffs = [smoothed[i + 1] - smoothed[i] for i in range(len(smoothed) - 1)]
    filtered_diffs = [diff for diff in diffs if diff > 0]
    sorted_diffs = sorted(filtered_diffs)
    cut_off_min = 3
    cut_off_max = 3
    filtered_further = sorted_diffs[cut_off_min:-cut_off_max]

    if not filtered_further:
        return 0, 0, 0, 0, mean_val, std_val, min_val, max_val

    mean_diff = np.mean(filtered_further)
    std_diff = np.std(filtered_further)
    final_filtered = [diff for diff in filtered_further if
                      mean_diff - 3 * std_diff <= diff <= mean_diff + 3 * std_diff]
    final_diffs = [final_filtered[i + 1] - final_filtered[i] for i in range(len(final_filtered) - 1)]
    average_value = np.mean(final_diffs)
    std_dev1 = np.std(final_diffs)
    return average_value, std_val, mean_diff, std_dev1, mean_val, std_val, min_val, max_val

def filter_and_calculate_stats2(subarray):
    """
    对输入的子数组进行数据平滑处理，并计算相关统计特征。
    与 filter_and_calculate_stats1 不同的是差分计算方式。
    :param subarray: 输入的子数组
    :return: 计算得到的统计特征
    """
    smoothed = median_filter(subarray, size=5).ravel()
    mean_val = np.mean(smoothed)
    std_val = np.std(smoothed)
    min_val = np.min(smoothed)
    max_val = np.max(smoothed)

    diffs = [smoothed[i] - smoothed[i + 1] for i in range(len(smoothed) - 1)]
    filtered_diffs = [diff for diff in diffs if diff > 0]
    sorted_diffs = sorted(filtered_diffs)
    cut_off_min = 3
    cut_off_max = 3
    filtered_further = sorted_diffs[cut_off_min:-cut_off_max]

    if not filtered_further:
        return 0, 0, 0, 0, mean_val, std_val, min_val, max_val

    mean_diff = np.mean(filtered_further)
    std_diff = np.std(filtered_further)
    final_filtered = [diff for diff in filtered_further if
                      mean_diff - 3 * std_diff <= diff <= mean_diff + 3 * std_diff]
    final_diffs = [final_filtered[i + 1] - final_filtered[i] for i in range(len(final_filtered) - 1)]
    average_value = np.mean(final_diffs)
    std_dev1 = np.std(final_diffs)
    return average_value, std_val, mean_diff, std_dev1, mean_val, std_val, min_val, max_val

def split_and_filter_data1(place):
    """
    将输入的数组按固定长度分割成多个子数组，并调用 filter_and_calculate_stats1 处理。
    :param place: 输入的数组
    :return: 处理后的结果数组
    """
    num_elements_per_subarray = 121
    num_subarrays = len(place) // num_elements_per_subarray
    subarrays = [place[i * num_elements_per_subarray:(i + 1) * num_elements_per_subarray]
                 for i in range(num_subarrays)]
    result = [filter_and_calculate_stats1(subarray) for subarray in subarrays]
    return np.array(result)

def split_and_filter_data3(place):
    """
    将输入的数组按固定长度分割成多个子数组，并调用 filter_and_calculate_stats2 处理。
    :param place: 输入的数组
    :return: 处理后的结果数组
    """
    num_elements_per_subarray = 124
    num_subarrays = len(place) // num_elements_per_subarray
    subarrays = [place[i * num_elements_per_subarray:(i + 1) * num_elements_per_subarray]
                 for i in range(num_subarrays)]
    result = [filter_and_calculate_stats2(subarray) for subarray in subarrays]
    return np.array(result)

# 留一法交叉验证函数
def leave_one_out_validation(X, y, classifier):
    """
    进行留一法交叉验证，并计算分类准确率。
    :param X: 特征矩阵
    :param y: 标签向量
    :param classifier: 分类器对象
    :return: 预测结果和整体准确率
    """
    n_samples = len(y)
    results = np.zeros_like(y)
    correct_normal = 0
    correct_patient = 0
    patient_count = 0

    for i in range(n_samples):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict([X[i, :]])
        results[i] = y_pred

        if i < 40:
            if y_pred == y[i]:
                correct_normal += 1
        else:
            patient_count += 1
            if y_pred == y[i]:
                correct_patient += 1

    accuracy_normal = correct_normal / 41
    accuracy_patient = correct_patient / (n_samples - 41)
    overall_accuracy = accuracy_score(results, y)

    print('正常人分类准确率：', accuracy_normal)
    print('患者分类准确率：', accuracy_patient)
    print("Accuracy:", overall_accuracy)
    print("\nClassification Report:")
    print(classification_report(results, y))
    return results, overall_accuracy

if __name__ == "__main__":
    # 数据路径
    path_normal = 'D:\\斜视仪器算法\\Y\\Y\\normal\\'
    path_patient = 'D:\\斜视仪器算法\\Y\\Y\\patient\\'

    # 提取特征
    array1 = extract_features_from_pkl(path_normal)
    array2 = extract_features_from_pkl(path_patient)
    X = np.vstack((array1, array2))

    # 读取斜视度数数据
    df = pd.read_excel('D:\\斜视仪器算法\\Y\\Y\\Data_patient_strabismus.xls')
    degree_33cm = df['33cm'].values
    degree_6m = df['6m'].values
    y = np.concatenate((np.zeros(40), degree_33cm))

    # 标签分类
    labels = np.zeros_like(y)
    labels[y > 0] = 1
    labels[y == 0] = 2
    labels[y < 0] = 1

    # 创建随机森林分类器
    rf_classifier = RandomForestClassifier(n_estimators=90, random_state=48)

    print('采用留一法，分析分类的准确率')
    leave_one_out_validation(X, labels, rf_classifier)