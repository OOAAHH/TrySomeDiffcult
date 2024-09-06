import numpy as np
import pandas as pd
import concurrent.futures


class SequenceAlignment:
    def __init__(self, A, B, a=-2, b=-1, c=3):
        """
        初始化类
        A: 序列1
        B: 序列2
        a: gap 罚分
        b: mismatch 罚分
        c: match 奖励
        """
        self.A = A
        self.B = B
        self.a = a
        self.b = b
        self.c = c
        self.len_A = len(A)
        self.len_B = len(B)
        # 创建一个 (len_A + 1) x (len_B + 1) 的矩阵
        self.matrix = np.zeros((self.len_A + 1, self.len_B + 1))

    def initialize_matrix(self):
        """初始化矩阵的第一行和第一列"""
        for i in range(self.len_A + 1):
            self.matrix[i][0] = i * self.a
        for j in range(self.len_B + 1):
            self.matrix[0][j] = j * self.a

    def calculate_matrix(self):
        """动态规划填充矩阵"""
        for i in range(1, self.len_A + 1):
            for j in range(1, self.len_B + 1):
                if self.A[i - 1] == self.B[j - 1]:
                    # 如果字符匹配，使用 match 奖励
                    self.matrix[i][j] = self.matrix[i - 1][j - 1] + self.c
                else:
                    # 如果字符不匹配，使用 mismatch 罚分和 gap 罚分
                    self.matrix[i][j] = max(self.matrix[i - 1][j] + self.a,  # gap in B
                                            self.matrix[i][j - 1] + self.a,  # gap in A
                                            self.matrix[i - 1][j - 1] + self.b)  # mismatch

    def get_matrix(self):
        """返回计算完成的矩阵"""
        return self.matrix

    def print_matrix(self):
        """打印矩阵"""
        print(pd.DataFrame(self.matrix))

    def split_matrix(self):
        """
        处理矩阵的主体
            第一拆分矩阵法
                用大于三的最小因数对矩阵进行拆分为多个小矩阵
        处理矩阵的接缝区域
            第二拆分矩阵法
                在第一拆分矩阵的基础上，取每行各个小矩阵相邻的两列组成新矩阵。在命名上，（0,0）矩阵和（0,1）矩阵形成的新矩阵叫((0,0),(0,1))
            第三拆分矩阵法
                在第一拆分矩阵的基础上，取每列各个小矩阵相邻的两行组成新矩阵。在命名上，(0,0)和(1,0)两个小矩阵形成的新矩阵叫((0,0),(1,0))
        三种矩阵都存储到各自的变量中，便于在接下来的函数中计算
        """

        # 找到大于 3 的最小因数
        def smallest_factor_greater_than_three(n):
            for i in range(4, n + 1):
                if n % i == 0:
                    return i
            return n  # 如果没有因数，返回 n 自己

        # 获取行数和列数
        rows, cols = self.matrix.shape

        # 找到大于 3 的最小因数 所以这个方法是适合长序列的 也是导向稀疏搜索的必由之路
        # row
        row_factor = smallest_factor_greater_than_three(rows)
        # column
        col_factor = smallest_factor_greater_than_three(cols)

        # 第一拆分方法：按因数拆分行和列
        row_split = np.vsplit(self.matrix, rows // row_factor)
        self.small_matrices = [np.hsplit(sub_matrix, cols // col_factor) for sub_matrix in row_split]

        # 第二拆分方法：取相邻列的接缝区域
        self.seam_matrices_cols = []
        for i, row_matrices in enumerate(self.small_matrices):
            row_seams = []
            for j in range(len(row_matrices) - 1):
                # 取相邻列的接缝区域（每个小矩阵的一列组合）
                seam_matrix = np.concatenate((row_matrices[j][:, -1:], row_matrices[j + 1][:, :1]), axis=1)
                row_seams.append(seam_matrix)
            self.seam_matrices_cols.append(row_seams)

        # 第三拆分方法：取相邻列的小矩阵进行垂直拼接
        self.seam_matrices_rows = []
        # 遍历列索引
        for i in range(len(self.small_matrices) - 1):  # 行的数量 - 1
            row_seams = []
            for j in range(len(self.small_matrices[0])):  # 列的数量
                # 获取相邻行的两个小矩阵
                top_matrix = self.small_matrices[i][j]  # 当前小矩阵
                bottom_matrix = self.small_matrices[i + 1][j]  # 垂直下方的小矩阵

                # 拼接相邻行的小矩阵的最后一行和第一行
                seam_matrix = np.concatenate((top_matrix[-1:, :], bottom_matrix[:1, :]), axis=0)

                # 将拼接后的结果添加到 row_seams 列表
                row_seams.append(seam_matrix)

            # 将每一行的拼接结果添加到 seam_matrices_rows
            self.seam_matrices_rows.append(row_seams)

        '''  
        # 输出拆分后的矩阵，打印第一种拆分法的小矩阵
        print("First split (small matrices):")
        for i, row_matrices in enumerate(self.small_matrices):
            for j, small_matrix in enumerate(row_matrices):
                print(f"Sub-matrix ({i}, {j}):\n", small_matrix)

        # 输出第二种拆分法的接缝区域
        print("\nSecond split (seam matrices by columns):")
        for i, row_seams in enumerate(self.seam_matrices_cols):
            for j, seam_matrix in enumerate(row_seams):
                print(f"Seam matrix (({i},{j}),({i},{j + 1})):\n", seam_matrix)

        # 输出第三种拆分法的接缝区域
        print("\nThird split (seam matrices by rows):")
        for i, col_seams in enumerate(self.seam_matrices_rows):
            for j, seam_matrix in enumerate(col_seams):
                print(f"Seam matrix (({i},{j}),({i + 1},{j})):\n", seam_matrix)
        '''

    def trace_back_matrix(self):
        """
        对第一、第二、第三拆分法产生的矩阵进行回溯，并将结果存储在不同的变量中。
        支持多方向回溯（当多个方向的值相等时，沿多个方向回溯）。
        """

        # 找到大于 3 的最小因数
        def smallest_factor_greater_than_three(n):
            for i in range(4, n + 1):
                if n % i == 0:
                    return i
            return n  # 如果没有因数，返回 n 自己

        # 获取行数和列数
        rows, cols = self.matrix.shape

        # 找到大于 3 的最小因数 所以这个方法是适合长序列的 也是导向稀疏搜索的必由之路
        # row
        row_factor = smallest_factor_greater_than_three(rows)
        # column
        col_factor = smallest_factor_greater_than_three(cols)

        # 计算第一拆分法的小矩阵的真实坐标
        def calculate_first_split_coordinates(ii, jj, iii, jjj, row_factor, col_factor):
            return (iii + ii * col_factor, jjj + jj * row_factor)

        # 计算第二拆分法的小矩阵的真实坐标
        def calculate_second_split_coordinates(ii, jj, iii_row, jjj_col, row_factor, col_factor):
            return (iii_row + ii * row_factor, jjj_col - 1 + (jj + 1) * col_factor)

        # 计算第三拆分法的小矩阵的真实坐标
        def calculate_third_split_coordinates(ii, jj, iii_row, jjj_col, row_factor, col_factor):
            return (iii_row + (ii + 1) * row_factor - 1, jjj_col + col_factor * jj)

        # 从给定起点 (i_start, j_start) 开始回溯，返回所有可能的路径
        def trace_back_from_point(matrix, i_start, j_start):
            def backtrack(i, j):
                if i == 0 or j == 0:
                    return [[(i, j)]]

                # 当前点
                path = [(i, j)]

                # 比较左、上、左上的值，找出所有相等的方向
                neighbors = [(i - 1, j), (i, j - 1), (i - 1, j - 1)]  # 上、左、左上
                valid_neighbors = [(ni, nj) for ni, nj in neighbors if ni >= 0 and nj >= 0]

                # 找出最大值
                max_value = max(matrix[ni, nj] for ni, nj in valid_neighbors)

                # 找到所有与最大值相等的邻居方向
                next_steps = [(ni, nj) for ni, nj in valid_neighbors if matrix[ni, nj] == max_value]

                # 如果有多个相等的方向，沿每个方向回溯
                all_paths = []
                for ni, nj in next_steps:
                    sub_paths = backtrack(ni, nj)
                    for sub_path in sub_paths:
                        all_paths.append(path + sub_path)

                return all_paths

            # 从给定起点开始回溯，得到所有路径
            return backtrack(i_start, j_start)

        # 第一拆分法回溯
        def trace_first_split():
            first_traceback_paths = []
            for ii, row_matrices in enumerate(self.small_matrices):
                for jj, small_matrix in enumerate(row_matrices):
                    paths = []
                    # 回溯起点：最后一列的部分行
                    for iii in range(1, small_matrix.shape[0]):
                        paths += trace_back_from_point(small_matrix, iii, small_matrix.shape[1] - 1)
                    # 回溯起点：最后一行的部分列
                    for jjj in range(1, small_matrix.shape[1]):
                        paths += trace_back_from_point(small_matrix, small_matrix.shape[0] - 1, jjj)

                    # 转换为实际坐标
                    for path in paths:
                        real_path = []
                        for iii, jjj in path:
                            real_coord = calculate_first_split_coordinates(ii, jj, iii, jjj, row_factor, col_factor)
                            real_path.append(real_coord)

                        # 存储路径
                        first_traceback_paths.append(real_path)

            return first_traceback_paths

        # 第二拆分法回溯
        def trace_second_split():
            second_traceback_paths = []
            for i, row_seams in enumerate(self.seam_matrices_cols):
                for j, seam_matrix in enumerate(row_seams):
                    paths = []
                    # 回溯起点：从 (1, 1) 到 (因数-1, 1)
                    for iii_row in range(1, seam_matrix.shape[0]):
                        paths += trace_back_from_point(seam_matrix, iii_row, 1)

                    # 拼接后的实际坐标计算
                    for path in paths:
                        real_path = []
                        for iii_row, jjj_col in path:
                            real_coord = calculate_second_split_coordinates(i, j, iii_row, jjj_col, row_factor,
                                                                            col_factor)
                            real_path.append(real_coord)

                        second_traceback_paths.append(real_path)

            return second_traceback_paths

        # 第三拆分法回溯
        def trace_third_split():
            third_traceback_paths = []
            for i, col_seams in enumerate(self.seam_matrices_rows):
                for j, seam_matrix in enumerate(col_seams):
                    paths = []
                    # 回溯起点：从 (1, 1) 到 (1, 因数-1)
                    for jjj_row in range(1, seam_matrix.shape[1]):
                        paths += trace_back_from_point(seam_matrix, 1, jjj_row)

                    # 拼接后的实际坐标计算
                    for path in paths:
                        real_path = []
                        for iii_row, jjj_col in path:
                            real_coord = calculate_third_split_coordinates(i, j, iii_row, jjj_col, row_factor,
                                                                           col_factor)
                            real_path.append(real_coord)

                        third_traceback_paths.append(real_path)

            return third_traceback_paths

        ### 计算并存储结果 ###
        self.first_traceback_paths = trace_first_split()
        self.second_traceback_paths = trace_second_split()
        self.third_traceback_paths = trace_third_split()

        """       
        # 输出回溯路径，存储在不同的变量中，便于后续使用
        print("First traceback paths (from first split):")
        for path in self.first_traceback_paths:
            print(" -> ".join([f"{coord}" for coord in path]))

        print("\nSecond traceback paths (from second split):")
        for path in self.second_traceback_paths:
            print(" -> ".join([f"{coord}" for coord in path]))

        print("\nThird traceback paths (from third split):")
        for path in self.third_traceback_paths:
            print(" -> ".join([f"{coord}" for coord in path]))
        """

    def merge_paths(self):
        """
        合并路径，路径的起点和终点确定为 (i, j) 到 (1, 1)，
        允许路径复用：如果路径是其他完整路径的组件，则可以复用。
        """

        # 合并所有路径到一个列表
        all_paths = self.first_traceback_paths + self.second_traceback_paths + self.third_traceback_paths

        # 结果存储变量
        merged_paths = []

        # 创建一个字典，用于根据起点和终点快速查找路径
        start_dict = {}
        end_dict = {}

        # 初始化字典，存储以每条路径的起点和终点为键的路径
        for path in all_paths:
            start = path[0]
            end = path[-1]

            if start not in start_dict:
                start_dict[start] = []
            if end not in end_dict:
                end_dict[end] = []

            start_dict[start].append(path)
            end_dict[end].append(path)

        def find_and_merge_path(path):
            """
            递归合并路径，允许多条路径复用相同的部分。
            """
            merged_path = list(path)  # 当前路径的副本
            current_end = merged_path[-1]

            # 寻找与当前路径的终点匹配的路径，并允许复用路径
            while current_end in start_dict:
                next_paths = [p for p in start_dict[current_end]]
                if not next_paths:
                    break  # 没有更多路径可合并

                # 对每个可能的路径进行递归合并
                for next_path in next_paths:
                    merged_path += next_path[1:]  # 去除重复的起点，合并路径
                    current_end = next_path[-1]  # 更新当前终点

            return merged_path

        # 开始合并所有路径
        for path in all_paths:
            merged_path = find_and_merge_path(path)
            merged_paths.append(merged_path)

        # 去除重复的完整路径
        unique_merged_paths = []
        seen = set()
        for path in merged_paths:
            path_tuple = tuple(path)
            if path_tuple not in seen:
                unique_merged_paths.append(path)
                seen.add(path_tuple)

        '''     
        # 打印输出合并后的路径
        print("Merged paths:")
        for path in unique_merged_paths:
            print(" -> ".join([f"{coord}" for coord in path]))
        '''

        # 返回合并后的路径
        return unique_merged_paths


# 测试该类


if __name__ == "__main__":
    # 示例输入
    A = "atttattttatttaccttaaaactagtattttaataaataaaattattttttcttatttatttaattataaaaacctcattatttttttaaaactctatttatttttaaataaaatattttttaatttattttacgaaaaatgagatattaatatataatgactgaaatgtaaagattttacaagtataagtattaaatgagtaattgaaaaaaagtttaatcttataaataatgctttttttccttctccatatatcatgaacacaataatacacaatccctttgaaattaaatacactttaggacaaaacatgaccccttaaaaagagatagtacctcccattccagttacatttgtgacaaagttagaaagaaggagaaagaaagaagatggcaggaaagaaaatcccacatgtgttgctgaattctggacacaaaatgccagttataggcatgggaacttcagtggagaatcgtccatcaaatga"
    B = "actccctgagttttccttgctgccaagatgggttcatttccacctgaaaaaacaataattcaacatcataacatgagtaccaagaagtccaagaccatcataataactatgtgccaaaccaaatatacatctctggagtaatcaaattatggtacctgattgactgcaggaggaaaggtggctatttccaagagtttggtgagttttttgacaccataattgcatatgccaatggactttgctaatcccaacttgtaacactcttccatagctttccatgtcccttctatatcaaagggcagaaaatcttccttggtaaaaacagttggattttaaagatcatgtctcagcctcactggccaatgaatcagataaagatccacatactg"

    # 初始化对象
    aligner = SequenceAlignment(A, B, a=-1, b=-6, c=10)

    # 初始化矩阵
    aligner.initialize_matrix()

    # 计算打分矩阵
    aligner.calculate_matrix()

    # 输出最终矩阵
    aligner.print_matrix()

    # 返回矩阵
    aligner.get_matrix()

    # 拆分矩阵
    aligner.split_matrix()

    # 回溯矩阵
    aligner.trace_back_matrix()

    # 拼接
    aligner.merge_paths()

