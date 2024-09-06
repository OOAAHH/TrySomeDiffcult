# 算法的构想基本能实现，但是要求矩阵尺寸不是质数，在拆矩阵方式中可以革新。
# 目前导出的结果是路径，并非比对的结果


import numpy as np
import pandas as pd
from joblib import Parallel, delayed


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

        # 获取行数和列数
        self.rows, self.cols = self.matrix.shape

        # 计算大于 3 的最小因数并存储为类的属性
        self.row_factor = self.smallest_factor_greater_than_three(self.rows)
        self.col_factor = self.smallest_factor_greater_than_three(self.cols)

    def smallest_factor_greater_than_three(self, n):
        """
        找到大于 3 的最小因数，如果没有合适的因数，则返回 n 自己
        """
        for i in range(4, n + 1):
            if n % i == 0:
                return i
        return n  # 如果没有因数，返回 n 自己


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

        # 第一拆分方法：按因数拆分行和列
        row_split = np.vsplit(self.matrix, self.rows // self.row_factor)
        self.small_matrices = [np.hsplit(sub_matrix, self.cols // self.col_factor) for sub_matrix in row_split]

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
    # 回溯

    def trace_back_from_point(self, matrix, i_start, j_start):
        """
        从给定起点 (i_start, j_start) 开始回溯，返回所有可能的路径
        使用迭代方式代替递归，避免递归深度限制
        """
        stack = [(i_start, j_start, [(i_start, j_start)])]  # 初始化堆栈
        all_paths = []
    
        while stack:
            i, j, path = stack.pop()
            
            if i == 0 or j == 0:
                all_paths.append(path)
                continue
    
            # 比较左、上、左上的值，找出所有相等的方向
            neighbors = [(i - 1, j), (i, j - 1), (i - 1, j - 1)]  # 上、左、左上
            valid_neighbors = [(ni, nj) for ni, nj in neighbors if ni >= 0 and nj >= 0]
    
            # 找出最大值
            max_value = max(matrix[ni][nj] for ni, nj in valid_neighbors)
    
            # 找到所有与最大值相等的邻居方向
            next_steps = [(ni, nj) for ni, nj in valid_neighbors if matrix[ni][nj] == max_value]
    
            for ni, nj in next_steps:
                # 将新路径添加到堆栈
                stack.append((ni, nj, path + [(ni, nj)]))
    
        return all_paths


    def calculate_first_split_coordinates(self, ii, jj, iii, jjj, row_factor, col_factor):
        """
        第一拆分法：计算小矩阵的真实坐标
        """
        return (iii + ii * self.col_factor, jjj + jj * self.row_factor)

    def calculate_second_split_coordinates(self, ii, jj, iii_row, jjj_col, row_factor, col_factor):
        """
        第二拆分法：计算拼接后的小矩阵的真实坐标
        """
        return (iii_row + ii * row_factor, jjj_col - 1 + (jj + 1) * col_factor)

    def calculate_third_split_coordinates(self, ii, jj, iii_row, jjj_col, row_factor, col_factor):
        """
        第三拆分法：计算拼接后的小矩阵的真实坐标
        """
        return (iii_row + (ii + 1) * row_factor - 1, jjj_col + col_factor * jj)



    def process_batch_of_matrices(self, batch, row_factor, col_factor):
        """
        批量处理多个小矩阵，返回回溯路径
        """
        batch_results = []
        for ii, jj, small_matrix in batch:
            paths = []
            # 回溯起点：最后一列的部分行
            for iii in range(1, small_matrix.shape[0]):
                paths += self.trace_back_from_point(small_matrix, iii, small_matrix.shape[1] - 1)
            # 回溯起点：最后一行的部分列
            for jjj in range(1, small_matrix.shape[1]):
                paths += self.trace_back_from_point(small_matrix, small_matrix.shape[0] - 1, jjj)

            # 转换为实际坐标
            real_paths = []
            for path in paths:
                real_path = []
                for iii, jjj in path:
                    real_coord = self.calculate_first_split_coordinates(ii, jj, iii, jjj, row_factor, col_factor)
                    real_path.append(real_coord)
                real_paths.append(real_path)

            batch_results.append(real_paths)
        return batch_results

    def process_second_split_batch(self, batch, row_factor, col_factor):
        """
        批量处理第二拆分法的矩阵
        """
        batch_results = []
        for i, j, seam_matrix in batch:
            paths = []
            # 回溯起点：从 (1, 1) 到 (因数-1, 1)
            for iii_row in range(1, seam_matrix.shape[0]):
                paths += self.trace_back_from_point(seam_matrix, iii_row, 1)

            # 转换为实际坐标
            real_paths = []
            for path in paths:
                real_path = []
                for iii_row, jjj_col in path:
                    real_coord = self.calculate_second_split_coordinates(i, j, iii_row, jjj_col, row_factor, col_factor)
                    real_path.append(real_coord)
                real_paths.append(real_path)

            batch_results.append(real_paths)
        return batch_results

    def process_third_split_batch(self, batch, row_factor, col_factor):
        """
        批量处理第三拆分法的矩阵
        """
        batch_results = []
        for i, j, seam_matrix in batch:
            paths = []
            # 回溯起点：从 (1, 1) 到 (1, 因数-1)
            for jjj_row in range(1, seam_matrix.shape[1]):
                paths += self.trace_back_from_point(seam_matrix, 1, jjj_row)

            # 转换为实际坐标
            real_paths = []
            for path in paths:
                real_path = []
                for iii_row, jjj_col in path:
                    real_coord = self.calculate_third_split_coordinates(i, j, iii_row, jjj_col, row_factor, col_factor)
                    real_path.append(real_coord)
                real_paths.append(real_path)

            batch_results.append(real_paths)
        return batch_results

    def trace_first_split(self, batch_size=100000):
        """
        第一拆分法回溯，使用 joblib 进行批量并行处理
        """
        # 将所有小矩阵分批
        tasks = []
        for ii, row_matrices in enumerate(self.small_matrices):
            for jj, small_matrix in enumerate(row_matrices):
                tasks.append((ii, jj, small_matrix))

        # 创建批次
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

        # 使用并行处理批次，并显式传递 row_factor 和 col_factor
        results = Parallel(n_jobs=-1, backend="loky")(delayed(self.process_batch_of_matrices)(batch, self.row_factor, self.col_factor) for batch in batches)

        # 合并结果
        first_traceback_paths = [path for result in results for batch in result for path in batch]
        return first_traceback_paths

    def trace_second_split(self, batch_size=100000):
        """
        第二拆分法回溯，使用 joblib 并行化处理
        """
        # 将所有 seam_matrices_cols 分批
        tasks = []
        for i, row_seams in enumerate(self.seam_matrices_cols):
            for j, seam_matrix in enumerate(row_seams):
                tasks.append((i, j, seam_matrix))

        # 创建批次
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

        # 使用并行处理批次
        results = Parallel(n_jobs=-1, backend="loky")(delayed(self.process_second_split_batch)(batch, self.row_factor, self.col_factor) for batch in batches)

        # 合并结果
        second_traceback_paths = [path for result in results for batch in result for path in batch]
        return second_traceback_paths

    def trace_third_split(self, batch_size=100000):
        """
        第三拆分法回溯，使用 joblib 并行化处理
        """
        # 将所有 seam_matrices_rows 分批
        tasks = []
        for i, col_seams in enumerate(self.seam_matrices_rows):
            for j, seam_matrix in enumerate(col_seams):
                tasks.append((i, j, seam_matrix))

        # 创建批次
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

        # 使用并行处理批次
        results = Parallel(n_jobs=-1, backend="loky")(delayed(self.process_third_split_batch)(batch, self.row_factor, self.col_factor) for batch in batches)

        # 合并结果
        third_traceback_paths = [path for result in results for batch in result for path in batch]
        return third_traceback_paths


    def trace_back_matrix(self):
        """
        对所有拆分方法的矩阵进行回溯，并将结果存储到类变量中
        """
        self.first_traceback_paths = self.trace_first_split()
        self.second_traceback_paths = self.trace_second_split()
        self.third_traceback_paths = self.trace_third_split()

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


# 测试该类


if __name__ == "__main__":
    # 示例输入
    A = "atttattttatttaccttaaaactagtattttaataaataaaattattttttcttatttatttaattataaaaacctcattatttttttaaaactctatttatttttaaataaaatattttttaatttattttacgaaaaatgagatattaatatataatgactgaaatgtaaagattttacaagtataagtattaaatgagtaattgaaaaaaagtttaatcttataaataatgctttttttccttctccatatatcatgaacacaataatacacaatccctttgaaattaaatacactttaggacaaaacatgaccccttaaaaagagatagtacctcccattccagttacatttgtgacaaagttagaaagaaggagaaagaaagaagatggcaggaaagaaaatcccacatgtgttgctgaattctggacacaaaatgccagttataggcatgggaacttcagtggagaatcgtccatcaaatgagacccttgcttcaatctatgttgaagccattgaggttggttaccgtcattttgacactgctgcagtgtatggaacagaggaagccataggcctggccgtggccaatgccatagaaaaaggcctaataaagagtagagatgaagttttcatcacttcaaaaccttggaacacagatgcacgccgtgatcttattgtcccagctctcaagaccacattaaagtacatataatctccattctcacattgtttctaatgatcttaatattgtgtttgtgctcactttttaattttattttattttgtggacaaaacttagatgcggtttcatataagcttagtgttgttctacaacttgaataggagtttcacctcggtaatacacattaacatttgatggaaacctttattacacggattacctgagtgaaactccaattctgattggagaacagcgccaaaagcactaatagaactacatctaagttttgtaattgcttttattatttttctataactactaatgttgtttgagtactttatgtgccacacaatgttgtttaggaagctggggacgcagtatgtggatctttatctgattcattggccagtgaggctgagacatgatctttaaaatccaactgtttttaccaaggaagattttctgccctttgatatagaagggacatggaaagctatggaagagtgttacaagttgggattagcaaagtccattggcatatgcaattatggtgtcaaaaaactcaccaaactcttggaaatagccacctttcctcctgcagtcaatcaggtaccataatttgattactccagagatgtatatttggtttggcacatagttattatgatggtcttggacttcttggtactcatgttatgatgttgaattattgttttttcaggtggaaatgaacccatcttggcagcaaggaaaactcagggag"
    B = "actccctgagttttccttgctgccaagatgggttcatttccacctgaaaaaacaataattcaacatcataacatgagtaccaagaagtccaagaccatcataataactatgtgccaaaccaaatatacatctctggagtaatcaaattatggtacctgattgactgcaggaggaaaggtggctatttccaagagtttggtgagttttttgacaccataattgcatatgccaatggactttgctaatcccaacttgtaacactcttccatagctttccatgtcccttctatatcaaagggcagaaaatcttccttggtaaaaacagttggattttaaagatcatgtctcagcctcactggccaatgaatcagataaagatccacatactgcgtccccagcttcctaaacaacattgtgtggcacataaagtactcaaacaacattagtagttatagaaaaataataaaagcaattacaaaacttagatgtagttctattagtgcttttggcgctgttctccaatcagaattggagtttcactcaggtaatccgtgtaataaaggtttccatcaaatgttaatgtgtattaccgaggtgaaactcctattcaagttgtagaacaacactaagcttatatgaaaccgcatctaagttttgtccacaaaataaaataaaattaaaaagtgagcacaaacacaatattaagatcattagaaacaatgtgagaatggagattatatgtactttaatgtggtcttgagagctgggacaataagatcacggcgtgcatctgtgttccaaggttttgaagtgatgaaaacttcatctctactctttattaggcctttttctatggcattggccacggccaggcctatggcttcctctgttccatacactgcagcagtgtcaaaatgacggtaaccaacctcaatggcttcaacatagattgaagcaagggtctcatttgatggacgattctccactgaagttcccatgcctataactggcattttgtgtccagaattcagcaacacatgtgggattttctttcctgccatcttctttctttctccttctttctaactttgtcacaaatgtaactggaatgggaggtactatctctttttaaggggtcatgttttgtcctaaagtgtatttaatttcaaagggattgtgtattattgtgttcatgatatatggagaaggaaaaaaagcattatttataagattaaactttttttcaattactcatttaatacttatacttgtaaaatctttacatttcagtcattatatattaatatctcatttttcgtaaaataaattaaaaaatattttatttaaaaataaatagagttttaaaaaaataatgaggtttttataattaaataaataagaaaaaataattttatttattaaaatactagttttaaggtaaataaaataaa"

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
