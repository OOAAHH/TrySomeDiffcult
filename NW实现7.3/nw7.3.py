# 在这里产生了分块计算的思路


import numpy as np
import pandas as pd


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
        """返回计算完成的矩阵 这个东西有什么用？"""
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
        self.row_factor = smallest_factor_greater_than_three(rows)
        # column
        self.col_factor = smallest_factor_greater_than_three(cols)

        # 第一拆分方法：按因数拆分行和列
        row_split = np.vsplit(self.matrix, rows // self.row_factor)
        self.small_matrices = [np.hsplit(sub_matrix, cols // self.col_factor) for sub_matrix in row_split]

        # 第二拆分方法：取相邻列的接缝区域
        self.seam_matrices_cols = []
        for i, row_matrices in enumerate(self.small_matrices):
            row_seams = []
            for j in range(len(row_matrices) - 1):
                # 取相邻列的接缝区域（每个小矩阵的一列组合）
                seam_matrix = np.concatenate((row_matrices[j][:, -1:], row_matrices[j + 1][:, :1]), axis=1)
                row_seams.append(seam_matrix)
            self.seam_matrices_cols.append(row_seams)

        # 第三拆分方法：取相邻行的接缝区域
        self.seam_matrices_rows = []
        for j in range(len(self.small_matrices[0])):
            col_seams = []
            for i in range(len(self.small_matrices) - 1):
                # 正确索引并进行切片
                top_matrix = self.small_matrices[i][j][-1:, :]  # 获取上方小矩阵的最后一行
                bottom_matrix = self.small_matrices[i + 1][j][:1, :]  # 获取下方小矩阵的前一行
                seam_matrix = np.concatenate((top_matrix, bottom_matrix), axis=0)
                col_seams.append(seam_matrix)
            self.seam_matrices_rows.append(col_seams)
        """
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
        """

    def trace_back_matrix(self):
        """
        对于第一拆分法产生的矩阵
        对输入的拆分后的矩阵开始回溯，输出拆分后的矩阵的局部路径；
            回溯的起点只有矩阵的最下一行和最右边一列的一部分，定义：对于给定大小的（i，j）的矩阵：左上角为（0，0），右下角为（i-1，j-1）
            对于这样的矩阵，回溯的起点有最下面一行的一部分（1，j-1）到（i-1，j-1），最右一列的一部分（i-1，1）到（i-1，j-2）；
            结果输出的方式是“实际坐标（起点）_方向_实际坐标,...,实际坐标_方向_实际坐标（终点）”

        实际坐标的计算：在一个大小为（i，j）的矩阵拆分后，新产生了两种坐标：
        第一是标记了存储了拆分后的矩阵的对象中的小矩阵的位置。
            例如，对于按4为因数进行拆分的8*8的矩阵，会产生4个新的小矩阵，标记为（ii,jj）矩阵。
        第二是小矩阵内部的坐标，标记为（iii,jjj）
        真实坐标的计算方式是（iii + (row_factor * ii), jjj + (col_factor * jj)）

        回溯的规则：从起点出发，只考虑起点的“左、上、左上/对角线”三个方向对应的三个值的大小关系，
            最大的值就是需要的回溯的方向，进行记录，允许多个值相等的情况。
        回溯的边界：小矩阵内部的坐标，标记为（iii,jjj）中任意值为0的时候，此为实际坐标（终点）。
                  实际坐标（终点）不作为回溯的起点使用。
                  实际坐标（i,j）中任意值为0则删除此路径。


        对于第二拆分法产生的矩阵，大小为（2，col_factor（列的因数））
            回溯的起点只有(1,1)到(1,col_factor-1)的一列的一部分
            定义：对于给定大小的（2，col_factor（列的因数））的矩阵：左上角为（0，0），右下角为（1，col_factor-1）
            结果输出的方式是“实际坐标（起点）_方向_实际坐标,...,实际坐标_方向_实际坐标（终点）”
            实际坐标的计算：在一个大小为（i，j）的矩阵经过第一和第二拆分法拆分后，新产生了四种坐标：
            第一是标记了存储了拆分后的矩阵的对象中的小矩阵的位置。
                例如，对于按4为因数进行拆分的8*8的矩阵，会产生4个新的小矩阵，标记为（ii,jj）矩阵。
            第二是小矩阵内部的坐标，标记为（iii,jjj）；
            第三是拼接后的小矩阵内部坐标，标记为（iii_col,jjj_col）
            第四是拼接后的小矩阵的标记坐标，如（0，0）小矩阵和（0，1）小矩阵拼接的矩阵，
                命名为((0,0),(0,1))矩阵。坐标形式为((ii,jj),(ii,jj+1))
            真实坐标的计算方式是：对于一个命名为((ii,jj),(ii,jj+1))的拼接后的小矩阵，
                为（iii_row + (ii+1) * （row_factor-1）, jjj_col + (jj*col_factor)
                    真实坐标的计算方式是：对于一个命名为((0,0),(0,1))的拼接后的小矩阵，
                        为（1 + (0+1) * （4-1）, jjj_col + (col_factor * jj)-1）


        对于第三拆分法产生的矩阵，大小为（2，row_factor（行的因数））
            回溯的起点只有(1,1)到(1,row_factor-1)的一列的一部分
            定义：对于给定大小的（2，row_factor（行的因数））的矩阵：左上角为（0，0），右下角为（1，row_factor-1）
            结果输出的方式是“实际坐标（起点）_方向_实际坐标,...,实际坐标_方向_实际坐标（终点）”
            实际坐标的计算：在一个大小为（i，j）的矩阵经过第一和第二拆分法拆分后，新产生了四种坐标：
            第一是标记了存储了拆分后的矩阵的对象中的小矩阵的位置。
                例如，对于按4为因数进行拆分的8*8的矩阵，会产生4个新的小矩阵，标记为（ii,jj）矩阵。
            第二是小矩阵内部的坐标，标记为（iii,jjj）；
            第三是拼接后的小矩阵内部坐标，标记为（iii_row,jjj_row）
            第四是拼接后的小矩阵的标记坐标，如（0，0）小矩阵和（1，0）小矩阵拼接的矩阵，
                命名为((0,0),(1,0))矩阵。坐标形式为((ii,jj),(ii+1,jj))
            真实坐标的计算方式是：对于一个命名为((ii,jj),(ii+1,jj))的拼接后的小矩阵，
                为（(iii_col + (ii)* row_factor)-1, (jjj_col + ((col_factor+1) * jj))-1）
        """

        
    
        # 辅助函数：计算真实的全局坐标
        def get_actual_coords(ii, jj, iii, jjj, row_factor, col_factor):
            return (iii + (row_factor * ii), jjj + (col_factor * jj))
    
        # 第一拆分法的回溯
        self.trace_first_split = []
        for ii, row_matrices in enumerate(self.small_matrices):
            for jj, matrix in enumerate(row_matrices):
                i_max, j_max = matrix.shape
                # 从最下面一行和最右一列开始回溯
                for iii in range(1, i_max):
                    # 回溯从最下行到上方
                    if matrix[iii, j_max-1] != 0:
                        current_trace = []
                        x, y = iii, j_max-1
                        while x > 0 and y > 0:
                            # 获取当前坐标
                            actual_start = get_actual_coords(ii, jj, x, y, self.row_factor, self.col_factor)
                            # 计算回溯方向
                            left = matrix[x, y-1] if y > 0 else -float('inf')
                            up = matrix[x-1, y] if x > 0 else -float('inf')
                            diagonal = matrix[x-1, y-1] if x > 0 and y > 0 else -float('inf')
                            max_value = max(left, up, diagonal)
                            # 确定回溯方向
                            if max_value == left:
                                y -= 1
                                direction = "左"
                            elif max_value == up:
                                x -= 1
                                direction = "上"
                            else:
                                x -= 1
                                y -= 1
                                direction = "左上"
                            # 记录路径
                            actual_end = get_actual_coords(ii, jj, x, y, self.row_factor, self.col_factor)
                            current_trace.append(f"{actual_start},{direction},{actual_end}")
                        # 将路径存储
                        self.trace_first_split.append(";".join(current_trace))
    
        # 第二拆分法的回溯
        self.trace_second_split = []
        for i, row_seams in enumerate(self.seam_matrices_cols):
            for j, seam_matrix in enumerate(row_seams):
                row_max, col_max = seam_matrix.shape
                # 只回溯指定列部分
                for col in range(1, col_max-1):
                    if seam_matrix[1, col] != 0:
                        current_trace = []
                        x, y = 1, col
                        while x > 0 and y > 0:
                            actual_start = (x + ((i+1) * self.row_factor) - 1, y + (self.col_factor * j) - 1)
                            # 计算回溯方向
                            left = seam_matrix[x, y-1] if y > 0 else -float('inf')
                            up = seam_matrix[x-1, y] if x > 0 else -float('inf')
                            diagonal = seam_matrix[x-1, y-1] if x > 0 and y > 0 else -float('inf')
                            max_value = max(left, up, diagonal)
                            # 确定回溯方向
                            if max_value == left:
                                y -= 1
                                direction = "左"
                            elif max_value == up:
                                x -= 1
                                direction = "上"
                            else:
                                x -= 1
                                y -= 1
                                direction = "左上"
                            actual_end = (x + ((i+1) * self.row_factor) - 1, y + (self.col_factor * j) - 1)
                            current_trace.append(f"{actual_start},{direction},{actual_end}")
                        self.trace_second_split.append(";".join(current_trace))
    
        # 第三拆分法的回溯
        self.trace_third_split = []
        for j, col_seams in enumerate(self.seam_matrices_rows):
            for i, seam_matrix in enumerate(col_seams):
                row_max, col_max = seam_matrix.shape
                # 只回溯指定行部分
                for row in range(1, row_max-1):
                    if seam_matrix[row, 1] != 0:
                        current_trace = []
                        x, y = row, 1
                        while x > 0 and y > 0:
                            actual_start = (x + (i * self.row_factor) - 1, y + (self.col_factor * j) - 1)
                            # 计算回溯方向
                            left = seam_matrix[x, y-1] if y > 0 else -float('inf')
                            up = seam_matrix[x-1, y] if x > 0 else -float('inf')
                            diagonal = seam_matrix[x-1, y-1] if x > 0 and y > 0 else -float('inf')
                            max_value = max(left, up, diagonal)
                            # 确定回溯方向
                            if max_value == left:
                                y -= 1
                                direction = "左"
                            elif max_value == up:
                                x -= 1
                                direction = "上"
                            else:
                                x -= 1
                                y -= 1
                                direction = "左上"
                            actual_end = (x + (i * self.row_factor) - 1, y + (self.col_factor * j) - 1)
                            current_trace.append(f"{actual_start},{direction},{actual_end}")
                        self.trace_third_split.append(";".join(current_trace))




