import numpy as np
import pandas as pd
from joblib import Parallel, delayed

class SequenceAlignment:
    def __init__(self, A, B, a=-2, b=-1, c=3):
        self.A = A
        self.B = B
        self.a = a
        self.b = b
        self.c = c
        self.len_A = len(A)
        self.len_B = len(B)
        self.matrix = np.zeros((self.len_A + 1, self.len_B + 1))
        self.traceback_matrix = np.empty((self.len_A + 1, self.len_B + 1), dtype=object)
        self.traceback_matrix.fill([])

    def initialize_matrix(self):
        """初始化矩阵的第一行和第一列"""
        for i in range(self.len_A + 1):
            self.matrix[i][0] = i * self.a
            self.traceback_matrix[i][0] = []
        for j in range(self.len_B + 1):
            self.matrix[0][j] = j * self.a
            self.traceback_matrix[0][j] = []

    def calculate_cell(self, i, j):
        """计算单个矩阵元素的值"""
        if self.A[i - 1] == self.B[j - 1]:
            diag = self.matrix[i - 1][j - 1] + self.c
        else:
            diag = self.matrix[i - 1][j - 1] + self.b
        up = self.matrix[i - 1][j] + self.a
        left = self.matrix[i][j - 1] + self.a
        scores = [diag, up, left]
        max_score = max(scores)
        # 记录所有得分等于 max_score 的方向
        directions = []
        if diag == max_score:
            directions.append(1)
        if up == max_score:
            directions.append(2)
        if left == max_score:
            directions.append(3)
        # 返回结果
        return (i, j, max_score, directions)

    def calculate_matrix_parallel(self):
        """使用反对角线并行计算矩阵"""
        max_index = self.len_A + self.len_B
        for k in range(2, max_index + 1):
            indices = []
            for i in range(1, self.len_A + 1):
                j = k - i
                if 1 <= j <= self.len_B:
                    indices.append((i, j))
            # 并行计算当前反对角线上的所有元素
            results = Parallel(n_jobs=-1)(
                delayed(self.calculate_cell)(i, j) for i, j in indices
            )
            # 在主进程中更新矩阵
            for i, j, max_score, directions in results:
                self.matrix[i][j] = max_score
                self.traceback_matrix[i][j] = directions  # 存储列表

    def traceback(self, max_paths=1):
        """沿着回溯指针寻找多条最优比对路径"""
        paths = []
        stack = [ (self.len_A, self.len_B, [], []) ]  # (i, j, aligned_A, aligned_B)

        while stack and len(paths) < max_paths:
            i, j, aligned_A, aligned_B = stack.pop()
            if i == 0 and j == 0:
                # 已到达起点，保存路径
                paths.append( (''.join(reversed(aligned_A)), ''.join(reversed(aligned_B))) )
                continue
            directions = self.traceback_matrix[i][j]
            for direction in directions:
                if direction == 1 and i > 0 and j > 0:
                    # Match/Mismatch
                    aligned_A_new = aligned_A + [self.A[i - 1]]
                    aligned_B_new = aligned_B + [self.B[j - 1]]
                    stack.append( (i -1, j -1, aligned_A_new, aligned_B_new) )
                elif direction == 2 and i > 0:
                    # Deletion (gap in B)
                    aligned_A_new = aligned_A + [self.A[i -1]]
                    aligned_B_new = aligned_B + ['-']
                    stack.append( (i -1, j, aligned_A_new, aligned_B_new) )
                elif direction == 3 and j > 0:
                    # Insertion (gap in A)
                    aligned_A_new = aligned_A + ['-']
                    aligned_B_new = aligned_B + [self.B[j -1]]
                    stack.append( (i, j -1, aligned_A_new, aligned_B_new) )
        # 保存结果
        self.aligned_sequences = paths

    def print_alignments(self):
        """打印所有比对结果"""
        for idx, (aligned_A, aligned_B) in enumerate(self.aligned_sequences):
            print(f"Alignment {idx + 1}:")
            print("Aligned Sequence A:", aligned_A)
            print("Aligned Sequence B:", aligned_B)
            print()

    def get_matrix(self):
        """返回计算完成的矩阵"""
        return self.matrix

    def print_matrix(self):
        """打印矩阵"""
        print(pd.DataFrame(self.matrix))

if __name__ == "__main__":
    # 示例输入（请替换为你的序列）
    A = "atttattttatttaccttaaaactagtattttaataaataaaattattttttcttatttatttaattataaaaacctcattatttttttaaaactctatttatttttaaataaaatattttttaatttattttacgaaaaatgagatattaatatataatgactgaaatgtaaagattttacaagtataagtattaaatgagtaattgaaaaaaagtttaatcttataaataatgctttttttccttctccatatatcatgaacacaataatacacaatccctttgaaattaaatacactttaggacaaaacatgaccccttaaaaagagatagtacctcccattccagttacatttgtgacaaagttagaaagaaggagaaagaaagaagatggcaggaaagaaaatcccacatgtgttgctgaattctggacacaaaatgccagttataggcatgggaacttcagtggagaatcgtccatcaaatga"
    B = "actccctgagttttccttgctgccaagatgggttcatttccacctgaaaaaacaataattcaacatcataacatgagtaccaagaagtccaagaccatcataataactatgtgccaaaccaaatatacatctctggagtaatcaaattatggtacctgattgactgcaggaggaaaggtggctatttccaagagtttggtgagttttttgacaccataattgcatatgccaatggactttgctaatcccaacttgtaacactcttccatagctttccatgtcccttctatatcaaagggcagaaaatcttccttggtaaaaacagttggattttaaagatcatgtctcagcctcactggccaatgaatcagataaagatccacatactg"
    # 初始化对象
    aligner = SequenceAlignment(A, B, a=-1, b=-1, c=2)

    # 初始化矩阵
    aligner.initialize_matrix()

    # 计算打分矩阵
    aligner.calculate_matrix_parallel()

    # 回溯得到比对结果，指定最多输出 5 条最优路径
    aligner.traceback(max_paths=5)

    # 输出比对结果
    aligner.print_alignments()
