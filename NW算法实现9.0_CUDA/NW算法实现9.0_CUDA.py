# 用法 
# python nw9.py -i input.fasta -o alignment_output.txt -g 10.0 -e 0.5 -p 5
# 没有实现 EDNAFULL 的算法，只是先写上去，都是计划。
# GPU的版本非常快，终于是有了一点可用的样子了。


import numpy as np
import pandas as pd
from numba import cuda, int32
import math
from Bio import SeqIO
import argparse


def read_fasta_biopython(file_path):
    """
    使用Biopython读取FASTA文件并返回一个列表，包含所有序列的ID和序列本身。

    参数:
    - file_path: FASTA文件的路径

    返回:
    - sequences: 一个列表，每个元素是一个元组（序列ID, 序列）
    """
    try:
        sequences = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append((record.id, str(record.seq).upper()))
        return sequences
    except Exception as e:
        print(f"读取FASTA文件时出错: {e}")
        return []


def validate_sequences(sequences):
    """
    验证序列数量是否满足要求。

    参数:
    - sequences: 包含所有序列的列表

    返回:
    - (bool, list): 第一个元素表示是否验证通过，第二个元素是错误信息（如果有）
    """
    if len(sequences) < 2:
        return (False, "FASTA文件中序列不足两条，请提供至少两条序列。")
    return (True, "")


def select_sequences(sequences):
    """
    如果有多于两条序列，显示列表并让用户选择要比对的两条序列。

    参数:
    - sequences: 包含所有序列的列表

    返回:
    - selected_sequences: 一个列表，包含两条用户选择的序列（每条为元组（ID, 序列））
    """
    print("FASTA文件中包含多于两条序列。请选择要比对的两条序列：")
    for idx, (seq_id, seq) in enumerate(sequences, start=1):
        print(f"{idx}: {seq_id} (长度: {len(seq)})")

    while True:
        try:
            selection = input("请输入两条序列的编号，用空格分隔（例如：1 3）： ")
            indices = selection.strip().split()
            if len(indices) != 2:
                print("请同时输入两条序列的编号。")
                continue
            idx1, idx2 = int(indices[0]), int(indices[1])
            if idx1 < 1 or idx1 > len(sequences) or idx2 < 1 or idx2 > len(sequences):
                print("输入的编号超出范围，请重新选择。")
                continue
            if idx1 == idx2:
                print("请不要选择相同的序列。")
                continue
            selected_sequences = [sequences[idx1 - 1], sequences[idx2 - 1]]
            return selected_sequences
        except ValueError:
            print("无效的输入，请输入数字编号。")


@cuda.jit
def compute_cells_gpu(matrix, traceback_matrix, A, B, len_A, len_B, a, b, c, k):
    # 共享内存大小根据每个块处理的元素数量调整
    shared_diag = cuda.shared.array(shape=256, dtype=int32)  # 假设每个块最多处理256个元素
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x

    i = tx + 1 + bx * bw
    j = k - i

    if i > len_A or j < 1 or j > len_B:
        return

    # 计算 diag, up, left
    match_score = c if A[i - 1] == B[j - 1] else b
    diag = matrix[i - 1, j - 1] + match_score
    up = matrix[i - 1, j] + a
    left = matrix[i, j - 1] + a

    max_score = diag
    directions = 1  # diag

    if up > max_score:
        max_score = up
        directions = 2  # up
    elif up == max_score:
        directions |= 2  # up

    if left > max_score:
        max_score = left
        directions = 4  # left
    elif left == max_score:
        directions |= 4  # left

    matrix[i, j] = max_score
    traceback_matrix[i, j] = directions


class SequenceAlignmentGPU:
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

        # 初始化矩阵
        self.matrix = np.zeros((self.len_A + 1, self.len_B + 1), dtype=np.int32)
        self.traceback_matrix = np.zeros((self.len_A + 1, self.len_B + 1), dtype=np.int32)

        # 初始化矩阵的第一行和第一列
        self.matrix[:, 0] = np.arange(self.len_A + 1) * self.a
        self.matrix[0, :] = np.arange(self.len_B + 1) * self.a

    def calculate_matrix_gpu(self):
        """
        使用 CUDA 并行计算得分矩阵
        """
        # 将数据拷贝到 GPU
        d_matrix = cuda.to_device(self.matrix)
        d_traceback = cuda.to_device(self.traceback_matrix)

        # 将序列转换为 ASCII 整数数组
        A_int = np.array([ord(c) for c in self.A], dtype=np.int8)
        B_int = np.array([ord(c) for c in self.B], dtype=np.int8)

        d_A = cuda.to_device(A_int)
        d_B = cuda.to_device(B_int)

        threads_per_block = 256
        for k in range(2, self.len_A + self.len_B + 1):
            # 计算当前反对角线上的元素数量
            max_elements = min(k - 1, self.len_A, self.len_B)
            blocks_per_grid = math.ceil(max_elements / threads_per_block)

            compute_cells_gpu[blocks_per_grid, threads_per_block](
                d_matrix, d_traceback, d_A, d_B,
                self.len_A, self.len_B, self.a, self.b, self.c, k
            )
            cuda.synchronize()

        # 从 GPU 拷贝回主机
        self.matrix = d_matrix.copy_to_host()
        self.traceback_matrix = d_traceback.copy_to_host()

    def traceback(self, max_paths=5):
        """沿着回溯指针寻找多条最优比对路径"""
        paths = []
        stack = [(self.len_A, self.len_B, [], [])]  # (i, j, aligned_A, aligned_B)

        while stack and len(paths) < max_paths:
            i, j, aligned_A, aligned_B = stack.pop()
            if i == 0 and j == 0:
                # 已到达起点，保存路径
                paths.append((''.join(reversed(aligned_A)), ''.join(reversed(aligned_B))))
                continue
            directions = self.traceback_matrix[i][j]
            # 解析方向位掩码
            if directions & 1:  # diag
                if i > 0 and j > 0:
                    aligned_A_new = aligned_A + [self.A[i - 1]]
                    aligned_B_new = aligned_B + [self.B[j - 1]]
                    stack.append((i - 1, j - 1, aligned_A_new, aligned_B_new))
            if directions & 2:  # up
                if i > 0:
                    aligned_A_new = aligned_A + [self.A[i - 1]]
                    aligned_B_new = aligned_B + ['-']
                    stack.append((i - 1, j, aligned_A_new, aligned_B_new))
            if directions & 4:  # left
                if j > 0:
                    aligned_A_new = aligned_A + ['-']
                    aligned_B_new = aligned_B + [self.B[j - 1]]
                    stack.append((i, j - 1, aligned_A_new, aligned_B_new))
        # 保存结果
        self.aligned_sequences = paths

    def print_alignments(self):
        """打印所有比对结果"""
        if not hasattr(self, 'aligned_sequences') or not self.aligned_sequences:
            print("No alignments found.")
            return
        for idx, (aligned_A, aligned_B) in enumerate(self.aligned_sequences):
            print(f"Alignment {idx + 1}:")
            print("Aligned Sequence A:", aligned_A)
            print("Aligned Sequence B:", aligned_B)
            print()

    def get_matrix(self):
        """返回计算完成的矩阵"""
        return self.matrix

    def write_alignment_to_file(self, file_path, alignment_num, seq1_id, seq2_id, matrix_type, gap_penalty,
                                extend_penalty, metrics, aligned_A, aligned_B):
        """
        将比对结果写入文件，包含头信息和比对内容。

        参数:
        - file_path: 输出文件路径
        - alignment_num: 比对编号
        - seq1_id: 序列1的ID
        - seq2_id: 序列2的ID
        - matrix_type: 矩阵类型
        - gap_penalty: gap罚分
        - extend_penalty: gap延伸罚分
        - metrics: 比对指标字典
        - aligned_A: 对齐后的序列A
        - aligned_B: 对齐后的序列B
        """
        # 计算比对匹配指示符
        aligned_matches = ''
        for a, b in zip(aligned_A, aligned_B):
            if a == b:
                aligned_matches += '|'
            elif a == '-' or b == '-':
                aligned_matches += '-'
            else:
                aligned_matches += '.'

        with open(file_path, 'a') as file:
            # 写入头部信息
            file.write("#=======================================\n")
            file.write("#\n")
            file.write(f"# Aligned_sequences: 2\n")
            file.write(f"# 1: {seq1_id}\n")
            file.write(f"# 2: {seq2_id}\n")
            file.write(f"# Matrix: {matrix_type}\n")
            file.write(f"# Gap_penalty: {gap_penalty}\n")
            file.write(f"# Extend_penalty: {extend_penalty}\n")
            file.write("#\n")
            file.write(f"# Length: {metrics['length']}\n")
            file.write(
                f"# Identity:     {metrics['identity']}/{metrics['length']} ({metrics['identity'] / metrics['length'] * 100:.1f}%)\n")
            file.write(
                f"# Similarity:   {metrics['similarity']}/{metrics['length']} ({metrics['similarity'] / metrics['length'] * 100:.1f}%)\n")
            file.write(
                f"# Gaps:         {metrics['gaps']}/{metrics['length']} ({metrics['gaps'] / metrics['length'] * 100:.1f}%)\n")
            file.write(f"# Score: {metrics['score']}\n")
            file.write("# \n")
            file.write("#\n")
            file.write("#=======================================\n\n")

            # 定义固定宽度
            id_width = 3  # 比对编号宽度
            pos_width = 6  # 位置数字宽度（根据序列长度动态调整）
            seq_width = 50  # 序列宽度
            end_pos_width = 6  # 结束位置宽度

            # 计算匹配符号行的缩进
            padding = ' ' * (id_width + 1 + pos_width + 1 - 3)   # 3 + 1 + 6 + 1 = 11

            # 设置段长度
            segment_length = 50
            for start in range(0, metrics['length'], segment_length):
                end = min(start + segment_length, metrics['length'])
                seg_A = aligned_A[start:end]
                seg_B = aligned_B[start:end]
                seg_match = aligned_matches[start:end]

                # 序列1行
                file.write(f"1  {start + 1:>5} {seg_A:<50} {start + len(seg_A):>5}\n")
                # 匹配符号行
                file.write(f"{padding} {seg_match}\n")
                # 序列2行
                file.write(f"2  {start + 1:>5} {seg_B:<50} {start + len(seg_B):>5}\n\n")

    def read_fasta_biopython(file_path):
        """
        使用Biopython读取FASTA文件并返回一个列表，包含所有序列的ID和序列本身。

        参数:
        - file_path: FASTA文件的路径

        返回:
        - sequences: 一个列表，每个元素是一个元组（序列ID, 序列）
        """
        sequences = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append((record.id, str(record.seq).upper()))
        return sequences

    def validate_sequences(sequences):
        """
        验证序列数量是否满足要求。

        参数:
        - sequences: 包含所有序列的列表

        返回:
        - (bool, list): 第一个元素表示是否验证通过，第二个元素是错误信息（如果有）
        """
        if len(sequences) < 2:
            return (False, "FASTA文件中序列不足两条，请提供至少两条序列。")
        return (True, "")

    def select_sequences(sequences):
        """
        如果有多于两条序列，显示列表并让用户选择要比对的两条序列。

        参数:
        - sequences: 包含所有序列的列表

        返回:
        - selected_sequences: 一个列表，包含两条用户选择的序列（每条为元组（ID, 序列））
        """
        print("FASTA文件中包含多于两条序列。请选择要比对的两条序列：")
        for idx, (seq_id, seq) in enumerate(sequences, start=1):
            print(f"{idx}: {seq_id} (长度: {len(seq)})")

        while True:
            try:
                selection = input("请输入两条序列的编号，用空格分隔（例如：1 3）： ")
                indices = selection.strip().split()
                if len(indices) != 2:
                    print("请同时输入两条序列的编号。")
                    continue
                idx1, idx2 = int(indices[0]), int(indices[1])
                if idx1 < 1 or idx1 > len(sequences) or idx2 < 1 or idx2 > len(sequences):
                    print("输入的编号超出范围，请重新选择。")
                    continue
                if idx1 == idx2:
                    print("请不要选择相同的序列。")
                    continue
                selected_sequences = [sequences[idx1 - 1], sequences[idx2 - 1]]
                return selected_sequences
            except ValueError:
                print("无效的输入，请输入数字编号。")


def calculate_alignment_metrics(aligned_A, aligned_B):
    """
    计算比对的各项指标。

    参数:
    - aligned_A: 对齐后的序列A
    - aligned_B: 对齐后的序列B

    返回:
    - metrics: 一个字典，包含长度、身份率、相似率、缺口数量、得分
    """
    length = len(aligned_A)
    identity = 0
    similarity = 0
    gaps = 0
    score = 0
    match_score = 1  # 匹配得分
    mismatch_score = 0  # 错配得分
    gap_penalty = -2  # gap罚分

    for a, b in zip(aligned_A, aligned_B):
        if a == '-' or b == '-':
            gaps += 1
            score += gap_penalty
        elif a == b:
            identity += 1
            similarity += 1
            score += match_score
        else:
            similarity += 1
            score += mismatch_score

    metrics = {
        'length': length,
        'identity': identity,
        'similarity': similarity,
        'gaps': gaps,
        'score': score
    }
    return metrics

def run_alignment(input_fasta, output_file, matrix_type="按计划应该是EDNAFULL", gap_penalty=10.0, extend_penalty=0.5, max_paths=5):
    # 读取FASTA文件
    sequences = read_fasta_biopython(input_fasta)

    # 验证序列数量
    valid, message = validate_sequences(sequences)
    if not valid:
        print(message)
        return

    # 选择序列（如果有多于两条序列）
    if len(sequences) > 2:
        selected_sequences = select_sequences(sequences)
    else:
        selected_sequences = sequences[:2]

    (seq1_id, seq1), (seq2_id, seq2) = selected_sequences

    # 初始化比对对象
    aligner = SequenceAlignmentGPU(seq1, seq2, a=-gap_penalty, b=-extend_penalty, c=10)  # 注意符号

    # 计算得分矩阵
    aligner.calculate_matrix_gpu()

    # 回溯得到比对结果
    aligner.traceback(max_paths=max_paths)

    # 获取比对结果
    alignments = aligner.aligned_sequences

    # 计算比对指标并写入文件
    for idx, (aligned_A, aligned_B) in enumerate(alignments, start=1):
        metrics = calculate_alignment_metrics(aligned_A, aligned_B)
        aligner.write_alignment_to_file(
            file_path=output_file,
            alignment_num=idx,
            seq1_id=seq1_id,
            seq2_id=seq2_id,
            matrix_type=matrix_type,
            gap_penalty=gap_penalty,
            extend_penalty=extend_penalty,
            metrics=metrics,
            aligned_A=aligned_A,
            aligned_B=aligned_B
        )

    print(f"比对完成，结果已写入 {output_file}")

def main():
    parser = argparse.ArgumentParser(description="GPU加速的Needleman-Wunsch序列比对工具")
    parser.add_argument("-i", "--input", required=True, help="输入的FASTA文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出的比对结果文件路径")
    parser.add_argument("-m", "--matrix", default="按计划是EDNAFULL", help="矩阵类型（默认：按计划是EDNAFULL）")
    parser.add_argument("-g", "--gap", type=float, default=10.0, help="gap罚分（默认：10.0）")
    parser.add_argument("-e", "--extend", type=float, default=0.5, help="gap延伸罚分（默认：0.5）")
    parser.add_argument("-p", "--paths", type=int, default=5, help="最多输出的比对路径数量（默认：5）")

    args = parser.parse_args()

    run_alignment(
        input_fasta=args.input,
        output_file=args.output,
        matrix_type=args.matrix,
        gap_penalty=args.gap,
        extend_penalty=args.extend,
        max_paths=args.paths
    )


if __name__ == "__main__":
    main()
