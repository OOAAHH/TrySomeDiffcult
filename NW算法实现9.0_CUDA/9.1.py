import numpy as np
import cupy as cp
import math
from Bio import SeqIO
import argparse
import pynvml
import time  # 用于计时
import psutil  # 用于获取CPU和内存使用情况


def get_gpu_info():
    """
    获取可用的GPU信息，包括每个GPU的总内存。
    返回一个列表，每个元素是一个字典，包含GPU的ID和总内存。
    """
    gpu_list = []
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mem = meminfo.total  # 总内存，单位为字节
            gpu_list.append({'id': i, 'total_mem': total_mem})
    except pynvml.NVMLError as e:
        print(f"无法获取GPU信息: {e}")
    finally:
        pynvml.nvmlShutdown()
    return gpu_list


def estimate_max_sequence_length(gpu_info):
    """
    根据可用的GPU内存，估计可处理的最大序列长度（假设两个序列长度相同）。
    """
    total_mem = sum([gpu['total_mem'] for gpu in gpu_info])
    # 估计需要的内存：memory_required = 8 * (L+1)*(L+1) 字节
    max_length = int((total_mem / 8) ** 0.5) - 1  # 减1以确保安全
    return max_length


def read_fasta_biopython(file_path):
    """
    使用Biopython读取FASTA文件并返回一个列表，包含所有序列的ID和序列本身。
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
    """
    if len(sequences) < 2:
        return (False, "FASTA文件中序列不足两条，请提供至少两条序列。")
    return (True, "")


def select_sequences(sequences, gpu_info):
    """
    选择要比对的两条序列，并根据GPU资源提供建议。
    """
    max_seq_length = estimate_max_sequence_length(gpu_info)
    print(f"根据当前可用的GPU资源，最大可处理的序列长度约为：{max_seq_length}")

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

            # 检查所选序列的长度是否超过最大可处理长度
            seq1_length = len(selected_sequences[0][1])
            seq2_length = len(selected_sequences[1][1])
            if seq1_length > max_seq_length or seq2_length > max_seq_length:
                print(f"所选序列长度超过了可用GPU资源的处理能力（最大长度：{max_seq_length}）。")
                print("请重新选择较短的序列或增加GPU资源。")
                continue

            return selected_sequences
        except ValueError:
            print("无效的输入，请输入数字编号。")


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
        使用 CuPy 并行计算得分矩阵
        """
        start_time = time.time()  # 开始计时

        # 将得分矩阵和回溯矩阵拷贝到GPU
        d_matrix = cp.asarray(self.matrix)
        d_traceback = cp.asarray(self.traceback_matrix)

        # 将序列转换为ASCII整数数组
        A_int = np.array([ord(c) for c in self.A], dtype=np.int8)
        B_int = np.array([ord(c) for c in self.B], dtype=np.int8)

        # 拷贝序列到GPU
        d_A = cp.asarray(A_int)
        d_B = cp.asarray(B_int)

        len_A = self.len_A
        len_B = self.len_B
        a = self.a
        b = self.b
        c = self.c

        # 记录内存使用情况
        mem_usage_before = cp.get_default_memory_pool().used_bytes()

        # 编译CUDA核函数（只编译一次）
        kernel_code = '''
        extern "C" __global__
        void compute_cells(int* matrix, int* traceback_matrix, char* A, char* B, int len_B_plus1, int a, int b, int c, int* i_list, int* j_list, int n)
        {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx < n)
            {
                int i = i_list[idx];
                int j = j_list[idx];
                int idx_ij = i * len_B_plus1 + j;
                int idx_i1j1 = (i - 1) * len_B_plus1 + (j - 1);
                int idx_i1j = (i - 1) * len_B_plus1 + j;
                int idx_ij1 = i * len_B_plus1 + (j - 1);

                int match_score = (A[i - 1] == B[j - 1]) ? c : b;
                int diag = matrix[idx_i1j1] + match_score;
                int up = matrix[idx_i1j] + a;
                int left = matrix[idx_ij1] + a;

                int max_score = diag;
                int directions = 1;  // diag

                if (up > max_score)
                {
                    max_score = up;
                    directions = 2;  // up
                }
                else if (up == max_score)
                {
                    directions |= 2;  // up
                }

                if (left > max_score)
                {
                    max_score = left;
                    directions = 4;  // left
                }
                else if (left == max_score)
                {
                    directions |= 4;  // left
                }

                matrix[idx_ij] = max_score;
                traceback_matrix[idx_ij] = directions;
            }
        }
        '''

        mod = cp.RawModule(code=kernel_code)
        compute_cells = mod.get_function('compute_cells')

        # 计算反对角线
        for k in range(2, len_A + len_B + 1):
            # 计算当前反对角线上的元素
            i_values = np.arange(max(1, k - len_B), min(len_A, k - 1) + 1)
            j_values = k - i_values

            if len(i_values) == 0:
                continue

            i_gpu = cp.asarray(i_values, dtype=cp.int32)
            j_gpu = cp.asarray(j_values, dtype=cp.int32)

            n = len(i_values)

            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

            compute_cells(
                (blocks_per_grid,), (threads_per_block,),
                (d_matrix, d_traceback, d_A, d_B, len_B + 1, a, b, c, i_gpu, j_gpu, n)
            )

        cp.cuda.Stream.null.synchronize()  # 等待GPU计算完成

        # 记录内存使用情况
        mem_usage_after = cp.get_default_memory_pool().used_bytes()
        gpu_memory_used = mem_usage_after - mem_usage_before

        # 将计算结果拷贝回主机
        self.matrix = cp.asnumpy(d_matrix)
        self.traceback_matrix = cp.asnumpy(d_traceback)

        end_time = time.time()  # 结束计时
        self.gpu_compute_time = end_time - start_time  # GPU计算耗时
        self.gpu_memory_used = gpu_memory_used  # GPU内存使用

    def traceback(self, max_paths=5):
        """沿着回溯指针寻找多条最优比对路径"""
        start_time = time.time()  # 开始计时

        paths = []
        stack = [(self.len_A, self.len_B, [], [])]

        while stack and len(paths) < max_paths:
            i, j, aligned_A, aligned_B = stack.pop()
            if i == 0 and j == 0:
                paths.append((''.join(reversed(aligned_A)), ''.join(reversed(aligned_B))))
                continue
            directions = self.traceback_matrix[i][j]
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

        end_time = time.time()  # 结束计时
        self.traceback_time = end_time - start_time  # 回溯耗时

        self.aligned_sequences = paths

    def write_alignment_to_file(self, file_path, alignment_num, seq1_id, seq2_id, matrix_type, gap_penalty,
                                extend_penalty, metrics, aligned_A, aligned_B):
        """
        将比对结果写入文件，包含头信息和比对内容。
        """
        # 计算比对匹配指示符
        aligned_matches = ''
        for a, b in zip(aligned_A, aligned_B):
            if a == b:
                aligned_matches += '|'
            elif a == '-' or b == '-':
                aligned_matches += ' '
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
            file.write("#=======================================\n\n")

            # 定义固定宽度
            id_width = 3
            pos_width = 6
            seq_width = 50
            end_pos_width = 6

            # 计算匹配符号行的缩进
            padding = ' ' * (id_width + 1 + pos_width + 1 - 3)

            segment_length = 50
            for start in range(0, metrics['length'], segment_length):
                end = min(start + segment_length, metrics['length'])
                seg_A = aligned_A[start:end]
                seg_B = aligned_B[start:end]
                seg_match = aligned_matches[start:end]

                # 序列1行
                file.write(f"1 {start + 1:>5} {seg_A:<50} {start + len(seg_A):>5}\n")
                # 匹配符号行
                file.write(f"{padding} {seg_match}\n")
                # 序列2行
                file.write(f"2 {start + 1:>5} {seg_B:<50} {start + len(seg_B):>5}\n\n")


def calculate_alignment_metrics(aligned_A, aligned_B):
    """
    计算比对的各项指标。
    """
    length = len(aligned_A)
    identity = 0
    similarity = 0
    gaps = 0
    score = 0
    match_score = 1
    mismatch_score = 0
    gap_penalty = -2

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


def run_alignment(input_fasta, output_file, matrix_type="EDNAFULL", gap_penalty=10.0, extend_penalty=0.5, max_paths=5):
    total_start_time = time.time()  # 总计时开始

    # 获取可用的GPU信息
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("未检测到可用的GPU设备。")
        return
    else:
        total_gpu_mem = sum([gpu['total_mem'] for gpu in gpu_info])
        print(f"检测到{len(gpu_info)}个GPU设备，总显存：{total_gpu_mem / (1024 ** 3):.2f} GB")

    # 读取FASTA文件
    read_start_time = time.time()
    sequences = read_fasta_biopython(input_fasta)
    read_end_time = time.time()
    read_time = read_end_time - read_start_time  # 读取文件耗时

    # 验证序列数量
    valid, message = validate_sequences(sequences)
    if not valid:
        print(message)
        return

    # 选择序列（如果有多于两条序列）
    select_start_time = time.time()
    if len(sequences) > 2:
        selected_sequences = select_sequences(sequences, gpu_info)
    else:
        selected_sequences = sequences[:2]
        # 检查序列长度
        max_seq_length = estimate_max_sequence_length(gpu_info)
        seq1_length = len(selected_sequences[0][1])
        seq2_length = len(selected_sequences[1][1])
        if seq1_length > max_seq_length or seq2_length > max_seq_length:
            print(f"所选序列长度超过了可用GPU资源的处理能力（最大长度：{max_seq_length}）。")
            print("请提供较短的序列或增加GPU资源。")
            return
    select_end_time = time.time()
    select_time = select_end_time - select_start_time  # 选择序列耗时

    (seq1_id, seq1), (seq2_id, seq2) = selected_sequences

    # 初始化比对对象
    aligner = SequenceAlignmentGPU(seq1, seq2, a=-gap_penalty, b=-extend_penalty, c=10)

    # 计算得分矩阵
    aligner.calculate_matrix_gpu()

    # 回溯得到比对结果
    aligner.traceback(max_paths=max_paths)

    # 获取比对结果
    alignments = aligner.aligned_sequences

    # 计算比对指标并写入文件
    metrics_list = []
    write_start_time = time.time()
    for idx, (aligned_A, aligned_B) in enumerate(alignments, start=1):
        metrics = calculate_alignment_metrics(aligned_A, aligned_B)
        metrics_list.append(metrics)
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
    write_end_time = time.time()
    write_time = write_end_time - write_start_time  # 写入文件耗时

    total_end_time = time.time()  # 总计时结束
    total_time = total_end_time - total_start_time  # 总耗时

    # 获取CPU和内存使用情况
    process = psutil.Process()
    cpu_percent = process.cpu_percent(interval=1)  # 获取CPU使用率
    mem_info = process.memory_info()
    memory_used = mem_info.rss  # 以字节为单位的内存使用

    # 打印统计信息
    print("\n========= 计算统计信息 =========")
    print(f"总耗时：{total_time:.2f} 秒")
    print(f"  - 读取FASTA文件耗时：{read_time:.2f} 秒")
    print(f"  - 选择序列耗时：{select_time:.2f} 秒")
    print(f"  - GPU计算得分矩阵耗时：{aligner.gpu_compute_time:.2f} 秒")
    print(f"    - GPU内存使用：{aligner.gpu_memory_used / (1024 ** 2):.2f} MB")
    print(f"  - 回溯耗时：{aligner.traceback_time:.2f} 秒")
    print(f"  - 写入比对结果耗时：{write_time:.2f} 秒")
    print(f"CPU使用率：{cpu_percent}%")
    print(f"内存使用：{memory_used / (1024 ** 2):.2f} MB")
    print("================================")

    print(f"\n比对完成，结果已写入 {output_file}")


def main():
    parser = argparse.ArgumentParser(description="GPU加速的Needleman-Wunsch序列比对工具")
    parser.add_argument("-i", "--input", required=True, help="输入的FASTA文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出的比对结果文件路径")
    parser.add_argument("-m", "--matrix", default="EDNAFULL", help="矩阵类型（默认：EDNAFULL）")
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
