use bio::io::fasta; // 引入 bio 模块
use clap::{Arg, Command}; // 引入 clap 库
use ndarray::Array2;
use rayon::prelude::*;
use std::collections::VecDeque;

/// 定义回溯方向
#[derive(Debug, Clone, Copy)]
enum Direction {
    Diag,
    Up,
    Left,
}

/// 序列比对结构体
struct SequenceAlignment {
    a: Vec<char>,
    b: Vec<char>,
    match_score: i32,      // 匹配的得分
    mismatch_penalty: i32, // 错配的罚分
    gap_penalty: i32,      // 缺失的罚分
    len_a: usize,
    len_b: usize,
    matrix: Array2<i32>,
    traceback_matrix: Vec<Vec<Vec<Direction>>>,
    aligned_sequences: Vec<(String, String)>,
}

impl SequenceAlignment {
    /// 创建新的序列比对对象
    fn new(a: String, b: String, match_score: i32, mismatch_penalty: i32, gap_penalty: i32) -> Self {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let len_a = a_chars.len();
        let len_b = b_chars.len();

        // 确保输入序列非空
        if len_a == 0 || len_b == 0 {
            panic!("输入序列 A 和 B 必须非空");
        }

        let matrix = Array2::<i32>::zeros((len_a + 1, len_b + 1));
        let traceback_matrix: Vec<Vec<Vec<Direction>>> =
            vec![vec![Vec::new(); len_b + 1]; len_a + 1];
        SequenceAlignment {
            a: a_chars,
            b: b_chars,
            match_score,
            mismatch_penalty,
            gap_penalty,
            len_a,
            len_b,
            matrix,
            traceback_matrix,
            aligned_sequences: Vec::new(),
        }
    }

    /// 初始化打分矩阵和回溯矩阵的第一行和第一列
    fn initialize_matrix(&mut self) {
        // 初始化第一列
        for i in 0..=self.len_a {
            self.matrix[[i, 0]] = (i as i32) * self.gap_penalty;
            if i > 0 {
                self.traceback_matrix[i][0].push(Direction::Up);
            }
        }

        // 初始化第一行
        for j in 0..=self.len_b {
            self.matrix[[0, j]] = (j as i32) * self.gap_penalty;
            if j > 0 {
                self.traceback_matrix[0][j].push(Direction::Left);
            }
        }
    }

    /// 计算单个矩阵单元的分数和方向
    fn calculate_cell(&self, i: usize, j: usize) -> (usize, usize, i32, Vec<Direction>) {
        // 确保 i >0 and j >0
        if i == 0 || j == 0 {
            panic!(
                "calculate_cell called with i=0 or j=0 (i={}, j={})",
                i, j
            );
        }

        let diag_score = if self.a[i - 1] == self.b[j - 1] {
            self.matrix[[i - 1, j - 1]] + self.match_score
        } else {
            self.matrix[[i - 1, j - 1]] + self.mismatch_penalty
        };
        let up_score = self.matrix[[i - 1, j]] + self.gap_penalty;
        let left_score = self.matrix[[i, j - 1]] + self.gap_penalty;

        let scores = [diag_score, up_score, left_score];
        let max_score = *scores.iter().max().unwrap();

        let mut directions = Vec::new();
        if diag_score == max_score {
            directions.push(Direction::Diag);
        }
        if up_score == max_score {
            directions.push(Direction::Up);
        }
        if left_score == max_score {
            directions.push(Direction::Left);
        }

        (i, j, max_score, directions)
    }

    /// 并行计算打分矩阵
    fn calculate_matrix_parallel(&mut self) {
        let max_index = self.len_a + self.len_b;
        for k in 2..=max_index {
            // 收集当前反对角线上的所有有效 (i, j) 对
            let indices: Vec<(usize, usize)> = (1..=self.len_a)
                .filter_map(|i| {
                    // 使用 checked_sub 以防止溢出
                    if let Some(j) = k.checked_sub(i) {
                        if j >= 1 && j <= self.len_b {
                            Some((i, j))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            if indices.is_empty() {
                continue;
            }

            // 并行计算当前反对角线上的所有单元
            let results: Vec<(usize, usize, i32, Vec<Direction>)> =
                indices.par_iter().map(|&(i, j)| self.calculate_cell(i, j)).collect();

            // 在主线程中更新矩阵和回溯矩阵
            for (i, j, max_score, directions) in results {
                self.matrix[[i, j]] = max_score;
                self.traceback_matrix[i][j] = directions;
            }
        }
    }

    /// 回溯以获取比对结果
    fn traceback(&mut self) {
        self.aligned_sequences.clear();
        let mut stack: VecDeque<(usize, usize, Vec<char>, Vec<char>)> =
            VecDeque::new();
        stack.push_back((self.len_a, self.len_b, Vec::new(), Vec::new()));

        while let Some((i, j, aligned_a, aligned_b)) = stack.pop_back() {
            // 不再限制路径数量
            if i == 0 && j == 0 {
                // 已到达起点，保存路径
                let aligned_a_str: String = aligned_a.into_iter().rev().collect();
                let aligned_b_str: String = aligned_b.into_iter().rev().collect();
                self.aligned_sequences.push((aligned_a_str, aligned_b_str));
                continue;
            }

            for &direction in &self.traceback_matrix[i][j] {
                match direction {
                    Direction::Diag => {
                        if i > 0 && j > 0 {
                            let mut new_aligned_a = aligned_a.clone();
                            let mut new_aligned_b = aligned_b.clone();
                            new_aligned_a.push(self.a[i - 1]);
                            new_aligned_b.push(self.b[j - 1]);
                            stack.push_back((i - 1, j - 1, new_aligned_a, new_aligned_b));
                        }
                    }
                    Direction::Up => {
                        if i > 0 {
                            let mut new_aligned_a = aligned_a.clone();
                            let mut new_aligned_b = aligned_b.clone();
                            new_aligned_a.push(self.a[i - 1]);
                            new_aligned_b.push('-');
                            stack.push_back((i - 1, j, new_aligned_a, new_aligned_b));
                        }
                    }
                    Direction::Left => {
                        if j > 0 {
                            let mut new_aligned_a = aligned_a.clone();
                            let mut new_aligned_b = aligned_b.clone();
                            new_aligned_a.push('-');
                            new_aligned_b.push(self.b[j - 1]);
                            stack.push_back((i, j - 1, new_aligned_a, new_aligned_b));
                        }
                    }
                }
            }
        }
    }

    /// 打印所有比对结果
    fn print_alignments(&self) {
        for (idx, (aligned_a, aligned_b)) in self.aligned_sequences.iter().enumerate() {
            println!("Alignment {}:", idx + 1);
            println!("Aligned Sequence A: {}", aligned_a);
            println!("Aligned Sequence B: {}", aligned_b);
            println!();
        }
    }
}

fn main() {
    // 解析命令行参数
    let matches = Command::new("Sequence Alignment")
        .arg(Arg::new("input")
            .short('i')
            .long("input")
            .takes_value(true)
            .required(true)
            .help("Input FASTA file"))
        .get_matches();

    // 获取输入文件路径
    let input_file = matches.value_of("input").expect("输入文件未指定");

    // 打开 FASTA 文件并读取内容
    let reader = fasta::Reader::from_file(input_file).expect("打开文件失败");

    // 读取所有序列
    let records: Vec<_> = reader.records().collect::<Result<_, _>>().expect("读取序列时出错");

    // 确保至少有两条序列
    if records.len() < 2 {
        panic!("FASTA 文件中必须至少有两条序列");
    }

    let a_seq = String::from_utf8_lossy(records[0].seq()).to_string();
    let b_seq = String::from_utf8_lossy(records[1].seq()).to_string();

    // 初始化比对对象
    let mut aligner = SequenceAlignment::new(a_seq, b_seq, 5, 1, -3);

    // 初始化矩阵
    aligner.initialize_matrix();

    // 计算打分矩阵
    aligner.calculate_matrix_parallel();

    // 回溯得到比对结果
    aligner.traceback();

    // 输出比对结果
    aligner.print_alignments();
}
