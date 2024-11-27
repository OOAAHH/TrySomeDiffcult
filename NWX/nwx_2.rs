use ndarray::Array2;
use rayon::prelude::*;

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
    a_score: i32,
    b_score: i32,
    c: i32,
    len_a: usize,
    len_b: usize,
    matrix: Array2<i32>,
    traceback_matrix: Vec<Vec<Vec<Direction>>>,
    aligned_sequences: Vec<(String, String)>,
}

impl SequenceAlignment {
    fn calculate_score(&self, aligned_a: &Vec<char>, aligned_b: &Vec<char>) -> i32 {
        let mut score = 0;
        let mut i = aligned_a.len();
        let mut j = aligned_b.len();

        while i > 0 || j > 0 {
            if i > 0 && j > 0 && aligned_a[i - 1] == aligned_b[j - 1] {
                score += self.c; // 匹配
                i -= 1;
                j -= 1;
            } else if i > 0 && (aligned_a[i - 1] != '-') {
                score += self.a_score; // A的惩罚
                i -= 1;
            } else if j > 0 && (aligned_b[j - 1] != '-') {
                score += self.a_score; // B的惩罚
                j -= 1;
            }
        }

        score
    }

    /// 创建新的序列比对对象
    fn new(a: String, b: String, a_score: i32, b_score: i32, c: i32) -> Self {
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
            a_score,
            b_score,
            c,
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
            self.matrix[[i, 0]] = (i as i32) * self.a_score;
            if i > 0 {
                self.traceback_matrix[i][0].push(Direction::Up);
            }
        }

        // 初始化第一行
        for j in 0..=self.len_b {
            self.matrix[[0, j]] = (j as i32) * self.a_score;
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
            self.matrix[[i - 1, j - 1]] + self.c
        } else {
            self.matrix[[i - 1, j - 1]] + self.b_score
        };
        let up_score = self.matrix[[i - 1, j]] + self.a_score;
        let left_score = self.matrix[[i, j - 1]] + self.a_score;

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
        let mut stack: Vec<(usize, usize, Vec<char>, Vec<char>)> =
            vec![(self.len_a, self.len_b, Vec::new(), Vec::new())];

        while let Some((i, j, aligned_a, aligned_b)) = stack.pop() {
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
                            stack.push((i - 1, j - 1, new_aligned_a, new_aligned_b));
                        }
                    }
                    Direction::Up => {
                        if i > 0 {
                            let mut new_aligned_a = aligned_a.clone();
                            let mut new_aligned_b = aligned_b.clone();
                            new_aligned_a.push(self.a[i - 1]);
                            new_aligned_b.push('-');
                            stack.push((i - 1, j, new_aligned_a, new_aligned_b));
                        }
                    }
                    Direction::Left => {
                        if j > 0 {
                            let mut new_aligned_a = aligned_a.clone();
                            let mut new_aligned_b = aligned_b.clone();
                            new_aligned_a.push('-');
                            new_aligned_b.push(self.b[j - 1]);
                            stack.push((i, j - 1, new_aligned_a, new_aligned_b));
                        }
                    }
                }
            }
        }
    }

    /// 打印所有比对结果和对应的得分
    fn print_alignments(&self) {
        for (idx, (aligned_a, aligned_b)) in self.aligned_sequences.iter().enumerate() {
            let score = self.calculate_score(&aligned_a.chars().collect(), &aligned_b.chars().collect());
            println!("Alignment {}:", idx + 1);
            println!("Score: {}", score);
            println!("Aligned Sequence A: {}", aligned_a);
            println!("Aligned Sequence B: {}", aligned_b);
            println!();
        }
    }

}

fn main() {
    let a = "GATTAGGATTACGATTACAAAGATTGATTAGGATTACGATTACAAAGATTACATTACACAACATTACACA".to_string();
    let b = "GCATGAGATTACATTGATTACAACGATTAGGATTACGATTACAAAGATTACATTACACAAGGATTACACU".to_string();

    // 初始化对象
    let mut aligner = SequenceAlignment::new(a, b, -1, -1, 2);

    // 初始化矩阵
    aligner.initialize_matrix();

    // 计算打分矩阵
    aligner.calculate_matrix_parallel();

    // 回溯得到比对结果，输出所有路径
    aligner.traceback();

    // 输出比对结果
    aligner.print_alignments();
}
