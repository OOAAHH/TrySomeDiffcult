use ndarray::Array2;
use rayon::prelude::*;
use bio::io::fasta;
use clap::Parser;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

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

    /// 回溯以获取比对结果（并行化）
    fn traceback_parallel(&mut self, max_paths: usize) {
        self.aligned_sequences.clear();
        let stack: Arc<Mutex<Vec<(usize, usize, Vec<char>, Vec<char>)>>> =
            Arc::new(Mutex::new(vec![(self.len_a, self.len_b, Vec::new(), Vec::new())]));
        let aligned_sequences: Arc<Mutex<Vec<(String, String)>>> =
            Arc::new(Mutex::new(Vec::new()));
        let max_paths = Arc::new(AtomicUsize::new(max_paths));

        let num_threads = num_cpus::get();
        let mut handles = vec![];

        for _ in 0..num_threads {
            let stack = Arc::clone(&stack);
            let aligned_sequences = Arc::clone(&aligned_sequences);
            let max_paths = Arc::clone(&max_paths);
            let a = self.a.clone();
            let b = self.b.clone();
            let traceback_matrix = self.traceback_matrix.clone();

            handles.push(std::thread::spawn(move || {
                loop {
                    let current = {
                        let mut stack_guard = stack.lock().unwrap();
                        stack_guard.pop()
                    };
                    match current {
                        Some((i, j, aligned_a, aligned_b)) => {
                            if aligned_sequences.lock().unwrap().len() >= max_paths.load(Ordering::SeqCst) {
                                break;
                            }
                            if i == 0 && j == 0 {
                                // 已到达起点，保存路径
                                let aligned_a_str: String = aligned_a.into_iter().rev().collect();
                                let aligned_b_str: String = aligned_b.into_iter().rev().collect();
                                let mut sequences_guard = aligned_sequences.lock().unwrap();
                                if sequences_guard.len() < max_paths.load(Ordering::SeqCst) {
                                    sequences_guard.push((aligned_a_str, aligned_b_str));
                                }
                                continue;
                            }

                            for &direction in &traceback_matrix[i][j] {
                                match direction {
                                    Direction::Diag => {
                                        if i > 0 && j > 0 {
                                            let mut new_aligned_a = aligned_a.clone();
                                            let mut new_aligned_b = aligned_b.clone();
                                            new_aligned_a.push(a[i - 1]);
                                            new_aligned_b.push(b[j - 1]);
                                            stack.lock().unwrap().push((i - 1, j - 1, new_aligned_a, new_aligned_b));
                                        }
                                    }
                                    Direction::Up => {
                                        if i > 0 {
                                            let mut new_aligned_a = aligned_a.clone();
                                            let mut new_aligned_b = aligned_b.clone();
                                            new_aligned_a.push(a[i - 1]);
                                            new_aligned_b.push('-');
                                            stack.lock().unwrap().push((i - 1, j, new_aligned_a, new_aligned_b));
                                        }
                                    }
                                    Direction::Left => {
                                        if j > 0 {
                                            let mut new_aligned_a = aligned_a.clone();
                                            let mut new_aligned_b = aligned_b.clone();
                                            new_aligned_a.push('-');
                                            new_aligned_b.push(b[j - 1]);
                                            stack.lock().unwrap().push((i, j - 1, new_aligned_a, new_aligned_b));
                                        }
                                    }
                                }
                            }
                        }
                        None => break,
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        self.aligned_sequences = Arc::try_unwrap(aligned_sequences)
            .expect("Too many Arc references")
            .into_inner()
            .expect("Mutex poisoned");
    }

    /// 打印所有比对结果
    fn print_alignments(&self, writer: &mut impl Write) -> io::Result<()> {
        for (idx, (aligned_a, aligned_b)) in self.aligned_sequences.iter().enumerate() {
            writeln!(writer, "Alignment {}:", idx + 1)?;
            writeln!(writer, "Aligned Sequence A: {}", aligned_a)?;
            writeln!(writer, "Aligned Sequence B: {}", aligned_b)?;
            writeln!(writer)?;
        }
        Ok(())
    }

    /// 获取打分矩阵
    fn get_matrix(&self) -> &Array2<i32> {
        &self.matrix
    }

    /// 打印打分矩阵
    fn print_matrix(&self, writer: &mut impl Write) -> io::Result<()> {
        for i in 0..=self.len_a {
            for j in 0..=self.len_b {
                write!(writer, "{:4} ", self.matrix[[i, j]])?;
            }
            writeln!(writer)?;
        }
        Ok(())
    }
}

/// 定义命令行参数结构体
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 第一个序列的 FASTA 文件
    #[arg(short = 'a', long = "fasta1", value_name = "FILE")]
    fasta1: PathBuf,

    /// 第二个序列的 FASTA 文件
    #[arg(short = 'b', long = "fasta2", value_name = "FILE")]
    fasta2: PathBuf,

    /// 输出比对结果的文件，默认为标准输出
    #[arg(short = 'o', long = "output", value_name = "FILE")]
    output: Option<PathBuf>,

    /// 设置匹配分数（默认：2）
    #[arg(short = 'c', long = "match_score", default_value_t = 2)]
    match_score: i32,

    /// 设置不匹配分数（默认：-1）
    #[arg(short = 'd', long = "mismatch_score", default_value_t = -1)]
    mismatch_score: i32,

    /// 设置间隙分数（默认：-1）
    #[arg(short = 'g', long = "gap_score", default_value_t = -1)]
    gap_score: i32,

    /// 指定最多输出的比对路径数（默认：1）
    #[arg(short = 'm', long = "max_paths", default_value_t = 1)]
    max_paths: usize,
}

fn main() -> io::Result<()> {
    let start_total = Instant::now();

    // 解析命令行参数
    let args = Args::parse();

    // 读取并解析第一个 FASTA 文件
    let start = Instant::now();
    let mut reader1 = fasta::Reader::from_file(&args.fasta1)
        .unwrap_or_else(|e| panic!("无法读取 FASTA 文件 {}: {}", args.fasta1.display(), e));
    let mut records1 = reader1.records();
    let mut seq1_count = 0;
    let mut seq1_str = String::new();
    while let Some(record) = records1.next() {
        match record {
            Ok(rec) => {
                seq1_count += 1;
                seq1_str = String::from_utf8(rec.seq().to_owned())
                    .unwrap_or_else(|e| panic!("FASTA 文件 {} 中的序列不是有效的 UTF-8: {}", args.fasta1.display(), e));
            }
            Err(e) => panic!("读取 FASTA 文件 {} 时出错: {}", args.fasta1.display(), e),
        }
    }
    let duration = start.elapsed();
    println!("读取第一个 FASTA 文件耗时: {:?}", duration);
    println!("FASTA 文件 {} 中的序列数: {}", args.fasta1.display(), seq1_count);

    // 读取并解析第二个 FASTA 文件
    let start = Instant::now();
    let mut reader2 = fasta::Reader::from_file(&args.fasta2)
        .unwrap_or_else(|e| panic!("无法读取 FASTA 文件 {}: {}", args.fasta2.display(), e));
    let mut records2 = reader2.records();
    let mut seq2_count = 0;
    let mut seq2_str = String::new();
    while let Some(record) = records2.next() {
        match record {
            Ok(rec) => {
                seq2_count += 1;
                seq2_str = String::from_utf8(rec.seq().to_owned())
                    .unwrap_or_else(|e| panic!("FASTA 文件 {} 中的序列不是有效的 UTF-8: {}", args.fasta2.display(), e));
            }
            Err(e) => panic!("读取 FASTA 文件 {} 时出错: {}", args.fasta2.display(), e),
        }
    }
    let duration = start.elapsed();
    println!("读取第二个 FASTA 文件耗时: {:?}", duration);
    println!("FASTA 文件 {} 中的序列数: {}", args.fasta2.display(), seq2_count);

    // 初始化序列比对对象
    let start = Instant::now();
    let mut aligner = SequenceAlignment::new(
        seq1_str,
        seq2_str,
        args.gap_score,
        args.mismatch_score,
        args.match_score,
    );
    let duration = start.elapsed();
    println!("初始化序列比对对象耗时: {:?}", duration);

    // 初始化矩阵
    let start = Instant::now();
    aligner.initialize_matrix();
    let duration = start.elapsed();
    println!("初始化矩阵耗时: {:?}", duration);

    // 计算打分矩阵
    let start = Instant::now();
    aligner.calculate_matrix_parallel();
    let duration = start.elapsed();
    println!("计算打分矩阵耗时: {:?}", duration);

    // 回溯得到比对结果
    let start = Instant::now();
    aligner.traceback_parallel(args.max_paths);
    let duration = start.elapsed();
    println!("回溯得到比对结果耗时: {:?}", duration);

    // 打开输出文件或使用标准输出
    let start = Instant::now();
    match &args.output {
        Some(path) => {
            let file = File::create(path)
                .unwrap_or_else(|e| panic!("无法创建输出文件 {}: {}", path.display(), e));
            let mut writer = io::BufWriter::new(file);
            aligner.print_alignments(&mut writer)?;
        }
        None => {
            // 使用标准输出
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            aligner.print_alignments(&mut handle)?;
        }
    }
    let duration = start.elapsed();
    println!("输出比对结果耗时: {:?}", duration);

    let duration_total = start_total.elapsed();
    println!("总耗时: {:?}", duration_total);

    Ok(())
}
