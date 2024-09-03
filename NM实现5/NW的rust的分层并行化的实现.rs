use ndarray::{Array2, s};
use std::fs::File;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

struct NeedlemanWunschOptimized {
    match_score: i32,
    mismatch_score: i32,
    gap_score: i32,
    max_subprocesses: usize,
    max_nodes_per_process: usize,
    active_processes: Arc<Mutex<Vec<Option<thread::JoinHandle<()>>>>>,
    total_paths: Arc<Mutex<usize>>,  // 新增路径计数
}

impl NeedlemanWunschOptimized {
    fn new(match_score: i32, mismatch_score: i32, gap_score: i32) -> Self {
        NeedlemanWunschOptimized {
            match_score,
            mismatch_score,
            gap_score,
            max_subprocesses: 70,
            max_nodes_per_process: 50,
            active_processes: Arc::new(Mutex::new(Vec::new())),
            total_paths: Arc::new(Mutex::new(0)),  // 初始化路径计数
        }
    }

    fn align(&self, a: &str, b: &str) -> Vec<(String, String)> {
        let len_a = a.len();
        let len_b = b.len();

        let mut directions = Array2::from_elem((len_a + 1, len_b + 1), Vec::new());
        let a_bytes = a.as_bytes();
        let b_bytes = b.as_bytes();

        let mut scores = Array2::<i32>::zeros((len_a + 1, len_b + 1));
        for j in 1..=len_b {
            scores[(0, j)] = j as i32 * self.gap_score;
        }

        for i in 1..=len_a {
            scores[(i, 0)] = i as i32 * self.gap_score;

            let mut previous_row = scores.slice(s![i - 1, ..]).to_vec();

            // 对每一列逐个计算
            for j in 1..=len_b {
                let match_score = if a_bytes[i - 1] == b_bytes[j - 1] {
                    previous_row[j - 1] + self.match_score
                } else {
                    previous_row[j - 1] + self.mismatch_score
                };
                let delete_score = previous_row[j] + self.gap_score;
                let insert_score = scores[(i, j - 1)] + self.gap_score;

                let max_score = *[match_score, delete_score, insert_score]
                    .iter()
                    .max()
                    .unwrap();

                let mut direction = Vec::new();
                if max_score == delete_score {
                    direction.push('u');
                }
                if max_score == insert_score {
                    direction.push('l');
                }
                if max_score == match_score {
                    direction.push('d');
                }

                scores[(i, j)] = max_score;
                directions[(i, j)] = direction;
            }

            let num_directions: usize = directions.slice(s![i, ..])
                .iter()
                .map(|d| d.len())
                .sum();

            println!("当前处理的层是：{}，上一次主进程计算得到的新增方向数量是：{}", i, num_directions);

            if num_directions <= self.max_nodes_per_process {
                println!("本次计算的进程：1 核心。主进程1个，计算方向 {} 个。", num_directions);
            } else {
                // 子进程的调度逻辑（如之前代码）
                self.spawn_subprocess(i, directions.slice(s![i, ..]).to_vec());
            }

            for j in 1..=len_b {
                println!("层: {} 列: {} 分数: {} 方向: {:?}", i, j, scores[(i, j)], directions[(i, j)]);
            }

            self.monitor_processes();
        }

        self.merge_results();

        let alignments = self.traceback(a, b, &directions, len_a, len_b, String::new(), String::new());

        // 输出路径数量
        let total_paths = *self.total_paths.lock().unwrap();
        println!("以及完成寻路计算，一共存储了 {} 条路径。", total_paths);
        println!("开始执行路径到比对结果的转换.");

        self.write_alignments_to_file("alignments.txt", &alignments)
            .expect("Failed to write alignments to file");

        alignments
    }

    fn spawn_subprocess(&self, process_id: usize, nodes: Vec<Vec<char>>) {
        let active_processes = Arc::clone(&self.active_processes);
        let total_paths = Arc::clone(&self.total_paths);

        let handle = thread::spawn(move || {
            let filename = format!("result_{}.txt", process_id);
            let mut file = File::create(filename).unwrap();

            let mut path_count = 0;
            for direction in nodes {
                writeln!(file, "{:?}", direction).unwrap();
                path_count += 1;
            }

            let mut total_paths = total_paths.lock().unwrap();
            *total_paths += path_count;
        });

        active_processes.lock().unwrap().push(Some(handle));
    }

    fn monitor_processes(&self) {
        let mut active_processes = self.active_processes.lock().unwrap();

        let mut i = 0;
        while i < active_processes.len() {
            if let Some(handle) = &mut active_processes[i] {
                if handle.is_finished() {
                    let handle = active_processes.remove(i).unwrap();
                    handle.join().unwrap();
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        while active_processes.len() >= self.max_subprocesses {
            thread::sleep(Duration::from_millis(100));

            let mut i = 0;
            while i < active_processes.len() {
                if let Some(handle) = &mut active_processes[i] {
                    if handle.is_finished() {
                        let handle = active_processes.remove(i).unwrap();
                        handle.join().unwrap();
                    } else {
                        i += 1;
                    }
                } else {
                    i += 1;
                }
            }
        }
    }

    fn merge_results(&self) {
        // 合并所有子进程结果的逻辑
    }

    fn traceback(
        &self,
        a: &str,
        b: &str,
        directions: &Array2<Vec<char>>,
        i: usize,
        j: usize,
        mut align_a: String,
        mut align_b: String,
    ) -> Vec<(String, String)> {
        let mut results = Vec::new();

        if i == 0 && j == 0 {
            results.push((align_a, align_b));
        } else {
            if i > 0 && directions[(i, j)].contains(&'u') {
                let mut new_align_a = align_a.clone();
                let mut new_align_b = align_b.clone();
                new_align_a.push(a.as_bytes()[i - 1] as char);
                new_align_b.push('-');
                results.extend(self.traceback(a, b, directions, i - 1, j, new_align_a, new_align_b));
            }
            if j > 0 && directions[(i, j)].contains(&'l') {
                let mut new_align_a = align_a.clone();
                let mut new_align_b = align_b.clone();
                new_align_a.push('-');
                new_align_b.push(b.as_bytes()[j - 1] as char);
                results.extend(self.traceback(a, b, directions, i, j - 1, new_align_a, new_align_b));
            }
            if i > 0 && j > 0 && directions[(i, j)].contains(&'d') {
                let mut new_align_a = align_a.clone();
                let mut new_align_b = align_b.clone();
                new_align_a.push(a.as_bytes()[i - 1] as char);
                new_align_b.push(b.as_bytes()[j - 1] as char);
                results.extend(self.traceback(a, b, directions, i - 1, j - 1, new_align_a, new_align_b));
            }
        }

        // 更新路径计数
        let mut total_paths = self.total_paths.lock().unwrap();
        *total_paths += results.len();

        results
            .into_iter()
            .map(|(align_a, align_b)| {
                (
                    align_a.chars().rev().collect(),
                    align_b.chars().rev().collect(),
                )
            })
            .collect()
    }

    fn write_alignments_to_file(
        &self,
        filename: &str,
        alignments: &[(String, String)],
    ) -> io::Result<()> {
        let mut file = File::create(filename)?;

        for (align_a, align_b) in alignments {
            writeln!(file, "Alignment A: {}", align_a)?;
            writeln!(file, "Alignment B: {}", align_b)?;
            writeln!(file, "")?;
        }

        Ok(())
    }
}

fn main() {
    let nw = NeedlemanWunschOptimized::new(1, -1, -1);
    let alignments = nw.align(
        "atttattttatttaccttaaaactagtattttaatatagtattttaatatagtatttttagtattttaattttaataatttattttatttaccttaaaactagtattttaatatagtattttaatatagtatttttagtaatttattttatttaccttaaaactagtattttaatatagtattttaatatagtatttttagtaatttattttatttaccttaaaactagtattttaatatagtattttaatatagtatttttagtaatttattttatttaccttaaaactagtattttaatatagtattttaatatagtatttttagta",
        "attacgttcatctttcggctagttttcttcattgcattagtattttaatatagtattttaatatagtattttaataatttattttatttaccttaaaactagtattttaatatagtattttaatatagtatttttagtaatttattttatttaccttaaaactagtattttaatatagtattttaatatagtatttttagtaatttattttatttaccttaaaactagtattttaatatagtattttaatatagtatttttagtaatttattttatttaccttaaaactagtattttaatatagtattttaatatagtatttttagta"
    );

    for (align_a, align_b) in alignments {
        println!("Alignment A: {}", align_a);
        println!("Alignment B: {}", align_b);
    }
}
