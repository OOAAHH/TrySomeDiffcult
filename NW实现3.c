#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <limits.h>

// 定义了我们将保存的最佳路径数量
#define MAX_RESULTS 5000

// 这是我们保存路径的文件名
#define OUTPUT_FILE "alignment_results.txt"

// 以下三个数字是比对时的得分参数：
// a 是插入或删除的惩罚（gap penalty）
// b 是错配的惩罚（mismatch penalty）
// c 是匹配的奖励（match reward）
int a = -1;
int b = -6;
int c = 10;

// 这是我们要比对的两个DNA序列，它们由字符表示
char A[] = "ATCGGGCTACATCGGGCTACGGATCGGGCTACGAAAAAAAAAAAAAAAAAAAAAA";
char B[] = "ATCGGATCGGGCTACGATCGGGCTACGATCGCGTTTTTTTAAAAAGAAAAAAAAGGGGGGGGTGTATTGTA";

// 这个矩阵将用于存储比对过程中的分数
int **matrix;
int len_A, len_B;

// 这是一个结构体（类似于一个容器），用于存储路径和分数
typedef struct {
    int i, j;     // 当前路径的坐标位置
    char* path;   // 当前路径的字符串表示
    int score;    // 当前路径的得分
} PathNode;

// 这是一个优先队列的结构体，它帮助我们找到得分最高的路径
typedef struct {
    PathNode **data;  // 存储路径节点的数组
    int size;         // 当前路径节点的数量
    int capacity;     // 队列的最大容量
} PriorityQueue;

// 这个函数初始化了用于存储得分的矩阵
void initialize_matrix() {
    matrix = (int **)malloc((len_A + 1) * sizeof(int *));
    for (int i = 0; i <= len_A; i++) {
        matrix[i] = (int *)malloc((len_B + 1) * sizeof(int));
        matrix[i][0] = i * a;
    }
    for (int j = 0; j <= len_B; j++) {
        matrix[0][j] = j * a;
    }
}

// 这个函数释放了之前分配的矩阵内存
void free_matrix() {
    for (int i = 0; i <= len_A; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// 这个函数根据比对规则计算每个位置的得分
void calculate_scores() {
    for (int i = 1; i <= len_A; i++) {
        for (int j = 1; j <= len_B; j++) {
            // 如果两个字符匹配，得分更高
            int match_mismatch = (A[i-1] == B[j-1]) ? c : b;
            int score_diag = matrix[i-1][j-1] + match_mismatch;
            int score_up = matrix[i-1][j] + a;
            int score_left = matrix[i][j-1] + a;
            matrix[i][j] = (score_diag > score_up) ? (score_diag > score_left ? score_diag : score_left) : (score_up > score_left ? score_up : score_left);
        }
    }
}

// 这个函数将字符串反转，比如 "abc" 变成 "cba"
void reverse_string(char* str) {
    int n = strlen(str);
    for (int i = 0; i < n / 2; i++) {
        char temp = str[i];
        str[i] = str[n - i - 1];
        str[n - i - 1] = temp;
    }
}

// 这个函数根据路径生成两个序列的对齐结果
void generate_alignment(char* path, char* alignA, char* alignB) {
    int i = len_A;
    int j = len_B;
    int k = 0;

    // 遍历路径字符串，根据方向进行相应的字符对齐
    for (int p = strlen(path) - 1; p >= 0; p--) {
        if (path[p] == '2') {
            alignA[k] = A[i-1];
            alignB[k] = B[j-1];
            i--;
            j--;
        } else if (path[p] == '0') {
            alignA[k] = A[i-1];
            alignB[k] = '-';
            i--;
        } else if (path[p] == '1') {
            alignA[k] = '-';
            alignB[k] = B[j-1];
            j--;
        }
        k++;
    }
    alignA[k] = '\0';
    alignB[k] = '\0';

    // 将生成的对齐字符串反转，使其与原始序列对应
    reverse_string(alignA);
    reverse_string(alignB);
}

// 这个函数创建了一个优先队列，用于管理路径
PriorityQueue* create_priority_queue(int capacity) {
    PriorityQueue *pq = (PriorityQueue *)malloc(sizeof(PriorityQueue));
    pq->data = (PathNode **)malloc(capacity * sizeof(PathNode *));
    pq->size = 0;
    pq->capacity = capacity;
    return pq;
}

// 这个函数释放优先队列中分配的内存
void free_priority_queue(PriorityQueue *pq) {
    for (int i = 0; i < pq->size; i++) {
        free(pq->data[i]->path);
        free(pq->data[i]);
    }
    free(pq->data);
    free(pq);
}

// 这个函数将路径节点插入优先队列，优先队列根据得分自动排序
void push(PriorityQueue *pq, PathNode *node) {
    if (pq->size == pq->capacity) {
        return; // 如果队列已满，不再插入新的节点
    }
    pq->data[pq->size] = node;
    int current = pq->size;
    pq->size++;
    while (current > 0) {
        int parent = (current - 1) / 2;
        if (pq->data[current]->score <= pq->data[parent]->score) {
            break;
        }
        PathNode *temp = pq->data[current];
        pq->data[current] = pq->data[parent];
        pq->data[parent] = temp;
        current = parent;
    }
}

// 这个函数从优先队列中弹出得分最高的路径节点
PathNode* pop(PriorityQueue *pq) {
    if (pq->size == 0) {
        return NULL; // 队列为空时返回空
    }
    PathNode *top = pq->data[0];
    pq->size--;
    pq->data[0] = pq->data[pq->size];
    int current = 0;
    while (1) {
        int left = 2 * current + 1;
        int right = 2 * current + 2;
        int largest = current;
        if (left < pq->size && pq->data[left]->score > pq->data[largest]->score) {
            largest = left;
        }
        if (right < pq->size && pq->data[right]->score > pq->data[largest]->score) {
            largest = right;
        }
        if (largest == current) {
            break;
        }
        PathNode *temp = pq->data[current];
        pq->data[current] = pq->data[largest];
        pq->data[largest] = temp;
        current = largest;
    }
    return top;
}

// 这个函数执行最佳优先搜索算法，计算比对过程中得分最高的路径
void calculate_directions_best_first(char **results, int *result_count) {
    // 创建优先队列，保存路径节点
    PriorityQueue *pq = create_priority_queue(MAX_RESULTS);
    
    // 初始化第一个节点
    PathNode *initial = (PathNode *)malloc(sizeof(PathNode));
    initial->i = len_A;
    initial->j = len_B;
    initial->path = strdup("");  // strdup用于复制字符串
    initial->score = matrix[len_A][len_B];
    push(pq, initial);

    // 只要队列不为空且结果数量未达到最大限制，就继续处理
    while (pq->size > 0 && *result_count < MAX_RESULTS) {
        PathNode *current = pop(pq);  // 取出得分最高的节点
        int i = current->i;
        int j = current->j;
        char *path = current->path;

        // 如果路径已经到达矩阵的左上角（0,0），说明比对完成
        if (i == 0 && j == 0) {
            results[*result_count] = path;  // 保存路径
            (*result_count)++;
            free(current);  // 释放当前节点
            continue;
        }

        // 向左扩展路径，表示插入一个gap
        if (i > 0 && matrix[i][j] == matrix[i-1][j] + a) {
            PathNode *next = (PathNode *)malloc(sizeof(PathNode));
            next->i = i - 1;
            next->j = j;
            next->path = (char *)malloc((strlen(path) + 2) * sizeof(char));
            sprintf(next->path, "%s0", path);
            next->score = matrix[next->i][next->j];
            push(pq, next);
        }

        // 向上扩展路径，表示删除一个gap
        if (j > 0 && matrix[i][j] == matrix[i][j-1] + a) {
            PathNode *next = (PathNode *)malloc(sizeof(PathNode));
            next->i = i;
            next->j = j - 1;
            next->path = (char *)malloc((strlen(path) + 2) * sizeof(char));
            sprintf(next->path, "%s1", path);
            next->score = matrix[next->i][next->j];
            push(pq, next);
        }

        // 向左上方扩展路径，表示匹配或错配
        if (i > 0 && j > 0 && matrix[i][j] == matrix[i-1][j-1] + ((A[i-1] == B[j-1]) ? c : b)) {
            PathNode *next = (PathNode *)malloc(sizeof(PathNode));
            next->i = i - 1;
            next->j = j - 1;
            next->path = (char *)malloc((strlen(path) + 2) * sizeof(char));
            sprintf(next->path, "%s2", path);
            next->score = matrix[next->i][next->j];
            push(pq, next);
        }

        free(current->path);  // 释放当前路径
        free(current);        // 释放当前节点
    }

    free_priority_queue(pq);  // 释放优先队列
}

// 这个函数将路径比对结果写入到文件中
void write_results_to_file(char **results, int result_count) {
    FILE *file = fopen(OUTPUT_FILE, "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        return;
    }

    for (int i = 0; i < result_count; i++) {
        int align_len = strlen(results[i]) + 1;
        char *alignA = (char *)malloc(align_len * sizeof(char));
        char *alignB = (char *)malloc(align_len * sizeof(char));

        generate_alignment(results[i], alignA, alignB);
        fprintf(file, "Path %d: %s\n", i+1, results[i]);
        fprintf(file, "Alignment:\n%s\n%s\n\n", alignA, alignB);
        
        free(results[i]);  // 释放路径内存
        free(alignA);      // 释放对齐字符串内存
        free(alignB);      // 释放对齐字符串内存
    }

    fclose(file);
}

// 主函数，程序从这里开始执行
int main() {
    len_A = strlen(A);
    len_B = strlen(B);
    
    // 初始化矩阵并计算得分
    initialize_matrix();
    calculate_scores();
    
    // 用于存储前5000条最佳路径
    char *results[MAX_RESULTS];
    int result_count = 0;
    
    // 执行最佳优先搜索算法，计算最优路径
    calculate_directions_best_first(results, &result_count);
    
    printf("Total Paths: %d\n", result_count);
    
    // 将路径结果写入文件
    write_results_to_file(results, result_count);
    
    // 释放矩阵内存
    free_matrix();
    
    return 0;
}
