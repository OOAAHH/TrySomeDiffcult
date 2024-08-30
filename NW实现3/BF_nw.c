#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <limits.h>

#define MAX_RESULTS 15
#define OUTPUT_FILE "alignment_results.txt"

// Gap, mismatch penalty, and match reward
int a = -1;
int b = -6;
int c = 10;

char A[] = "atttattttatttaccttaaaactagtattttaataaataaaattattttttcttatttatttaattataaaaacctcattatttttttaaaactctatttatttttaaataaaatattttttaatttattttacgaaaaatgagatattaatatataatgactgaaatgtaaagattttacaagtataagtattaaatgagtaattgaaaaaaagtttaatcttataaataatgctttttttccttctccatatatcatgaacacaataatacacaatccctttgaaattaaatacactttaggacaaaacatgaccccttaaaaagagatagtacctcccattccagttacatttgtgacaaagttagaaagaaggagaaagaaagaagatggcaggaaagaaaatcccacatgtgttgctgaattctggacacaaaatgccagttataggcatgggaacttcagtggagaatcgtccatcaaatgagacccttgcttcaatctatgttgaagccattgaggttggttaccgtcattttgacactgctgcagtgtatggaacagaggaagccataggcctggccgtggccaatgccatagaaaaaggcctaataaagagtagagatgaagttttcatcacttcaaaaccttggaacacagatgcacgccgtgatcttattgtcccagctctcaagaccacattaaagtacatataatctccattctcacattgtttctaatgatcttaatattgtgtttgtgctcactttttaattttattttattttgtggacaaaacttagatgcggtttcatataagcttagtgttgttctacaacttgaataggagtttcacctcggtaatacacattaacatttgatggaaacctttattacacggattacctgagtgaaactccaattctgattggagaacagcgccaaaagcactaatagaactacatctaagttttgtaattgcttttattatttttctataactactaatgttgtttgagtactttatgtgccacacaatgttgtttaggaagctggggacgcagtatgtggatctttatctgattcattggccagtgaggctgagacatgatctttaaaatccaactgtttttaccaaggaagattttctgccctttgatatagaagggacatggaaagctatggaagagtgttacaagttgggattagcaaagtccattggcatatgcaattatggtgtcaaaaaactcaccaaactcttggaaatagccacctttcctcctgcagtcaatcaggtaccataatttgattactccagagatgtatatttggtttggcacatagttattatgatggtcttggacttcttggtactcatgttatgatgttgaattattgttttttcaggtggaaatgaacccatcttggcagcaaggaaaactcagggagt";
char B[] = "actccctgagttttccttgctgccaagatgggttcatttccacctgaaaaaacaataattcaacatcataacatgagtaccaagaagtccaagaccatcataataactatgtgccaaaccaaatatacatctctggagtaatcaaattatggtacctgattgactgcaggaggaaaggtggctatttccaagagtttggtgagttttttgacaccataattgcatatgccaatggactttgctaatcccaacttgtaacactcttccatagctttccatgtcccttctatatcaaagggcagaaaatcttccttggtaaaaacagttggattttaaagatcatgtctcagcctcactggccaatgaatcagataaagatccacatactgcgtccccagcttcctaaacaacattgtgtggcacataaagtactcaaacaacattagtagttatagaaaaataataaaagcaattacaaaacttagatgtagttctattagtgcttttggcgctgttctccaatcagaattggagtttcactcaggtaatccgtgtaataaaggtttccatcaaatgttaatgtgtattaccgaggtgaaactcctattcaagttgtagaacaacactaagcttatatgaaaccgcatctaagttttgtccacaaaataaaataaaattaaaaagtgagcacaaacacaatattaagatcattagaaacaatgtgagaatggagattatatgtactttaatgtggtcttgagagctgggacaataagatcacggcgtgcatctgtgttccaaggttttgaagtgatgaaaacttcatctctactctttattaggcctttttctatggcattggccacggccaggcctatggcttcctctgttccatacactgcagcagtgtcaaaatgacggtaaccaacctcaatggcttcaacatagattgaagcaagggtctcatttgatggacgattctccactgaagttcccatgcctataactggcattttgtgtccagaattcagcaacacatgtgggattttctttcctgccatcttctttctttctccttctttctaactttgtcacaaatgtaactggaatggaggtactatctctttttaaggggtcagttttgtcctaaagtgtatttaatttcaaagggattgtgtattattgtgttcatgatatatggagaaggaaaaaaagcattatttataagattaaactttttttcaattactcatttaatacttatacttgtaaaatctttacatttcagtcattatatattaatatctcatttttcgtaaaataaattaaaaaatattttatttaaaaataaatagagttttaaaaaaataatgaggtttttataattaaataaataagaaaaaataattttatttattaaaatactagttttaaggtaaataaaataaat";

int **matrix;
int len_A, len_B;

typedef struct {
    int i, j;
    char* path;
    int score;
} PathNode;

typedef struct {
    PathNode **data;
    int size;
    int capacity;
} PriorityQueue;

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

void free_matrix() {
    for (int i = 0; i <= len_A; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void calculate_scores() {
    for (int i = 1; i <= len_A; i++) {
        for (int j = 1; j <= len_B; j++) {
            int match_mismatch = (A[i-1] == B[j-1]) ? c : b;
            int score_diag = matrix[i-1][j-1] + match_mismatch;
            int score_up = matrix[i-1][j] + a;
            int score_left = matrix[i][j-1] + a;
            matrix[i][j] = (score_diag > score_up) ? (score_diag > score_left ? score_diag : score_left) : (score_up > score_left ? score_up : score_left);
        }
    }
}

void reverse_string(char* str) {
    int n = strlen(str);
    for (int i = 0; i < n / 2; i++) {
        char temp = str[i];
        str[i] = str[n - i - 1];
        str[n - i - 1] = temp;
    }
}

void generate_alignment(char* path, char* alignA, char* alignB) {
    int i = len_A;
    int j = len_B;
    int k = 0;

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

    reverse_string(alignA);
    reverse_string(alignB);
}

PriorityQueue* create_priority_queue(int capacity) {
    PriorityQueue *pq = (PriorityQueue *)malloc(sizeof(PriorityQueue));
    pq->data = (PathNode **)malloc(capacity * sizeof(PathNode *));
    pq->size = 0;
    pq->capacity = capacity;
    return pq;
}

void free_priority_queue(PriorityQueue *pq) {
    for (int i = 0; i < pq->size; i++) {
        free(pq->data[i]->path);
        free(pq->data[i]);
    }
    free(pq->data);
    free(pq);
}

void push(PriorityQueue *pq, PathNode *node) {
    if (pq->size == pq->capacity) {
        return; // Priority queue is full
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

PathNode* pop(PriorityQueue *pq) {
    if (pq->size == 0) {
        return NULL;
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

void calculate_directions_best_first(char **results, int *result_count) {
    PriorityQueue *pq = create_priority_queue(MAX_RESULTS);
    PathNode *initial = (PathNode *)malloc(sizeof(PathNode));
    initial->i = len_A;
    initial->j = len_B;
    initial->path = strdup("");
    initial->score = matrix[len_A][len_B];
    push(pq, initial);

    while (pq->size > 0 && *result_count < MAX_RESULTS) {
        PathNode *current = pop(pq);
        int i = current->i;
        int j = current->j;
        char *path = current->path;

        if (i == 0 && j == 0) {
            results[*result_count] = path;
            (*result_count)++;
            free(current);
            continue;
        }

        if (i > 0 && matrix[i][j] == matrix[i-1][j] + a) {
            PathNode *next = (PathNode *)malloc(sizeof(PathNode));
            next->i = i - 1;
            next->j = j;
            next->path = (char *)malloc((strlen(path) + 2) * sizeof(char));
            sprintf(next->path, "%s0", path);
            next->score = matrix[next->i][next->j];
            push(pq, next);
        }
        if (j > 0 && matrix[i][j] == matrix[i][j-1] + a) {
            PathNode *next = (PathNode *)malloc(sizeof(PathNode));
            next->i = i;
            next->j = j - 1;
            next->path = (char *)malloc((strlen(path) + 2) * sizeof(char));
            sprintf(next->path, "%s1", path);
            next->score = matrix[next->i][next->j];
            push(pq, next);
        }
        if (i > 0 && j > 0 && matrix[i][j] == matrix[i-1][j-1] + ((A[i-1] == B[j-1]) ? c : b)) {
            PathNode *next = (PathNode *)malloc(sizeof(PathNode));
            next->i = i - 1;
            next->j = j - 1;
            next->path = (char *)malloc((strlen(path) + 2) * sizeof(char));
            sprintf(next->path, "%s2", path);
            next->score = matrix[next->i][next->j];
            push(pq, next);
        }

        free(current->path);
        free(current);
    }

    free_priority_queue(pq);
}

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
        
        free(results[i]);  // Free the path memory
        free(alignA);      // Free the alignment string memory
        free(alignB);      // Free the alignment string memory
    }

    fclose(file);
}

int main() {
    len_A = strlen(A);
    len_B = strlen(B);
    
    initialize_matrix();
    calculate_scores();
    
    char *results[MAX_RESULTS];  // Store top 5000 paths
    int result_count = 0;
    
    calculate_directions_best_first(results, &result_count);
    
    printf("Total Paths: %d\n", result_count);
    write_results_to_file(results, result_count);
    
    free_matrix();  // Free matrix memory
    
    return 0;
}
