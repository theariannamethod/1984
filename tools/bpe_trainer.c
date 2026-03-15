/*
 * bpe_trainer.c — Learn BPE merge table from text corpus.
 *
 * Reads text files, learns 1792 merges (256 bytes + 1792 = 2048 BPE vocab).
 * Outputs C array literal for embedding in penelope.c and friends.
 *
 *   cc bpe_trainer.c -O2 -o bpe_trainer
 *   ./bpe_trainer corpus1.txt corpus2.txt ... > merges.h
 *
 * By Arianna Method.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define BASE_VOCAB  256
#define NUM_MERGES  1792
#define BPE_VOCAB   (BASE_VOCAB + NUM_MERGES)  /* 2048 */
#define MAX_TEXT    (64 * 1024 * 1024)  /* 64MB max corpus */
#define MAX_TOKENS (MAX_TEXT * 2)

/* token string representation */
static char tok_str[BPE_VOCAB][64];
static int  tok_str_len[BPE_VOCAB];

/* pair counting: pair_count[left][right] */
/* too large for stack, use heap */
static int *pair_count;  /* [BPE_VOCAB * BPE_VOCAB] */

#define PC(l,r) pair_count[(l) * BPE_VOCAB + (r)]

/* corpus as token sequence */
static int *tokens;
static int  n_tokens;

/* linked list for efficient merge application */
static int *tok_next;  /* next index (-1 = end) */
static int *tok_prev;  /* prev index (-1 = start) */

/* merge table output */
static int merge_left[NUM_MERGES];
static int merge_right[NUM_MERGES];

static void init_base_vocab(void) {
    for (int i = 0; i < BASE_VOCAB; i++) {
        if (isprint(i) && i != '\\' && i != '"') {
            tok_str[i][0] = (char)i;
            tok_str[i][1] = '\0';
        } else {
            snprintf(tok_str[i], 64, "\\x%02x", i);
        }
        tok_str_len[i] = 1;  /* logical length = 1 byte */
    }
}

static void load_corpus(int argc, char **argv) {
    tokens = (int *)malloc(MAX_TOKENS * sizeof(int));
    n_tokens = 0;

    for (int f = 1; f < argc; f++) {
        FILE *fp = fopen(argv[f], "r");
        if (!fp) { fprintf(stderr, "cannot open %s\n", argv[f]); continue; }

        int c;
        while ((c = fgetc(fp)) != EOF && n_tokens < MAX_TOKENS) {
            /* lowercase, skip non-printable except whitespace */
            if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
            tokens[n_tokens++] = (unsigned char)c;
        }
        fclose(fp);
        fprintf(stderr, "  loaded %s (total %d bytes)\n", argv[f], n_tokens);
    }
}

static void init_linked_list(void) {
    tok_next = (int *)malloc(n_tokens * sizeof(int));
    tok_prev = (int *)malloc(n_tokens * sizeof(int));
    for (int i = 0; i < n_tokens; i++) {
        tok_next[i] = i + 1;
        tok_prev[i] = i - 1;
    }
    tok_next[n_tokens - 1] = -1;
}

static void count_all_pairs(void) {
    memset(pair_count, 0, (size_t)BPE_VOCAB * BPE_VOCAB * sizeof(int));
    int i = 0;
    while (i >= 0 && tok_next[i] >= 0) {
        int j = tok_next[i];
        if (j < 0) break;
        PC(tokens[i], tokens[j])++;
        i = j;
    }
}

static void find_best_pair(int *best_l, int *best_r, int *best_count, int next_id) {
    *best_count = 0;
    *best_l = -1;
    *best_r = -1;
    for (int l = 0; l < next_id; l++) {
        for (int r = 0; r < next_id; r++) {
            if (PC(l, r) > *best_count) {
                *best_count = PC(l, r);
                *best_l = l;
                *best_r = r;
            }
        }
    }
}

static void apply_merge(int left, int right, int new_id) {
    /* scan through linked list, replace (left, right) pairs with new_id */
    int i = 0;
    /* find first valid token */
    while (i >= 0 && i < n_tokens) {
        int j = tok_next[i];
        if (j < 0) break;

        if (tokens[i] == left && tokens[j] == right) {
            /* merge: replace i with new_id, remove j from list */
            tokens[i] = new_id;

            /* remove j from linked list */
            int k = tok_next[j];
            tok_next[i] = k;
            if (k >= 0) tok_prev[k] = i;

            /* update pair counts for neighbors */
            /* don't advance — check if new_id can merge with next */
        } else {
            i = j;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s corpus1.txt [corpus2.txt ...] > merges.h\n", argv[0]);
        return 1;
    }

    init_base_vocab();

    fprintf(stderr, "loading corpus...\n");
    load_corpus(argc, argv);
    fprintf(stderr, "  %d bytes total\n", n_tokens);

    init_linked_list();

    pair_count = (int *)calloc((size_t)BPE_VOCAB * BPE_VOCAB, sizeof(int));
    if (!pair_count) { fprintf(stderr, "OOM for pair_count\n"); return 1; }

    fprintf(stderr, "learning %d merges...\n", NUM_MERGES);

    int next_id = BASE_VOCAB;

    for (int m = 0; m < NUM_MERGES; m++) {
        count_all_pairs();

        int best_l, best_r, best_count;
        find_best_pair(&best_l, &best_r, &best_count, next_id);

        if (best_count < 2) {
            fprintf(stderr, "  stopping at merge %d: best pair count = %d\n", m, best_count);
            break;
        }

        merge_left[m] = best_l;
        merge_right[m] = best_r;

        /* build string for new token */
        snprintf(tok_str[next_id], 64, "%s%s", tok_str[best_l], tok_str[best_r]);
        tok_str_len[next_id] = tok_str_len[best_l] + tok_str_len[best_r];

        apply_merge(best_l, best_r, next_id);

        if (m % 100 == 0 || m == NUM_MERGES - 1) {
            fprintf(stderr, "  merge %4d/%d: (%3d, %3d) -> %d  count=%d  \"%s\"\n",
                    m, NUM_MERGES, best_l, best_r, next_id, best_count, tok_str[next_id]);
        }

        next_id++;
    }

    int actual_merges = next_id - BASE_VOCAB;
    fprintf(stderr, "learned %d merges\n", actual_merges);

    /* output C header */
    printf("/* BPE merge table — %d merges, learned from %d bytes of English text */\n",
           actual_merges, n_tokens);
    printf("/* Generated by bpe_trainer.c. Do not edit. */\n\n");
    printf("#define BPE_VOCAB   %d\n", BASE_VOCAB + actual_merges);
    printf("#define BPE_MERGES  %d\n\n", actual_merges);
    printf("static const int BPE_TABLE[BPE_MERGES][2] = {\n");
    for (int m = 0; m < actual_merges; m++) {
        printf("    {%d, %d},", merge_left[m], merge_right[m]);
        /* add comment with string representation */
        printf("  /* %d: \"%s\" + \"%s\" -> \"%s\" */\n",
               BASE_VOCAB + m, tok_str[merge_left[m]], tok_str[merge_right[m]],
               tok_str[BASE_VOCAB + m]);
    }
    printf("};\n");

    /* also output token strings for debugging */
    printf("\n/* Token string representations (for debugging) */\n");
    printf("/* static const char *BPE_STRS[BPE_VOCAB] = { ... }; */\n");

    free(pair_count);
    free(tokens);
    free(tok_next);
    free(tok_prev);

    return 0;
}
