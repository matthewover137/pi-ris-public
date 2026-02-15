// llama2.c — bare-metal quantized inference engine
// Ported from karpathy/llama2.c runq.c (int8 quantized forward pass)
// Changes: no filesystem, no OS, kmalloc instead of malloc, printk instead of printf
#include "rpi.h"
#include "memmap.h"
#include <math.h>
#include "llama2.h"

// ----------------------------------------------------------------------------
// stubs needed by libm

static int errno_val;
int *__errno(void) { return &errno_val; }

// ----------------------------------------------------------------------------
// Globals

static int GS = 0; // group size for quantization

// ----------------------------------------------------------------------------
// stdlib replacements for bare metal

static int my_abs(int x) { return x < 0 ? -x : x; }
static int my_isprint(int c) { return c >= 32 && c <= 126; }
static int my_isspace(int c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// simple qsort (shell sort — good enough for our sizes)
static void my_qsort(void *base, unsigned nel, unsigned width,
                      int (*cmp)(const void *, const void *)) {
    uint8_t *arr = base;
    uint8_t tmp[64]; // max element size we support
    if (width > sizeof(tmp)) panic("qsort element too large\n");

    for (unsigned gap = nel / 2; gap > 0; gap /= 2) {
        for (unsigned i = gap; i < nel; i++) {
            memcpy(tmp, arr + i * width, width);
            unsigned j = i;
            while (j >= gap && cmp(arr + (j - gap) * width, tmp) > 0) {
                memcpy(arr + j * width, arr + (j - gap) * width, width);
                j -= gap;
            }
            memcpy(arr + j * width, tmp, width);
        }
    }
}

// simple bsearch
static void *my_bsearch(const void *key, const void *base, unsigned nel,
                         unsigned width, int (*cmp)(const void *, const void *)) {
    const uint8_t *arr = base;
    unsigned lo = 0, hi = nel;
    while (lo < hi) {
        unsigned mid = lo + (hi - lo) / 2;
        int r = cmp(key, arr + mid * width);
        if (r == 0) return (void *)(arr + mid * width);
        if (r < 0) hi = mid;
        else lo = mid + 1;
    }
    return 0;
}

// parse "<0xHH>" hex byte tokens — replaces sscanf(piece, "<0x%02hhX>", &byte_val)
static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static int parse_hex_byte(const char *s, unsigned char *out) {
    if (s[0] != '<' || s[1] != '0' || s[2] != 'x') return 0;
    int hi = hex_val(s[3]), lo = hex_val(s[4]);
    if (hi < 0 || lo < 0 || s[5] != '>') return 0;
    *out = (hi << 4) | lo;
    return 1;
}

// timer
static unsigned long time_in_ms(void) {
    return timer_get_usec() / 1000;
}

// fabsf replacement (avoid potential libm issues on bare metal)
static float my_fabsf(float x) { return x < 0.0f ? -x : x; }

// ----------------------------------------------------------------------------
// Quantization functions

static void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

static void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {
        // find the max absolute value in the current group
        float wmax = 0.0f;
        for (int i = 0; i < GS; i++) {
            float val = my_fabsf(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale;
            int8_t quantized = (int8_t)(quant_value + (quant_value >= 0 ? 0.5f : -0.5f));
            qx->q[group * GS + i] = quantized;
        }
    }
}

// Initialize n quantized tensors (with size_each elements each), reading from *ptr
static QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = kmalloc(n * sizeof(QuantizedTensor));
    for (int i = 0; i < n; i++) {
        // map quantized int8 values
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        // map scale factors
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

// ----------------------------------------------------------------------------
// Transformer model

static void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    // first are the parameters kept in fp32 (the rmsnorm 1D weights)
    float* fptr = (float*) ptr;
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    // now read all the quantized weights
    ptr = (void*)fptr;
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // dequantize token embedding table (needed for lookup by index)
    w->token_embedding_table = kmalloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

static void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = kmalloc(p->dim * sizeof(float));
    s->xb = kmalloc(p->dim * sizeof(float));
    s->xb2 = kmalloc(p->dim * sizeof(float));
    s->hb = kmalloc(p->hidden_dim * sizeof(float));
    s->hb2 = kmalloc(p->hidden_dim * sizeof(float));
    s->xq.q = kmalloc(p->dim * sizeof(int8_t));
    s->xq.s = kmalloc(p->dim * sizeof(float));
    s->hq.q = kmalloc(p->hidden_dim * sizeof(int8_t));
    s->hq.s = kmalloc(p->hidden_dim * sizeof(float));
    s->q = kmalloc(p->dim * sizeof(float));
    s->k = kmalloc(kv_dim * sizeof(float));
    s->v = kmalloc(kv_dim * sizeof(float));
    s->att = kmalloc(p->n_heads * p->seq_len * sizeof(float));
    s->logits = kmalloc(p->vocab_size * sizeof(float));
    s->key_cache = kmalloc(p->n_layers * p->seq_len * kv_dim * sizeof(float));
    s->value_cache = kmalloc(p->n_layers * p->seq_len * kv_dim * sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits
     || !s->key_cache || !s->value_cache) {
        panic("kmalloc failed allocating RunState!\n");
    }
}

void build_transformer(Transformer *t, const uint8_t *model_data) {
    // quantized format: 256-byte header
    // [0..3]  magic number 0x616b3432 ("ak42")
    // [4..7]  version (must be 2)
    // [8..35] Config struct (7 ints)
    // [36]    shared_classifier (uint8_t)
    // [37..40] group_size (int)
    // [41..255] padding
    // [256+]  weights data

    const uint8_t *ptr = model_data;

    uint32_t magic;
    memcpy(&magic, ptr, sizeof(uint32_t));
    if (magic != 0x616b3432) {
        panic("bad magic number — is this a quantized model? expected 0x616b3432\n");
    }
    ptr += sizeof(uint32_t);

    int version;
    memcpy(&version, ptr, sizeof(int));
    if (version != 2) {
        panic("bad version %d, need version 2\n");
    }
    ptr += sizeof(int);

    // read config
    memcpy(&t->config, ptr, sizeof(Config));
    ptr += sizeof(Config);

    // read flags
    uint8_t shared_classifier;
    memcpy(&shared_classifier, ptr, sizeof(uint8_t));
    ptr += sizeof(uint8_t);

    int group_size;
    memcpy(&group_size, ptr, sizeof(int));
    GS = group_size;

    // weights start at byte 256
    void *weights_ptr = (void *)(model_data + 256);
    memory_map_weights(&t->weights, &t->config, weights_ptr, shared_classifier);

    // allocate RunState buffers
    malloc_run_state(&t->state, &t->config);
}

// ----------------------------------------------------------------------------
// neural net blocks — quantized forward pass

static void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

static void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Quantized matmul: W (d,n) @ x (n,) -> xout (d,)
// Both inputs are quantized; accumulate in int32, scale by group
static void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }

        xout[i] = val;
    }
}

static float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        // RoPE relative positional encoding
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step to our kv cache
        int loff = l * p->seq_len * kv_dim;
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(float));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(float));

        // multihead attention
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);

            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get attention output
        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // FFN: w2(SwiGLU(w1(x), w3(x)))
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // SwiGLU
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get ffn output
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// Tokenizer

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, const uint8_t *data, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)kmalloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)kmalloc(vocab_size * sizeof(float));
    t->sorted_vocab = 0;

    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    // parse tokenizer.bin from baked-in data
    const uint8_t *ptr = data;

    // first 4 bytes: max_token_length
    memcpy(&t->max_token_length, ptr, sizeof(int));
    ptr += sizeof(int);

    // for each token: float score, int len, then len bytes of string
    for (int i = 0; i < vocab_size; i++) {
        float score;
        memcpy(&score, ptr, sizeof(float));
        t->vocab_scores[i] = score;
        ptr += sizeof(float);

        int len;
        memcpy(&len, ptr, sizeof(int));
        ptr += sizeof(int);

        t->vocab[i] = (char *)kmalloc(len + 1);
        memcpy(t->vocab[i], ptr, len);
        t->vocab[i][len] = '\0';
        ptr += len;
    }
}

static char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (parse_hex_byte(piece, &byte_val)) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

static void safe_printk(char *piece) {
    if (piece == 0) return;
    if (piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(my_isprint(byte_val) || my_isspace(byte_val))) {
            return;
        }
    }
    printk("%s", piece);
}

static int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok;
    tok.str = str;
    tok.id = 0;
    TokenIndex *res = my_bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != 0 ? res->id : -1;
}

static void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == 0) { panic("cannot encode NULL text\n"); }

    if (t->sorted_vocab == 0) {
        t->sorted_vocab = kmalloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        my_qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    unsigned bufsize = t->max_token_length * 2 + 1 + 2;
    char* str_buffer = kmalloc(bufsize);
    unsigned str_len = 0;

    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (unsigned i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    // merge the best consecutive pair each iteration
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            // manual strcpy+strcat instead of sprintf
            strcpy(str_buffer, t->vocab[tokens[i]]);
            strcat(str_buffer, t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) break;

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;
}

// ----------------------------------------------------------------------------
// Sampler

static int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

static int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

static int compare_probindex(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*)a;
    ProbIndex* b_ = (ProbIndex*)b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

static int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    my_qsort(probindex, n0, sizeof(ProbIndex), compare_probindex);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) return probindex[i].index;
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = kmalloc(vocab_size * sizeof(ProbIndex));
}

static unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

static float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// Generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == 0) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)kmalloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        panic("expected at least 1 prompt token\n");
    }

    unsigned long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 1) break; // BOS = end of sequence

        char* piece = decode(tokenizer, token, next);
        safe_printk(piece);
        token = next;

        if (start == 0) { start = time_in_ms(); }
    }
    printk("\n");

    if (pos > 1) {
        unsigned long end = time_in_ms();
        unsigned long elapsed = end - start;
        if (elapsed > 0) {
            printk("[%d tokens, %d ms, %d tok/s]\n",
                   pos - 1, (int)elapsed, (int)((pos - 1) * 1000 / elapsed));
        }
    }
}

// ----------------------------------------------------------------------------
// Chat-aware generation with I:/M: message boundary detection
//
// After the prompt tokens, the model generates continuation text.
// We parse character-by-character looking for "M: " or "I: " at the
// start of each new line. M: lines become messages sent to the host.
// I: or EOS means we're done. No M: at all means NOREPLY.

// Send a protocol line: \x04<text>\n
static void proto_send(const char *line) {
    uart_put8(SOH);
    for (const char *p = line; *p; p++)
        uart_put8(*p);
    uart_put8('\n');
}

// Send a MSG protocol line: \x04MSG:<text>\n (safe — no printk format strings)
static void proto_send_msg(const char *text, int len) {
    uart_put8(SOH);
    uart_put8('M'); uart_put8('S'); uart_put8('G'); uart_put8(':');
    for (int i = 0; i < len; i++)
        uart_put8(text[i]);
    uart_put8('\n');
}

// --- KV cache reuse state (persistent between calls) ---
// Saves the token sequence from the last generation so we can skip
// re-processing the common prefix on the next call.
static int cached_seq[256];    // token IDs at each position (matches seq_len)
static int cached_len = 0;     // number of valid cached positions

int chat_generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                  char *prompt, int steps) {
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)kmalloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        return CHAT_NOREPLY;
    }

    // --- KV cache reuse: find common prefix with cached sequence ---
    int common = 0;
    while (common < cached_len && common < num_prompt_tokens &&
           cached_seq[common] == prompt_tokens[common]) {
        common++;
    }
    // Always re-run at least the last prompt token to get logits for sampling
    int start_pos = common;
    if (start_pos > num_prompt_tokens - 1)
        start_pos = num_prompt_tokens - 1;
    if (start_pos < 0) start_pos = 0;

    printk("GEN: %d prompt, %d cached, skip %d\n",
           num_prompt_tokens, cached_len, start_pos);

    // Track tokens at each position for saving later
    int seq_buf[256];
    // Copy cached prefix (positions 0..start_pos-1 are already in KV cache)
    for (int i = 0; i < start_pos && i < 256; i++)
        seq_buf[i] = cached_seq[i];

    // State machine for message boundary detection
    enum { SKIP_LINE, LINE_START, IN_PREFIX, IN_M_MSG } state = SKIP_LINE;
    char msg_buf[512];
    int msg_len = 0;
    char pfx[4];
    int pfx_len = 0;
    int sent_any = 0;
    int last_msg_end_pos = num_prompt_tokens; // save point after last M: message

    // Debug buffer for NOREPLY diagnostics
    char dbg_buf[128];
    int dbg_len = 0;

    int token = prompt_tokens[start_pos];
    int pos = start_pos;
    int next;

    while (pos < steps) {
        // Check for cancel
        if (uart_has_data()) {
            int c = uart_get8();
            if (c == CAN) {
                proto_send("CANCEL_ACK");
                // Save prompt tokens only (discard generation)
                int save = num_prompt_tokens < 256 ? num_prompt_tokens : 256;
                for (int i = start_pos; i < save; i++) seq_buf[i] = prompt_tokens[i];
                memcpy(cached_seq, seq_buf, save * sizeof(int));
                cached_len = save;
                return CHAT_CANCELLED;
            }
        }

        // Record token at this position
        if (pos < 256)
            seq_buf[pos] = token;

        unsigned t0 = timer_get_usec();
        float *logits = forward(transformer, token, pos);
        unsigned t1 = timer_get_usec();

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
            // Show every prompt forward pass
            printk("  fwd[%d/%d] %dms\n", pos - start_pos + 1,
                   num_prompt_tokens - start_pos, (t1 - t0) / 1000);
        } else {
            if (pos == num_prompt_tokens - 1)
                printk("  prompt done, fwd=%dms, sampling\n", (t1 - t0) / 1000);
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 1) break; // EOS

        // Parse generated text AFTER the prompt
        if (pos >= num_prompt_tokens) {
            // Show first few generated tokens with timing
            if (pos - num_prompt_tokens < 5)
                printk("  gen[%d] %dms\n", pos - num_prompt_tokens, (t1 - t0) / 1000);
            char *piece = decode(tokenizer, token, next);

            for (int ci = 0; piece[ci] != '\0'; ci++) {
                char c = piece[ci];

                // Debug buffer
                if (dbg_len < (int)sizeof(dbg_buf) - 1)
                    dbg_buf[dbg_len++] = c;

                switch (state) {
                case SKIP_LINE:
                    if (c == '\n') state = LINE_START;
                    break;

                case LINE_START:
                    if (c == '\n' || c == '\r' || c == ' ') break;
                    pfx[0] = c;
                    pfx_len = 1;
                    state = IN_PREFIX;
                    break;

                case IN_PREFIX:
                    if (c == '\n') { state = LINE_START; break; }
                    pfx[pfx_len++] = c;
                    if (pfx_len == 3) {
                        if (pfx[0] == 'M' && pfx[1] == ':' && pfx[2] == ' ') {
                            state = IN_M_MSG;
                            msg_len = 0;
                            proto_send("TYPING");
                        } else if (pfx[0] == 'I' && pfx[1] == ':' && pfx[2] == ' ') {
                            goto done;
                        } else {
                            state = SKIP_LINE;
                        }
                    }
                    break;

                case IN_M_MSG:
                    if (c == '\n') {
                        msg_buf[msg_len] = '\0';
                        proto_send_msg(msg_buf, msg_len);
                        sent_any = 1;
                        last_msg_end_pos = pos; // checkpoint: save up to here
                        state = LINE_START;
                    } else {
                        if (msg_len < (int)sizeof(msg_buf) - 1)
                            msg_buf[msg_len++] = c;
                    }
                    break;
                }
            }
        }

        token = next;
    }

done:
    // Flush mid-message if generation ended while buffering
    if (state == IN_M_MSG && msg_len > 0) {
        msg_buf[msg_len] = '\0';
        proto_send_msg(msg_buf, msg_len);
        sent_any = 1;
        last_msg_end_pos = pos;
    }

    // Send debug info on NOREPLY so host can see what the model generated
    if (!sent_any && dbg_len > 0) {
        dbg_buf[dbg_len] = '\0';
        uart_put8(SOH);
        uart_put8('D'); uart_put8('B'); uart_put8('G'); uart_put8(':');
        for (int i = 0; i < dbg_len; i++) {
            if (dbg_buf[i] == '\n') { uart_put8('\\'); uart_put8('n'); }
            else if (dbg_buf[i] == '\r') { uart_put8('\\'); uart_put8('r'); }
            else uart_put8(dbg_buf[i]);
        }
        uart_put8('\n');
    }

    // --- Save valid tokens for KV cache reuse ---
    // COMPLETE: save prompt + M: messages (up to last_msg_end_pos)
    // NOREPLY:  save prompt only (discard failed generation)
    int save_len = sent_any ? last_msg_end_pos : num_prompt_tokens;
    if (save_len > 256) save_len = 256;
    // Fill in any prompt tokens not yet in seq_buf (from start_pos onward)
    for (int i = start_pos; i < save_len && i < num_prompt_tokens; i++)
        seq_buf[i] = prompt_tokens[i];
    memcpy(cached_seq, seq_buf, save_len * sizeof(int));
    cached_len = save_len;

    printk("GEN: result=%s, saved %d tokens\n",
           sent_any ? "COMPLETE" : "NOREPLY", cached_len);

    return sent_any ? CHAT_COMPLETE : CHAT_NOREPLY;
}
