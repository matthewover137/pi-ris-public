// llama2.h — bare-metal quantized inference engine API
// Ported from karpathy/llama2.c runq.c for Raspberry Pi bare metal.
#ifndef __LLAMA2_H__
#define __LLAMA2_H__

#include "rpi.h"

// ----------------------------------------------------------------------------
// Structs (matches runq.c quantized format)

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    int8_t* q;    // quantized values
    float* s;     // scaling factors
} QuantizedTensor;

typedef struct {
    QuantizedTensor *q_tokens; // (vocab_size, dim)
    float* token_embedding_table; // dequantized copy
    float* rms_att_weight; // (layer, dim)
    float* rms_ffn_weight; // (layer, dim)
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    float* rms_final_weight; // (dim,)
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    float *x;      // activation at current time stamp (dim,)
    float *xb;     // inside a residual branch (dim,)
    float *xb2;    // convenience buffer (dim,)
    float *hb;     // hidden dimension ffn (hidden_dim,)
    float *hb2;    // hidden dimension ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q;      // query (dim,)
    float *k;      // key (dim,)
    float *v;      // value (dim,)
    float *att;    // attention scores (n_heads, seq_len)
    float *logits; // output logits
    float* key_cache;   // (layer, seq_len, kv_dim)
    float* value_cache; // (layer, seq_len, kv_dim)
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
} Transformer;

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

// ----------------------------------------------------------------------------
// Chat protocol constants

#define STX 0x02  // host→pi: start of prompt frame
#define ETX 0x03  // host→pi: end of prompt frame
#define SOH 0x04  // pi→host: start of protocol line
#define ACK 0x06  // host→pi: acknowledge READY
#define CAN 0x18  // host→pi: cancel current generation

// Chat generation return codes
#define CHAT_COMPLETE   0  // generated M: messages, finished normally
#define CHAT_NOREPLY    1  // model didn't produce M: response
#define CHAT_CANCELLED  2  // cancelled by new data on UART

// ----------------------------------------------------------------------------
// Public API

// Build from baked-in data pointers (no filesystem)
void build_transformer(Transformer *t, const uint8_t *model_data);
void build_tokenizer(Tokenizer *t, const uint8_t *tokenizer_data, int vocab_size);
void build_sampler(Sampler *s, int vocab_size, float temperature, float topp, unsigned long long rng_seed);

// Simple generation (prints tokens directly — for testing)
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps);

// Chat-aware generation with I:/M: message boundary detection.
// Sends \x04TYPING\n and \x04MSG:<text>\n over UART.
// Returns CHAT_COMPLETE, CHAT_NOREPLY, or CHAT_CANCELLED.
int chat_generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                  char *prompt, int steps);

#endif
