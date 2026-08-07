// Runtime for instrument-agnostic-amt (Transkun-derived) GGUF models.
//
// Pipeline: audio -> HCQT (CPU) -> ggml graph (conv stem, dual-axis
// transformer, semi-CRF interval scorer) -> Viterbi decode (CPU) -> MIDI.
//
// See docs/iaamt-conversion.md for the GGUF layout this consumes.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

#define IAAMT_MODEL_TYPE_TRANSCRIPTION "transcription"
#define IAAMT_MODEL_TYPE_VELOCITY      "velocity"
#define IAAMT_MODEL_TYPE_BEAT_CHORD    "beat_chord"

//
// hyper-parameters, all read from GGUF KV
//

struct iaamt_cqt_stage {
    int n_fft    = 0;
    int bin_beg  = 0;   // first CQT bin this stage produces
    int bin_end  = 0;   // one past the last
};

struct iaamt_hparams {
    std::string model_type;

    int   sample_rate         = 22050;
    int   hop_length          = 512;
    int   input_audio_channels = 2;
    int   num_audio_channels  = 10;  // input_audio_channels * n_harmonics

    // HCQT
    float cqt_fmin            = 27.5f;
    float cqt_fmin_large      = 27.5f;
    int   cqt_n_bins          = 312;
    int   cqt_actual_bins     = 312;
    int   cqt_n_bins_large    = 396;
    int   cqt_bins_per_octave = 36;
    float cqt_filter_scale    = 0.475f;
    bool  cqt_log_scale       = false;
    std::vector<float> harmonics;
    std::vector<float> harmonic_shifts;
    std::vector<iaamt_cqt_stage> cqt_stages;
    bool  cqt_kernels_embedded = true;

    // backbone
    int   hidden_size         = 384;
    int   base_ch             = 64;
    int   n_layer             = 6;
    int   n_head              = 12;
    int   pitch_query_count   = 88;

    // beat / chord / key
    int   num_input_channels  = 72;   // 36 instrument classes x {sustain, onset}
    int   num_global_tokens   = 4;
    int   num_meter_classes   = 60;
    int   num_root_chord_classes = 745;
    int   time_downsample     = 8;
    int   ffn_multiplier      = 4;
    std::vector<int> inter_refine_layers;

    // velocity
    int   pitch_min           = 21;
    int   pitch_max           = 108;
    int   num_stem_classes    = 7;
    int   note_hidden_size    = 256;
    bool  absolute_velocity_energy = true;
    std::vector<float> local_frame_offsets;

    // head
    int   num_pitch_slots     = 1;
    int   semi_crf_head_dim   = 256;
    float semi_crf_length_penalty = 0.0f;
    std::string semi_crf_length_scaling = "none";
    int   num_instrument_classes = 0;
    bool  has_slot_embedding  = false;
    bool  has_interval_instrument_head = false;
    bool  use_interval_boundary_head   = true;

    // derived
    int token_dim() const { return base_ch * 4; }
    int head_dim()  const { return hidden_size / n_head; }
    int n_bands()   const;   // frequency bands surviving the conv stem
    int n_tokens()  const { return n_bands() + pitch_query_count; }
};

//
// weights
//

struct iaamt_attn {
    ggml_tensor * norm_q     = nullptr;
    ggml_tensor * norm_kv    = nullptr;
    ggml_tensor * q_w        = nullptr;
    ggml_tensor * q_b        = nullptr;
    ggml_tensor * k_w        = nullptr;
    ggml_tensor * k_b        = nullptr;
    ggml_tensor * v_w        = nullptr;
    ggml_tensor * v_b        = nullptr;
    ggml_tensor * gate_w     = nullptr;
    ggml_tensor * gate_b     = nullptr;
    ggml_tensor * out_w      = nullptr;
    ggml_tensor * out_b      = nullptr;

    ggml_tensor * ffn_norm   = nullptr;
    ggml_tensor * ffn_up_w   = nullptr;
    ggml_tensor * ffn_up_b   = nullptr;
    ggml_tensor * ffn_down_w = nullptr;
    ggml_tensor * ffn_down_b = nullptr;
};

// one dual-axis layer: a band-axis transformer and a time-axis transformer
struct iaamt_layer {
    iaamt_attn time;
    iaamt_attn band;
};

struct iaamt_stem_block {
    ggml_tensor * conv_w = nullptr;
    ggml_tensor * conv_b = nullptr;
    ggml_tensor * norm_w = nullptr;
    ggml_tensor * norm_b = nullptr;
};

// LayerNorm + Linear + GELU + Linear, wrapped in a residual
struct iaamt_adapter {
    ggml_tensor * norm_w = nullptr;
    ggml_tensor * norm_b = nullptr;
    ggml_tensor * fc1_w  = nullptr;
    ggml_tensor * fc1_b  = nullptr;
    ggml_tensor * fc2_w  = nullptr;
    ggml_tensor * fc2_b  = nullptr;
};

// beat / chord / key output projections, all fed by one shared LayerNorm+Linear
struct iaamt_bc_head {
    ggml_tensor * norm_w = nullptr;
    ggml_tensor * norm_b = nullptr;
    ggml_tensor * fc_w   = nullptr;
    ggml_tensor * fc_b   = nullptr;
    // beat: frame (beat, downbeat, meter) + group_boundary
    // chord: boundary, root_chord, bass, key_boundary, key, pitch
    ggml_tensor * proj_w[6] = { nullptr };
    ggml_tensor * proj_b[6] = { nullptr };
    int n_proj = 0;
};

// one intermediate refinement stage, applied after a given backbone layer
struct iaamt_bc_refine {
    int layer = -1;
    ggml_tensor * up_conv_w  = nullptr;
    ggml_tensor * up_conv_b  = nullptr;
    ggml_tensor * merge_w    = nullptr;
    ggml_tensor * merge_b    = nullptr;
    iaamt_adapter beat_adapter;
    iaamt_adapter chord_adapter;
    iaamt_bc_head beat_head;
    iaamt_bc_head chord_head;
    ggml_tensor * beat_down_w   = nullptr;
    ggml_tensor * beat_down_b   = nullptr;
    ggml_tensor * chord_down_w  = nullptr;
    ggml_tensor * chord_down_b  = nullptr;
    ggml_tensor * beat_fb_w     = nullptr;
    ggml_tensor * beat_fb_b     = nullptr;
    ggml_tensor * chord_fb_w    = nullptr;
    ggml_tensor * chord_fb_b    = nullptr;
    ggml_tensor * beat_gate     = nullptr;
    ggml_tensor * chord_gate    = nullptr;
};

struct iaamt_model {
    iaamt_hparams hparams;

    // stem
    ggml_tensor * stem_conv1_w = nullptr;
    ggml_tensor * stem_conv1_b = nullptr;
    ggml_tensor * stem_conv2_w = nullptr;
    ggml_tensor * stem_conv2_b = nullptr;
    ggml_tensor * stem_freq_embd = nullptr;
    iaamt_stem_block stem_blocks[4];

    // tokens
    ggml_tensor * band_type_embd  = nullptr;
    ggml_tensor * pitch_type_embd = nullptr;
    ggml_tensor * pitch_query_fc1_w = nullptr;
    ggml_tensor * pitch_query_fc1_b = nullptr;
    ggml_tensor * pitch_query_fc2_w = nullptr;
    ggml_tensor * pitch_query_fc2_b = nullptr;

    std::vector<iaamt_layer> layers;

    ggml_tensor * output_norm = nullptr;
    ggml_tensor * up_conv_w   = nullptr;
    ggml_tensor * up_conv_b   = nullptr;

    // beat / chord / key
    ggml_tensor * bc_conv1_w   = nullptr;   // depthwise
    ggml_tensor * bc_conv1_b   = nullptr;
    ggml_tensor * bc_conv2_w   = nullptr;   // depthwise
    ggml_tensor * bc_conv2_b   = nullptr;
    ggml_tensor * bc_chan_w    = nullptr;   // 1x1 pointwise
    ggml_tensor * bc_chan_b    = nullptr;
    ggml_tensor * bc_pitch_embd = nullptr;
    ggml_tensor * bc_input_proj_w = nullptr;
    ggml_tensor * bc_input_proj_b = nullptr;
    ggml_tensor * bc_pitch_pos_embd = nullptr;
    ggml_tensor * bc_global_tokens  = nullptr;
    ggml_tensor * bc_global_type    = nullptr;
    ggml_tensor * bc_global_up_w    = nullptr;
    ggml_tensor * bc_global_up_b    = nullptr;
    ggml_tensor * bc_merge_w        = nullptr;
    ggml_tensor * bc_merge_b        = nullptr;
    iaamt_adapter bc_beat_adapter;
    iaamt_adapter bc_chord_adapter;
    iaamt_bc_head bc_beat_head;
    iaamt_bc_head bc_chord_head;
    std::vector<iaamt_bc_refine> bc_refine;

    // transcription head
    iaamt_adapter interval_adapter;
    iaamt_adapter instrument_adapter;
    ggml_tensor * slot_embd        = nullptr;
    ggml_tensor * interval_scorer_w = nullptr;
    ggml_tensor * interval_scorer_b = nullptr;
    ggml_tensor * boundary_fc1_w   = nullptr;
    ggml_tensor * boundary_fc1_b   = nullptr;
    ggml_tensor * boundary_fc2_w   = nullptr;
    ggml_tensor * boundary_fc2_b   = nullptr;
    ggml_tensor * instrument_cls_w = nullptr;
    ggml_tensor * instrument_cls_b = nullptr;

    // The boundary head runs per decoded interval on the CPU, so its weights are
    // mirrored to host memory -- the tensors above may live on an accelerator.
    std::vector<float> boundary_w1, boundary_b1, boundary_w2, boundary_b2;

    // The velocity head is a handful of small per-note MLPs; it runs on the CPU
    // for the same reason, so its weights are mirrored too.
    struct velocity_weights {
        std::vector<float> pitch_embd, program_embd, drum_embd, stem_embd;
        std::vector<float> dur_w1, dur_b1, dur_w2, dur_b2;
        std::vector<float> nq_w1, nq_b1, nq_w2, nq_b2;
        std::vector<float> local_w, local_b;
        std::vector<float> att_w1, att_b1, att_w2, att_b2;
        std::vector<float> fuse_norm_w, fuse_norm_b, fuse_w, fuse_b;
        std::vector<float> vel_w, vel_b;
    } vel;

    // CQT tables emitted by the converter (kept on the CPU, never in the graph)
    std::vector<std::vector<float>> cqt_kernel_re;   // [stage][n_freqs * n_rfft]
    std::vector<std::vector<float>> cqt_kernel_im;
    std::vector<std::vector<float>> cqt_window;      // [stage][n_fft]
    std::vector<std::vector<float>> cqt_resampler;   // [stage][taps]

    // backend state.  `backend` holds the weights and runs most of the graph;
    // `backend_cpu` is always present so the scheduler can fall back for ops the
    // accelerator does not implement.
    ggml_backend_t            backend     = nullptr;
    ggml_backend_t            backend_cpu = nullptr;
    ggml_backend_buffer_t     buf_weights = nullptr;
    ggml_context            * ctx_data    = nullptr;

    ~iaamt_model();
};

//
// model loading (iaamt-model.cpp)
//

// Loads a GGUF produced by convert_iaamt_to_gguf.py. Returns false and fills
// `err` on failure. `n_gpu` < 0 selects the best available backend.
bool iaamt_model_load(iaamt_model & model, const std::string & fname, bool use_gpu, std::string & err);

//
// HCQT front end (iaamt-cqt.cpp)
//

// Computes the network input for one window of stereo audio.
//
// `pcm` is planar: pcm[ch] holds `n_samples` floats.
// `out` is filled with [F=cqt_n_bins, T, num_audio_channels] in ggml order
// (frequency fastest), which is exactly the layout the graph expects.
// Returns the number of frames T.
int iaamt_cqt_apply(const iaamt_model & model,
                    const std::vector<std::vector<float>> & pcm,
                    int n_samples,
                    std::vector<float> & out);

//
// graph evaluation (iaamt-graph.cpp)
//

// Outputs of one forward pass over a single window.
struct iaamt_window_out {
    int n_frames = 0;   // T, at the audio frame rate (hop_length)
    int n_tracks = 0;   // pitch_query_count * num_pitch_slots

    // interval scorer, laid out [track][frame][dim]
    int head_dim = 0;
    std::vector<float> query;   // n_tracks * n_frames * head_dim, pre-scaled
    std::vector<float> key;     // n_tracks * n_frames * head_dim
    std::vector<float> diag;    // n_tracks * n_frames

    // adapter features, needed by the boundary head; [track][frame][token_dim]
    int feat_dim = 0;
    std::vector<float> interval_features;

    // velocity models stop before the head: raw pitch features [pitch][frame][dim]
    std::vector<float> pitch_features;

    // populated only when iaamt_ctx_set_debug() is on: named graph intermediates
    // in ne order, for stage-by-stage comparison against the reference model
    struct debug_tensor {
        std::string          name;
        std::vector<int64_t> ne;
        std::vector<float>   data;
    };
    std::vector<debug_tensor> debug;
};

struct iaamt_context;

iaamt_context * iaamt_ctx_init(iaamt_model & model, int n_threads);
void            iaamt_ctx_free(iaamt_context * ctx);

// When enabled, iaamt_forward also returns every named intermediate.
void            iaamt_ctx_set_debug(iaamt_context * ctx, bool enable);

// Runs stem -> transformer -> head for one window's HCQT features.
bool iaamt_forward(iaamt_context * ctx,
                   const std::vector<float> & feats,
                   int n_frames_in,
                   int crop_length,
                   iaamt_window_out & out,
                   std::string & err);

//
// decoding (iaamt-decode.cpp)
//

struct iaamt_note {
    int   pitch        = 0;
    int   slot         = 0;
    int64_t start_sample = 0;
    int64_t end_sample   = 0;
    int   velocity     = 100;
    bool  has_onset    = true;
    bool  has_offset   = true;
    int   program      = 0;      // GM program, carried through from input MIDI
    bool  is_drum      = false;

    // Semi-CRF score of the interval this note came from: unbounded, positive,
    // and larger for intervals the model was more willing to pay for. Not a
    // probability and deliberately not squashed into one here - the mapping to a
    // confidence depends on what the consumer is comparing against, and doing it
    // at the source would bake in a choice this decoder has no basis for.
    // Zero when the note did not come from a Viterbi interval.
    //
    // Two things a consumer must know before comparing these. `build_score`
    // applies `semi_crf_length_scaling` and `semi_crf_length_penalty`, so the
    // magnitude depends on interval length as well as on the model's
    // confidence: a long note and a short note with the same score were not
    // believed equally. And the scale is per-model and per-stem, so a threshold
    // learned on one checkpoint does not transfer to another. Ranking notes
    // within a single transcription is what this supports; an absolute cutoff
    // is not, without a calibration that does not exist yet.
    float crf_score    = 0.0f;

    // Boundary-head probabilities for this note's onset and offset, in 0..1.
    // These come from a different head than `crf_score` and answer a different
    // question - whether this edge is a real note boundary, rather than whether
    // the interval is a real note - so they are genuine separate measurements
    // and not restatements of the score. Being trained binary classifiers they
    // are calibrated, which `crf_score` is not, so a fixed threshold is
    // meaningful here in a way it is not there. `has_onset` and `has_offset`
    // are exactly these thresholded at 0.5.
    //
    // **Negative means the head did not run**, because the model carries no
    // boundary weights or the caller disabled it. That is "unknown", not
    // "improbable": the flags then reflect where the interval sat in its window
    // rather than anything the model asserted, and reading the absence as a low
    // confidence would turn window layout into evidence it is not.
    float onset_confidence  = -1.0f;
    float offset_confidence = -1.0f;
};

struct iaamt_decode_params {
    float note_bias        = 0.0f;
    float merge_onset_ms   = 50.0f;
    float merge_gap_ms     = -1.0f;   // < 0 falls back to one hop, as the reference does
    int   velocity         = 100;
    float min_note_ms      = 5.0f;
    bool  use_boundary_head = true;
};

// Accumulates decoded intervals across overlapping windows.  A sustained note
// spans several windows, so each track remembers where it was last closed;
// that position seeds the next window's Viterbi backtrace.
struct iaamt_stitcher {
    int     hop_length    = 512;
    int     sample_rate   = 22050;
    int64_t total_samples = 0;
    int     n_slots       = 1;
    int     velocity      = 100;
    int64_t merge_gap_samples   = 0;
    int64_t merge_onset_samples = 0;

    std::vector<int64_t>                 last_closed;   // per track, in model frames
    std::vector<std::vector<iaamt_note>> by_track;
};

// _build_window_starts: analysis windows advance by `stride`, and a final one
// is placed at `total - window` so the tail is covered by a full window rather
// than a short one. Shared rather than duplicated because getting this wrong
// changes which notes are found, and silently: a second copy that merely looks
// right would drift from this one without failing anything.
std::vector<int64_t> iaamt_build_window_starts(int64_t total, int64_t window, int64_t stride);

void iaamt_stitcher_init(iaamt_stitcher & st,
                         const iaamt_model & model,
                         const iaamt_decode_params & params,
                         int64_t total_samples);

// Viterbi-decodes one window and folds its notes into the stitcher.
// Returns the number of intervals decoded.
int iaamt_stitcher_consume(iaamt_stitcher & st,
                           const iaamt_model & model,
                           const iaamt_window_out & out,
                           const iaamt_decode_params & params,
                           int valid_model_frames,
                           int valid_audio_samples,
                           int64_t window_start_sample);

std::vector<iaamt_note> iaamt_stitcher_finalize(iaamt_stitcher & st,
                                                const iaamt_decode_params & params);

//
// velocity prediction (iaamt-velocity.cpp)
//

// Predicts MIDI velocity for every note whose onset falls inside this window.
// `pcm` is the same planar window handed to the CQT, needed for the log-energy
// channels the backbone normalizes away.  Notes are updated in place.
void iaamt_velocity_apply(const iaamt_model & model,
                          const iaamt_window_out & out,
                          const std::vector<std::vector<float>> & pcm,
                          int n_window_samples,
                          int valid_model_frames,
                          int64_t window_start_sample,
                          int64_t assign_end_sample,
                          std::vector<iaamt_note> & notes);

//
// beat / chord / key (iaamt-beatchord.cpp)
//

// Frame-level outputs for one window, at the input frame rate.
struct iaamt_bc_out {
    int n_frames = 0;
    std::vector<float> beat;            // sigmoid probability
    std::vector<float> downbeat;        // sigmoid probability
    std::vector<int>   meter;           // argmax meter class per frame
    std::vector<float> chord_boundary;  // sigmoid probability
    std::vector<int>   root_chord;      // argmax over num_root_chord_classes
    std::vector<int>   bass;            // argmax over 13 (12 pitch classes + none)
    std::vector<int>   key;             // argmax over 13
};

// Rasterizes notes into the [72, T, 88] roll the model consumes: one sustain
// channel and one onset channel per instrument class.
void iaamt_bc_build_roll(const iaamt_model & model,
                         const std::vector<iaamt_note> & notes,
                         int64_t window_start_sample,
                         int n_frames,
                         std::vector<float> & roll);

// Runs the MIDI-roll backbone and heads for one window.
bool iaamt_bc_forward(iaamt_context * ctx,
                      const std::vector<float> & roll,
                      int n_frames,
                      iaamt_bc_out & out,
                      std::string & err);

// Writes beats, chords and key as a plain-text analysis alongside the MIDI.
bool iaamt_bc_write(const std::string & fname,
                    const iaamt_model & model,
                    const iaamt_bc_out & out,
                    int64_t window_start_sample,
                    std::string & err);

//
// MIDI I/O (iaamt-midi.cpp)
//

bool iaamt_write_midi(const std::string & fname,
                      const std::vector<iaamt_note> & notes,
                      int sample_rate,
                      std::string & err);

// Reads a type-0 or type-1 SMF into notes, resolving tempo changes.  Program and
// drum flags come from program-change events and channel 9.
bool iaamt_read_midi(const std::string & fname,
                     int sample_rate,
                     std::vector<iaamt_note> & notes,
                     std::string & err);
