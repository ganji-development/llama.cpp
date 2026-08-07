// Flat C ABI for the iaamt transcriber.
//
// iaamt.h is a C++ header: it passes std::vector and std::string by reference
// and returns containers by value, so a consumer must be built by the same
// compiler with the same standard library and the same CRT. MIDIGen is not -
// the core libraries build with MSVC against the static runtime while this tree
// builds with ROCm's clang against the dynamic one - so nothing in the main
// repository can include iaamt.h and link the objects.
//
// This header is what crosses that boundary. It is C, it passes only
// primitives, pointers and POD structs, and every allocation it hands out is
// freed by the function provided here rather than by the caller's allocator,
// because the two do not share a heap. It mirrors what libs/model-core already
// does for llama.dll: resolve by name from a DLL and never link the C++ side.
//
// Nothing in here throws. Failures are reported through the return value and a
// caller-provided error buffer.

#ifndef IAAMT_C_H
#define IAAMT_C_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(IAAMT_C_BUILD)
#    define IAAMT_C_API __declspec(dllexport)
#  else
#    define IAAMT_C_API __declspec(dllimport)
#  endif
#else
#  define IAAMT_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Bumped on any change to a struct layout or a function signature here. A
// consumer resolving these symbols at run time cannot be recompiled in step
// with this library, so it must check this before trusting anything else: a
// mismatched struct read through a stale layout is silent corruption, not a
// load failure.
#define IAAMT_C_ABI_VERSION 1

typedef struct iaamt_session iaamt_session;

// One transcribed note. Mirrors iaamt_note, minus the fields that only apply to
// MIDI round-tripping, with bools widened to int so the layout does not depend
// on how a compiler packs them.
//
// Times are in samples at the rate iaamt_c_sample_rate() reports, which is the
// model's rate and not necessarily the source's.
typedef struct {
    int32_t pitch;              // MIDI note number
    int32_t slot;               // pitch slot, for multi-slot checkpoints
    int64_t start_sample;
    int64_t end_sample;
    int32_t velocity;           // 1..127
    int32_t has_onset;          // nonzero if the model asserts an onset here
    int32_t has_offset;         // nonzero if the model asserts an offset here

    // Semi-CRF score of the interval. Unbounded and positive, larger where the
    // model was more willing to pay for the note. Not a probability: it is
    // length-scaled and its scale is per-model, so it ranks notes within one
    // transcription but does not support a threshold carried between models.
    // The decode admits an interval exactly when this exceeds zero, so
    // sigmoid() of it is the mapping the decoder itself implies - and no note
    // that survives can therefore fall below 0.5.
    float crf_score;

    // Boundary-head probabilities in 0..1, or negative when the head did not
    // run. Negative is "unknown", not "improbable" - the flags above then
    // reflect where the interval sat in its analysis window rather than
    // anything the model asserted, and reading it as a low score would turn
    // window layout into evidence.
    //
    // Measured behaviour a consumer should know: the onset head is saturated.
    // Across four checkpoints and three sources it sits at 1.0 for roughly 98%
    // of notes. It is a gate that flags the few onsets the model doubts, not a
    // ranking signal; crf_score is what ranks.
    float onset_confidence;
    float offset_confidence;
} iaamt_c_note;

typedef struct {
    float   note_bias;
    float   merge_onset_ms;
    float   merge_gap_ms;      // negative falls back to one hop
    int32_t velocity;
    float   min_note_ms;
    int32_t use_boundary_head; // nonzero to run the boundary head
    // Analysis window. Not a performance knob: the semi-CRF decodes each
    // window independently, so this changes the transcription. On one bass stem
    // the default 8000 gives 408 notes where 60000 gives 27. Change it only
    // with a measurement in hand.
    float   window_ms;
    float   stride_ms;         // zero or less means half the window

    // Windows quieter than this are skipped entirely. Negative, in dBFS.
    // Zero or above disables the gate: 0 dBFS is full scale, so a gate there
    // would skip every window, which can only be a mistake. The CLI computes a
    // gate for 0 as well; this refuses instead, because a silent transcription
    // from a plausible-looking parameter is worse than an ignored one.
    float   silence_dbfs;
    int32_t n_threads;         // zero or less picks a default
} iaamt_c_params;

// Always safe to call, including before any session exists.
IAAMT_C_API int32_t iaamt_c_abi_version(void);

// Fills in the same defaults the CLI uses. Call this before overriding fields
// so that a parameter added in a later version does not arrive as zero.
IAAMT_C_API void iaamt_c_default_params(iaamt_c_params * params);

// Loads a GGUF checkpoint. Returns NULL on failure and writes a reason into
// `err` when `err` is non-NULL. `use_gpu` selects the HIP/CUDA backend when one
// is available and silently stays on CPU when it is not.
IAAMT_C_API iaamt_session * iaamt_c_open(const char * gguf_path,
                                         int32_t      use_gpu,
                                         char *       err,
                                         size_t       err_size);

IAAMT_C_API void iaamt_c_close(iaamt_session * session);

// The rate and channel count the model expects. A caller must resample and
// channel-match before calling iaamt_c_transcribe; this library deliberately
// does no resampling, because the caller already owns an audio pipeline and two
// resamplers disagreeing is worse than one.
IAAMT_C_API int32_t iaamt_c_sample_rate(const iaamt_session * session);
IAAMT_C_API int32_t iaamt_c_channels(const iaamt_session * session);

// True for checkpoints that transcribe audio. The velocity and beat/chord
// checkpoints share this file format but score existing notes instead, and
// iaamt_c_transcribe rejects them rather than returning something meaningless.
IAAMT_C_API int32_t iaamt_c_is_transcription_model(const iaamt_session * session);

// Transcribes planar audio. `channels` holds `n_channels` pointers to
// `n_samples` floats each, at iaamt_c_sample_rate(). Passing fewer channels
// than the model wants duplicates the last one, which is what a mono source
// needs against a stereo model.
//
// On success returns 0, sets `*out_notes` and `*out_count`, and the caller must
// release the array with iaamt_c_free_notes. `*out_notes` is NULL when the
// model found nothing, which is a success and not an error. On failure returns
// nonzero and leaves the outputs untouched.
IAAMT_C_API int32_t iaamt_c_transcribe(iaamt_session *            session,
                                       const float * const *      channels,
                                       int32_t                    n_channels,
                                       int64_t                    n_samples,
                                       const iaamt_c_params *     params,
                                       iaamt_c_note **            out_notes,
                                       size_t *                   out_count,
                                       char *                     err,
                                       size_t                     err_size);

// Frees an array returned by iaamt_c_transcribe. Must be used instead of the
// caller's free: the array was allocated by this library's CRT.
IAAMT_C_API void iaamt_c_free_notes(iaamt_c_note * notes);

#ifdef __cplusplus
}
#endif

#endif // IAAMT_C_H
