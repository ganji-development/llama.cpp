// Minimal SMF writer (type 0) and reader (type 0 and 1).

#include "iaamt.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <vector>

namespace {

const int    TICKS_PER_QUARTER = 480;
const int    TEMPO_US_PER_QN   = 500000;             // 120 BPM
const double TICKS_PER_SECOND  = TICKS_PER_QUARTER * 1e6 / TEMPO_US_PER_QN;

void push_u16(std::vector<uint8_t> & v, uint16_t x) {
    v.push_back((uint8_t) (x >> 8));
    v.push_back((uint8_t) (x & 0xff));
}

void push_u32(std::vector<uint8_t> & v, uint32_t x) {
    v.push_back((uint8_t) (x >> 24));
    v.push_back((uint8_t) (x >> 16));
    v.push_back((uint8_t) (x >>  8));
    v.push_back((uint8_t) (x & 0xff));
}

void push_varlen(std::vector<uint8_t> & v, uint32_t x) {
    uint8_t buf[4];
    int n = 0;
    buf[n++] = (uint8_t) (x & 0x7f);
    x >>= 7;
    while (x) {
        buf[n++] = (uint8_t) ((x & 0x7f) | 0x80);
        x >>= 7;
    }
    for (int i = n - 1; i >= 0; --i) {
        v.push_back(buf[i]);
    }
}

struct midi_event {
    uint32_t tick;
    bool     on;
    uint8_t  pitch;
    uint8_t  velocity;
};

} // namespace

bool iaamt_write_midi(const std::string & fname,
                      const std::vector<iaamt_note> & notes,
                      int sample_rate,
                      std::string & err) {
    std::vector<midi_event> events;
    events.reserve(notes.size() * 2);

    for (const iaamt_note & n : notes) {
        if (n.pitch < 0 || n.pitch > 127) {
            continue;
        }
        uint32_t on  = (uint32_t) std::max<int64_t>(0,
            llround((double) n.start_sample / sample_rate * TICKS_PER_SECOND));
        uint32_t off = (uint32_t) std::max<int64_t>(0,
            llround((double) n.end_sample / sample_rate * TICKS_PER_SECOND));
        if (off <= on) {
            off = on + 1;   // keep every note audible
        }
        events.push_back({ on,  true,  (uint8_t) n.pitch,
                           (uint8_t) std::max(1, std::min(127, n.velocity)) });
        events.push_back({ off, false, (uint8_t) n.pitch, 0 });
    }

    // note-off must precede note-on at the same tick so repeats retrigger
    std::sort(events.begin(), events.end(), [](const midi_event & a, const midi_event & b) {
        if (a.tick != b.tick) {
            return a.tick < b.tick;
        }
        if (a.on != b.on) {
            return !a.on;
        }
        return a.pitch < b.pitch;
    });

    std::vector<uint8_t> track;

    // tempo
    push_varlen(track, 0);
    track.push_back(0xff);
    track.push_back(0x51);
    track.push_back(0x03);
    track.push_back((uint8_t) (TEMPO_US_PER_QN >> 16));
    track.push_back((uint8_t) (TEMPO_US_PER_QN >> 8));
    track.push_back((uint8_t) (TEMPO_US_PER_QN & 0xff));

    uint32_t prev = 0;
    for (const midi_event & e : events) {
        push_varlen(track, e.tick - prev);
        prev = e.tick;
        track.push_back(e.on ? 0x90 : 0x80);
        track.push_back(e.pitch);
        track.push_back(e.velocity);
    }

    // end of track
    push_varlen(track, 0);
    track.push_back(0xff);
    track.push_back(0x2f);
    track.push_back(0x00);

    std::vector<uint8_t> file;
    file.insert(file.end(), { 'M', 'T', 'h', 'd' });
    push_u32(file, 6);
    push_u16(file, 0);                       // format 0
    push_u16(file, 1);                       // one track
    push_u16(file, TICKS_PER_QUARTER);
    file.insert(file.end(), { 'M', 'T', 'r', 'k' });
    push_u32(file, (uint32_t) track.size());
    file.insert(file.end(), track.begin(), track.end());

    std::ofstream fout(fname, std::ios::binary);
    if (!fout) {
        err = "failed to open " + fname + " for writing";
        return false;
    }
    fout.write((const char *) file.data(), (std::streamsize) file.size());
    if (!fout) {
        err = "failed while writing " + fname;
        return false;
    }
    return true;
}

namespace {

struct reader {
    const uint8_t * p;
    const uint8_t * end;
    bool ok = true;

    uint8_t u8() {
        if (p >= end) { ok = false; return 0; }
        return *p++;
    }
    uint32_t be(int n) {
        uint32_t v = 0;
        for (int i = 0; i < n; ++i) {
            v = (v << 8) | u8();
        }
        return v;
    }
    uint32_t varlen() {
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            const uint8_t b = u8();
            v = (v << 7) | (b & 0x7f);
            if (!(b & 0x80)) {
                break;
            }
        }
        return v;
    }
};

struct tempo_change {
    uint32_t tick;
    uint32_t us_per_qn;
};

// Integrates the tempo map so a tick converts to seconds in one lookup.
double tick_to_seconds(const std::vector<tempo_change> & tempos,
                       const std::vector<double> & starts,
                       int tpq,
                       uint32_t tick) {
    size_t i = 0;
    while (i + 1 < tempos.size() && tempos[i + 1].tick <= tick) {
        ++i;
    }
    return starts[i] + (double) (tick - tempos[i].tick) * tempos[i].us_per_qn
           / 1e6 / (double) tpq;
}

} // namespace

bool iaamt_read_midi(const std::string & fname,
                     int sample_rate,
                     std::vector<iaamt_note> & notes,
                     std::string & err) {
    std::ifstream fin(fname, std::ios::binary | std::ios::ate);
    if (!fin) {
        err = "failed to open " + fname;
        return false;
    }
    const std::streamsize size = fin.tellg();
    fin.seekg(0);
    std::vector<uint8_t> buf((size_t) size);
    fin.read((char *) buf.data(), size);
    if (!fin) {
        err = "failed to read " + fname;
        return false;
    }

    reader r{ buf.data(), buf.data() + buf.size() };
    if (buf.size() < 14 || std::memcmp(buf.data(), "MThd", 4) != 0) {
        err = fname + " is not a Standard MIDI File";
        return false;
    }
    r.p += 4;
    const uint32_t hdr_len = r.be(4);
    r.be(2);                                  // format
    const uint16_t n_tracks = (uint16_t) r.be(2);
    const int16_t  division = (int16_t)  r.be(2);
    if (division <= 0) {
        err = "SMPTE time division is not supported";
        return false;
    }
    const int tpq = division;
    r.p += (hdr_len > 6) ? (hdr_len - 6) : 0;

    // Pass 1 gathers the tempo map; ticks are absolute so tracks are independent.
    std::vector<tempo_change> tempos{ { 0, 500000 } };
    struct raw_event {
        uint32_t tick;
        uint8_t  status;
        uint8_t  d1;
        uint8_t  d2;
    };
    std::vector<raw_event> events;

    for (uint16_t t = 0; t < n_tracks && r.ok; ++t) {
        if (r.p + 8 > r.end || std::memcmp(r.p, "MTrk", 4) != 0) {
            break;
        }
        r.p += 4;
        const uint32_t len = r.be(4);
        const uint8_t * track_end = r.p + len;
        if (track_end > r.end) {
            track_end = r.end;
        }

        uint32_t tick = 0;
        uint8_t  running = 0;
        while (r.p < track_end && r.ok) {
            tick += r.varlen();
            uint8_t status = r.u8();
            if (status < 0x80) {
                // running status: the byte we read is really the first data byte
                r.p--;
                status = running;
                if (status < 0x80) {
                    break;
                }
            } else if (status < 0xf0) {
                running = status;
            }

            if (status == 0xff) {
                const uint8_t type = r.u8();
                const uint32_t n = r.varlen();
                if (type == 0x51 && n == 3) {
                    const uint32_t us = r.be(3);
                    if (us > 0) {
                        tempos.push_back({ tick, us });
                    }
                } else {
                    r.p += n;
                }
                if (type == 0x2f) {
                    break;
                }
            } else if (status == 0xf0 || status == 0xf7) {
                r.p += r.varlen();
            } else {
                const uint8_t hi = status & 0xf0;
                const uint8_t d1 = r.u8();
                const uint8_t d2 = (hi == 0xc0 || hi == 0xd0) ? 0 : r.u8();
                if (hi == 0x80 || hi == 0x90 || hi == 0xc0) {
                    events.push_back({ tick, status, d1, d2 });
                }
            }
        }
        r.p = track_end;
    }

    std::sort(tempos.begin(), tempos.end(),
              [](const tempo_change & a, const tempo_change & b) { return a.tick < b.tick; });
    std::vector<double> starts(tempos.size(), 0.0);
    for (size_t i = 1; i < tempos.size(); ++i) {
        starts[i] = starts[i - 1] + (double) (tempos[i].tick - tempos[i - 1].tick)
                    * tempos[i - 1].us_per_qn / 1e6 / (double) tpq;
    }

    // Pass 2 pairs note-ons with note-offs, in tick order across all tracks.
    std::stable_sort(events.begin(), events.end(),
                     [](const raw_event & a, const raw_event & b) { return a.tick < b.tick; });

    int program[16] = { 0 };
    struct pending {
        uint32_t tick;
        int      program;
    };
    std::map<int, std::vector<pending>> open;   // key = channel * 128 + pitch

    notes.clear();
    for (const raw_event & e : events) {
        const uint8_t hi = e.status & 0xf0;
        const int     ch = e.status & 0x0f;
        if (hi == 0xc0) {
            program[ch] = e.d1;
            continue;
        }
        const int key = ch * 128 + e.d1;
        if (hi == 0x90 && e.d2 > 0) {
            open[key].push_back({ e.tick, program[ch] });
            continue;
        }
        // note-off, or note-on with zero velocity
        auto it = open.find(key);
        if (it == open.end() || it->second.empty()) {
            continue;
        }
        const pending on = it->second.front();
        it->second.erase(it->second.begin());

        const double t0 = tick_to_seconds(tempos, starts, tpq, on.tick);
        const double t1 = tick_to_seconds(tempos, starts, tpq, e.tick);

        iaamt_note note;
        note.pitch        = e.d1;
        note.start_sample = (int64_t) llround(t0 * sample_rate);
        note.end_sample   = std::max<int64_t>(note.start_sample + 1,
                                              (int64_t) llround(t1 * sample_rate));
        note.program      = on.program;
        note.is_drum      = (ch == 9);
        notes.push_back(note);
    }

    std::sort(notes.begin(), notes.end(), [](const iaamt_note & a, const iaamt_note & b) {
        if (a.start_sample != b.start_sample) return a.start_sample < b.start_sample;
        return a.pitch < b.pitch;
    });
    return true;
}
