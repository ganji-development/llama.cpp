from __future__ import annotations

import re

from typing import Any, Iterable, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, MmprojModel, gguf
from .qwen import Qwen3Model


@ModelBase.register("MossTTSDelayModel", "MossTTSDelayForCausalLM")
class MossTTSDelayModel(Qwen3Model):
    model_arch = gguf.MODEL_ARCH.MOSS_TTS_DELAY

    def __init__(self, *args, **kwargs):
        hparams = kwargs.get("hparams")
        if hparams is None:
            hparams = ModelBase.load_hparams(args[0], self.is_mistral_format)
        else:
            hparams = dict(hparams)

        language_config = hparams.get("language_config")
        if isinstance(language_config, dict):
            # Expose the Qwen3 backbone params at the root level so TextModel can
            # discover block_count / hidden_size / attention params without
            # losing the top-level MOSS architecture identity.
            language_hparams = {
                key: value
                for key, value in language_config.items()
                if key not in ("architectures", "model_type")
            }
            hparams = {**hparams, **language_hparams}

        kwargs["hparams"] = hparams
        super().__init__(*args, **kwargs)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        arch = self.gguf_writer.arch
        self.gguf_writer.add_uint32(gguf.Keys.LLM.N_VQ.format(arch=arch), self.hparams["n_vq"])
        self.gguf_writer.add_uint32(gguf.Keys.LLM.AUDIO_VOCAB_SIZE.format(arch=arch), self.hparams["audio_vocab_size"])
        self.gguf_writer.add_uint32(gguf.Keys.LLM.AUDIO_PAD_CODE.format(arch=arch), self.hparams["audio_pad_code"])
        self.gguf_writer.add_uint32(gguf.Keys.LLM.AUDIO_START_TOKEN_ID.format(arch=arch), self.hparams["audio_start_token_id"])
        self.gguf_writer.add_uint32(gguf.Keys.LLM.AUDIO_END_TOKEN_ID.format(arch=arch), self.hparams["audio_end_token_id"])
        self.gguf_writer.add_uint32(gguf.Keys.LLM.AUDIO_USER_SLOT_TOKEN_ID.format(arch=arch), self.hparams["audio_user_slot_token_id"])
        self.gguf_writer.add_uint32(
            gguf.Keys.LLM.AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID.format(arch=arch),
            self.hparams["audio_assistant_gen_slot_token_id"],
        )
        self.gguf_writer.add_uint32(
            gguf.Keys.LLM.AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID.format(arch=arch),
            self.hparams["audio_assistant_delay_slot_token_id"],
        )
        if (sampling_rate := self.hparams.get("sampling_rate")) is not None:
            self.gguf_writer.add_uint32(gguf.Keys.LLM.SAMPLING_RATE.format(arch=arch), sampling_rate)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("language_model."):
            name = name.replace("language_model.", "", 1)

        if (match := re.fullmatch(r"emb_ext\.(\d+)\.weight", name)) is not None:
            vq_idx = int(match.group(1))
            yield (f"{gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.TOKEN_EMBD_AUDIO]}.{vq_idx}.weight", data_torch)
            return

        if (match := re.fullmatch(r"lm_heads\.(\d+)\.weight", name)) is not None:
            head_idx = int(match.group(1))
            if head_idx == 0:
                yield (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.OUTPUT] + ".weight", data_torch)
            else:
                yield (f"{gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.OUTPUT_AUDIO]}.{head_idx - 1}.weight", data_torch)
            return

        yield from super().modify_tensors(data_torch, name, bid)


def moss_music_n_deepstack(global_config: dict[str, Any]) -> int:
    # Number of DeepStack mergers that are actually injected into the LLM.
    # Merger k is added to the output of decoder layer k, so this is also the
    # number of decoder layers touched. Both the text and the mmproj converter
    # must agree on it: it sets n_embd_inp on one side and the projector output
    # width on the other, and mtmd refuses to load the pair if they disagree.
    audio_config = global_config.get("audio_config") or {}
    n_indexes = len(audio_config.get("deepstack_encoder_layer_indexes") or [])
    n_inject = global_config.get("deepstack_num_inject_layers")
    return n_indexes if n_inject is None else min(n_indexes, int(n_inject))


@ModelBase.register("MossMusicModel")
class MossMusicModel(Qwen3Model):
    # MOSS-Music is an audio-understanding model: a Whisper-style audio encoder
    # feeding a plain Qwen3-8B backbone. Only the backbone lands in this file;
    # the encoder and the projectors are written by MossMusicWhisperEncoderModel
    # into a separate mmproj GGUF (--mmproj).
    model_arch = gguf.MODEL_ARCH.MOSS_MUSIC

    def __init__(self, *args, **kwargs):
        hparams = kwargs.get("hparams")
        if hparams is None:
            hparams = ModelBase.load_hparams(args[0], self.is_mistral_format)
        else:
            hparams = dict(hparams)

        language_config = hparams.get("language_config")
        if isinstance(language_config, dict):
            # Expose the Qwen3 backbone params at the root level so TextModel can
            # discover block_count / hidden_size / attention params without
            # losing the top-level MOSS architecture identity.
            language_hparams = {
                key: value
                for key, value in language_config.items()
                if key not in ("architectures", "model_type")
            }
            hparams = {**hparams, **language_hparams}

        kwargs["hparams"] = hparams
        super().__init__(*args, **kwargs)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # DeepStack: the mmproj emits one widened embedding row per audio token,
        # [main | ds_0 | ds_1 | ds_2], and the decoder adds slice k+1 to the
        # output of layer k. This count is what widens n_embd_inp accordingly.
        self.gguf_writer.add_num_deepstack_layers(moss_music_n_deepstack(self.hparams))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # audio encoder and projectors belong to the mmproj file
        if name.startswith(("audio_encoder.", "audio_adapter.", "deepstack_audio_merger_list.")):
            return

        if name.startswith("language_model."):
            name = name.replace("language_model.", "", 1)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("MossMusicModel")
class MossMusicWhisperEncoderModel(MmprojModel):
    # MOSS-Music's audio tower is a Whisper encoder stack behind a *Conv2d* stem
    # (Whisper/Ultravox use Conv1d), followed by GatedMLP projectors:
    #   audio_adapter                 -> merged into the input embeddings
    #   deepstack_audio_merger_list.K -> added to the output of LLM layer K
    # The DeepStack mergers consume encoder hidden states from the layers listed
    # in audio_config.deepstack_encoder_layer_indexes.
    #
    # Not derived from WhisperEncoderModel: the Conv1d bias fixup it applies to
    # conv1/conv2 is wrong for a Conv2d stem.
    has_vision_encoder = False
    has_audio_encoder = True

    n_conv2d_layers = 3  # conv1, conv2, conv3 -- fixed by the reference model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.hparams is audio_config here; give the Whisper-style names the
        # aliases MmprojModel.set_gguf_parameters looks for.
        self.hparams["hidden_size"] = self.hparams["d_model"]
        self.hparams["intermediate_size"] = self.hparams["encoder_ffn_dim"]
        self.hparams["num_attention_heads"] = self.hparams["encoder_attention_heads"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.MOSS_MUSIC)
        self.gguf_writer.add_audio_num_mel_bins(self.hparams["num_mel_bins"])
        self.gguf_writer.add_audio_attention_layernorm_eps(self.hparams.get("layer_norm_eps", 1e-5))
        self.gguf_writer.add_audio_downsample_rate(self.hparams["downsample_rate"])
        self.gguf_writer.add_audio_num_conv2d_layers(self.n_conv2d_layers)
        self.gguf_writer.add_audio_adapter_hidden_size(self.global_config["adapter_hidden_size"])

        # encoder layers whose hidden states feed the DeepStack mergers
        deepstack_indexes = list(self.hparams.get("deepstack_encoder_layer_indexes") or [])
        n_inject = moss_music_n_deepstack(self.global_config)

        self.gguf_writer.add_audio_deepstack_layer_indexes(deepstack_indexes[:n_inject])
        self.gguf_writer.add_audio_deepstack_num_inject(n_inject)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # SinusoidsPositionEmbedding is a non-trainable buffer that the reference
        # implementation rebuilds per forward pass. Precompute the full table so
        # the runtime can just slice the first seq_len rows.
        # Matches SinusoidsPositionEmbedding in modeling_moss_music.py.
        max_timescale = 10000
        length = self.hparams["max_source_positions"]
        channels = self.hparams["hidden_size"]  # == d_model
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        pos_embd = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1).to(dtype=torch.float32)
        yield ("audio_encoder.embed_positions.weight", pos_embd)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".conv" in name and ".weight" in name:
            return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # skip the Qwen3 backbone, it goes into the main GGUF
        if name.startswith("language_model.") or name.startswith("lm_head."):
            return

        # rebuilt in generate_extra_tensors() as the full position table
        if name == "audio_encoder.embed_positions.inv_timescales":
            return

        if re.fullmatch(r"audio_encoder\.conv[123]\.bias", name):
            # ggml_conv_2d yields [OW, OH, OC, N]; store the bias as [1, 1, OC]
            # so it broadcasts over the frequency and time axes.
            data_torch = data_torch.reshape(-1, 1, 1)

        # Conv2d weights are [OC, IC, KH, KW] in torch, which GGUF stores as
        # ne = [KW, KH, IC, OC] -- exactly ggml_conv_2d's kernel layout, so they
        # pass through untouched.

        yield from super().modify_tensors(data_torch, name, bid)
