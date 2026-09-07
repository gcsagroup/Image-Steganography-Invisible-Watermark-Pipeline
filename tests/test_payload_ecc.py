from dataclasses import replace

import pytest

from PipeLine.configuration import load_config
from PipeLine.errors import PayloadCapacityError
from PipeLine.preprocessing.ecc import hamming74_decode, hamming74_encode
from PipeLine.preprocessing.payload import PayloadCodec, bits_to_bytes, bytes_to_bits


def test_byte_bit_round_trip():
    data = bytes([0x00, 0xA1, 0xFF])
    assert bits_to_bytes(bytes_to_bits(data)) == data


@pytest.mark.parametrize("error_index", range(7))
def test_hamming74_corrects_every_single_bit(error_index):
    source = [1, 0, 1, 1]
    encoded = hamming74_encode(source)
    encoded[error_index] ^= 1
    decoded, stats = hamming74_decode(encoded)
    assert decoded == source
    assert stats.corrected_blocks == 1


@pytest.mark.parametrize("text", ["VAE 隐写", "ASCII", "结束标记"])
def test_payload_text_round_trip(text):
    config = load_config()
    codec = PayloadCodec(config.preprocessing)
    encoded = codec.encode_text(text, 6000)
    decoded = codec.decode(encoded.final_bits + [0, 1, 0])
    assert decoded.status == "ok"
    assert decoded.decoded_text == text
    assert decoded.payload_bits == encoded.original_bits


def test_capacity_includes_ecc_and_fixed_header():
    codec = PayloadCodec(load_config().preprocessing)
    with pytest.raises(PayloadCapacityError):
        codec.encode_bytes(b"x" * 600, 600)
    assert codec.maximum_payload_bytes(6000, 1.0) > 0


def test_incomplete_header_is_explicit():
    codec = PayloadCodec(load_config().preprocessing)
    decoded = codec.decode([0, 1] * 20)
    assert decoded.status == "incomplete_header"


def test_repeated_header_uses_per_bit_majority_vote():
    codec = PayloadCodec(load_config().preprocessing)
    assert codec.logical_header_bit_length == 67
    assert codec.header_bit_length == 201
    encoded = codec.encode_text("header majority", 6000)
    damaged = list(encoded.final_bits)
    for index in range(codec.logical_header_bit_length):
        damaged[index] ^= 1
    decoded = codec.decode(damaged)
    assert decoded.status == "ok"
    assert decoded.decoded_text == "header majority"
    assert decoded.metadata is not None
    assert decoded.metadata.header_bit_length == codec.header_bit_length


def test_payload_crc_detects_uncorrected_damage():
    config = load_config()
    prep = replace(
        config.preprocessing,
        ecc=replace(config.preprocessing.ecc, enabled=False),
    )
    codec = PayloadCodec(prep)
    encoded = codec.encode_bytes(b"payload", 6000)
    damaged = list(encoded.final_bits)
    damaged[codec.header_bit_length] ^= 1
    decoded = codec.decode(damaged)
    assert decoded.status == "payload_crc_error"
    assert decoded.decoded_text is None


def test_header_carries_text_encoding_and_ecc_scheme():
    config = load_config()
    prep = replace(config.preprocessing, text_encoding="gbk")
    encoded = PayloadCodec(prep).encode_text("帧头", 6000)
    # Decode with the default UTF-8 runtime config; the frame must select GBK.
    decoded = PayloadCodec(config.preprocessing).decode(encoded.final_bits)
    assert decoded.status == "ok"
    assert decoded.decoded_text == "帧头"
    assert decoded.metadata is not None
    assert decoded.metadata.text_encoding == "gbk"
    assert decoded.metadata.ecc_scheme == "hamming74"


@pytest.mark.parametrize(
    ("encoding", "text"),
    [("utf-8", "隐写"), ("gbk", "隐写"), ("utf-16", "隐写"), ("ascii", "stego")],
)
def test_all_configured_text_encodings(encoding, text):
    config = load_config()
    prep = replace(config.preprocessing, text_encoding=encoding)
    codec = PayloadCodec(prep)
    encoded = codec.encode_text(text, 6000)
    assert codec.decode(encoded.final_bits).decoded_text == text


def test_ecc_can_be_disabled():
    config = load_config()
    prep = replace(
        config.preprocessing,
        ecc=replace(config.preprocessing.ecc, enabled=False),
    )
    codec = PayloadCodec(prep)
    encoded = codec.encode_text("none", 6000)
    assert encoded.ecc_bits == encoded.original_bits
    assert codec.decode(encoded.final_bits).decoded_text == "none"
