"""LZ-String URI-safe compression and decompression compatible with Jstris."""

from __future__ import annotations

from typing import Callable

KEY_STR_URI_SAFE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-$"
BASE_REVERSE_DIC: dict[str, dict[str, int]] = {}


def _get_base_value(alphabet: str, character: str) -> int:
    if alphabet not in BASE_REVERSE_DIC:
        BASE_REVERSE_DIC[alphabet] = {ch: idx for idx, ch in enumerate(alphabet)}
    return BASE_REVERSE_DIC[alphabet][character]


def decompress_from_encoded_uri_component(input_str: str | None) -> str | None:
    """Decompress an LZ-String URI-safe encoded string."""
    if input_str is None:
        return ""
    if input_str == "":
        return None
    normalized = input_str.replace(" ", "+")
    return _decompress(
        len(normalized),
        32,
        lambda index: _get_base_value(KEY_STR_URI_SAFE, normalized[index]),
    )


def _decompress(length: int, reset_value: int, get_next_value: Callable[[int], int]) -> str | None:
    dictionary: dict[int, str] = {0: "0", 1: "1", 2: "2"}
    enlarge_in = 4
    dict_size = 4
    num_bits = 3
    result: list[str] = []

    data_val = get_next_value(0)
    data_position = reset_value
    data_index = 1

    def read_bits(n_bits: int) -> int:
        nonlocal data_val, data_position, data_index
        bits = 0
        maxpower = 1 << n_bits
        power = 1
        while power != maxpower:
            resb = data_val & data_position
            data_position >>= 1
            if data_position == 0:
                data_position = reset_value
                if data_index < length:
                    data_val = get_next_value(data_index)
                else:
                    data_val = 0
                data_index += 1
            bits |= power if resb > 0 else 0
            power <<= 1
        return bits

    next_type = read_bits(2)
    if next_type == 0:
        c = chr(read_bits(8))
    elif next_type == 1:
        c = chr(read_bits(16))
    elif next_type == 2:
        return ""
    else:
        return None

    dictionary[3] = c
    w = c
    result.append(c)

    while True:
        if data_index > length:
            return ""

        cc = read_bits(num_bits)
        if cc == 0:
            dictionary[dict_size] = chr(read_bits(8))
            cc = dict_size
            dict_size += 1
            enlarge_in -= 1
        elif cc == 1:
            dictionary[dict_size] = chr(read_bits(16))
            cc = dict_size
            dict_size += 1
            enlarge_in -= 1
        elif cc == 2:
            return "".join(result)

        if enlarge_in == 0:
            enlarge_in = 1 << num_bits
            num_bits += 1

        if cc in dictionary:
            entry = dictionary[cc]
        else:
            if cc == dict_size:
                entry = w + w[0]
            else:
                return None

        result.append(entry)
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
        enlarge_in -= 1
        w = entry

        if enlarge_in == 0:
            enlarge_in = 1 << num_bits
            num_bits += 1


def compress_to_encoded_uri_component(input_str: str | None) -> str:
    """Compress a string to LZ-String URI-safe format."""
    if input_str is None:
        return ""
    return _compress(input_str, 6, lambda a: KEY_STR_URI_SAFE[a])


def _compress(uncompressed: str, bits_per_char: int, get_char_from_int: Callable[[int], str]) -> str:
    if uncompressed is None:
        return ""
    context_dictionary: dict[str, int] = {}
    context_dictionary_to_create: set[str] = set()
    context_w = ""
    context_enlarge_in = 2
    context_dict_size = 3
    context_num_bits = 2
    context_data: list[str] = []
    context_data_val = 0
    context_data_position = 0

    def write_bit(bit: int) -> None:
        nonlocal context_data_val, context_data_position
        context_data_val = (context_data_val << 1) | (bit & 1)
        if context_data_position == bits_per_char - 1:
            context_data_position = 0
            context_data.append(get_char_from_int(context_data_val))
            context_data_val = 0
        else:
            context_data_position += 1

    def write_bits(n_bits: int, value: int) -> None:
        val = value
        for _ in range(n_bits):
            write_bit(val & 1)
            val >>= 1

    def produce_w() -> None:
        nonlocal context_enlarge_in, context_num_bits
        if context_w in context_dictionary_to_create:
            ch_code = ord(context_w[0])
            if ch_code < 256:
                write_bits(context_num_bits, 0)
                write_bits(8, ch_code)
            else:
                write_bits(context_num_bits, 1)
                write_bits(16, ch_code)
            context_enlarge_in -= 1
            if context_enlarge_in == 0:
                context_enlarge_in = 1 << context_num_bits
                context_num_bits += 1
            context_dictionary_to_create.remove(context_w)
        else:
            write_bits(context_num_bits, context_dictionary[context_w])
        context_enlarge_in -= 1
        if context_enlarge_in == 0:
            context_enlarge_in = 1 << context_num_bits
            context_num_bits += 1

    for ch in uncompressed:
        if ch not in context_dictionary:
            context_dictionary[ch] = context_dict_size
            context_dict_size += 1
            context_dictionary_to_create.add(ch)

        context_wc = context_w + ch
        if context_wc in context_dictionary:
            context_w = context_wc
        else:
            produce_w()
            context_dictionary[context_wc] = context_dict_size
            context_dict_size += 1
            context_w = ch

    if context_w != "":
        produce_w()

    write_bits(context_num_bits, 2)

    while True:
        context_data_val <<= 1
        if context_data_position == bits_per_char - 1:
            context_data.append(get_char_from_int(context_data_val))
            break
        context_data_position += 1

    return "".join(context_data)
