"""
Fast distinct-value counting for float16 columns.

Shared by both `patchInputs` (per-weight patch SELECTION) and `cpaExtractCore`
(the per-(patch, weight) unique-count MATRIX the extractor/sign-correction use
to choose a patch). Kept in its own NumPy-only module so neither caller has to
duplicate the bit-pattern handling and so `patchInputs` stays free of the CuPy
import that `cpaExtractCore` pulls in.

Method: reinterpret the float16 column as uint16 and count distinct codes with a
65536-slot presence buffer (O(N), no sort) -- ~8x faster than np.unique(col).size
and byte-identical to it, after canonicalising the two float16 value-equality
quirks:
  * +0.0 (0x0000) and -0.0 (0x8000) are distinct bit patterns but equal values
  * every NaN bit pattern is a distinct code but np.unique collapses them to one
(ported from cpa_resnet/utils.compute_unique_count_matrix).
"""

import numpy as np

_F16_NEG_ZERO = np.uint16(0x8000)
_F16_POS_ZERO = np.uint16(0x0000)
_F16_CANON_NAN = np.uint16(0x7E00)


def new_present_buffer():
    """A reusable all-False 65536-slot presence buffer for count_distinct."""
    return np.zeros(65536, dtype=bool)


def count_distinct(col, present):
    """Distinct float16 values in a 1-D column, using the shared `present`
    buffer (left all-False on return). Byte-identical to np.unique(col).size."""
    col = np.ascontiguousarray(col, dtype=np.float16)
    codes = col.view(np.uint16).copy()
    codes[codes == _F16_NEG_ZERO] = _F16_POS_ZERO
    nan_mask = np.isnan(col)
    if nan_mask.any():
        codes[nan_mask] = _F16_CANON_NAN
    present[codes] = True
    n = int(np.count_nonzero(present))
    present[codes] = False  # cheap targeted reset (only the codes we set)
    return n
