from typing import List, Optional, Tuple

import numpy as np


class Tokenizer:
    """
    An encoding scheme for creating discrete token sequences from vector
    graphics. Images defined in terms of loops of Bezier curves can be encoded
    via encode_loops() and decoded via decode_loops().

    Floating point values are represented as two tokens: a significant token
    and a fractional token, where the former selects the broad range and the
    latter selects a precise value within it.

    The vocabulary space looks as follows:
     - end
     - start
     - close path
     - ...significant tokens...
     - ...fractional tokens...

    To avoid redundant information, control points are often encoded as a
    scalar [s1] or [s2] multiplied by the known tangent to the curve (since
    smoothness is always preserved). The final curve is encoded by two control
    points of this form, since its endpoints and tangents are fully
    constrained.

    Here is how a loop is encoded. Each loop begins with a <start> token.
    The first bezier of every loop is fully encoded, and the following curves
    consist of one scalar s1 followed by two 2D points. The final curve starts
    with a <close path> token and then consists of two scalars s1, s2.

        <start>
        [x0] [y0] [x1] [y1] [x2] [y2] [x3] [y3] # First Bezier curve.
        [s1] # Next control point is (x3,y3) + s1*normalize((x3,y3) - (x2,y2)).
        [x2] [y2] [x3] [y3]
        [s1]
        [x2] [y2] [x3] [y3]
        ...
        <close path>
        [s1]
        # Final control point is (x0,y0) + s2*normalize((x0,y0) - (x1,y1)),
        # relative to the first Bezier curve of the path.
        [s2]

    To encode a shape, we may encode multiple loops by concatenating them
    together (each with its own <start> token). Afterwards, we tack on an
    <end> token to indicate no more loops.
    """

    def __init__(self, min=-20.0, max=20.0 + 28, num_bins=256):
        self.min = min
        self.max = max
        self.num_bins = num_bins

        self.num_delimiters = 3
        self.end_token = 0
        self.start_token = 1
        self.close_token = 2

        bin_size = (max - min) / num_bins
        self.bin_size = bin_size
        self.bin_starts = np.linspace(min, max - bin_size, num=num_bins)
        self.bin_ends = np.linspace(min + bin_size, max, num=num_bins)

        small_bin_size = bin_size / num_bins
        self.small_bins = np.linspace(
            small_bin_size + small_bin_size / 2,
            bin_size - small_bin_size / 2,
            num=num_bins,
        )

    @property
    def num_tokens(self) -> int:
        return self.num_delimiters + self.num_bins * 2

    def encode_float(self, x: float) -> Tuple[int, int]:
        """
        Encode a number of a sequence of two tokens.
        """
        bin_idx = int(np.argmax(self.bin_ends >= x))
        residual = x - self.bin_starts[bin_idx]
        closest = int(np.argmin(np.abs(self.small_bins - residual)))
        return (
            bin_idx + self.num_delimiters,
            closest + self.num_delimiters + self.num_bins,
        )

    def decode_float(self, x: Tuple[int, int]) -> float:
        """
        Decode a sequence into a floating-point number.

        Will raise an exception if the token sequence is invalid.
        """
        if x[0] < self.num_delimiters or x[0] >= self.num_delimiters + self.num_bins:
            raise ValueError(f"not a significant token: {x[0]}")
        elif (
            x[1] < self.num_delimiters + self.num_bins
            or x[1] >= self.num_delimiters + self.num_bins * 2
        ):
            raise ValueError(f"not a fractionla token: {x[1]}")
        return (
            self.bin_starts[x[0] - self.num_delimiters]
            + self.small_bins[x[1] - (self.num_delimiters + self.num_bins)]
        )

    def encode_loop(self, loop: List[List[Tuple[float, float]]]) -> List[int]:
        """
        Encode one Bezier loop into a sequence of tokens.
        """
        tokens = [self.start_token]

        def put_delta(known_endpoint, known_control, target_endpoint, target_control):
            tangent = np.array(known_endpoint) - np.array(known_control)
            tangent /= np.sqrt(np.sum(tangent ** 2))
            delta = np.array(target_control) - np.array(target_endpoint)
            delta_dot = np.sum(delta * tangent)
            tokens.extend(self.encode_float(delta_dot))

        for coord in loop[0]:
            for x in coord:
                tokens.extend(self.encode_float(x))
        last_curve = loop[0]
        for i, curve in enumerate(loop[1:]):
            if i + 2 == len(loop):
                tokens.append(self.close_token)
            put_delta(last_curve[3], last_curve[2], curve[0], curve[1])
            last_curve = curve
            if i + 2 < len(loop):
                for coord in curve[2:]:
                    for x in coord:
                        tokens.extend(self.encode_float(x))
            else:
                put_delta(loop[0][0], loop[0][1], curve[3], curve[2])
        return tokens

    def decode_loop(self, tokens: List[int]) -> List[List[Tuple[float, float]]]:
        """
        Decode a bezier loop from a sequence of tokens.

        Will raise an exception if the token sequence is invalid.
        """
        if len(tokens) < 12:
            raise ValueError("unexpected end of sequence")

        tokens = tokens.copy()

        start_tok = tokens.pop(0)
        if start_tok != self.start_token:
            raise ValueError(f"expected start token, got: {start_tok}")

        def read_number_or_close() -> Optional[float]:
            if len(tokens) < 2:
                raise ValueError("unexpected end of sequence")
            t1 = tokens.pop(0)
            if t1 == self.close_token:
                return None
            return self.decode_float((t1, tokens.pop(0)))

        def read_number() -> float:
            if len(tokens) < 2:
                raise ValueError("unexpected end of sequence")
            t1 = tokens.pop(0)
            t2 = tokens.pop(0)
            return self.decode_float((t1, t2))

        def tangent_control_point(last_control, last_end, scale):
            tangent = np.array(last_end) - np.array(last_control)
            tangent /= np.sqrt(np.sum(tangent ** 2))
            return tuple((tangent * scale + np.array(last_end)).tolist())

        curve_0 = []
        for _ in range(4):
            x = read_number()
            y = read_number()
            curve_0.append((x, y))
        loop = [curve_0]
        while True:
            last_curve = loop[-1]

            s1 = read_number_or_close()
            if s1 is None:
                s1 = read_number()
                c1 = tangent_control_point(last_curve[2], last_curve[3], s1)
                s2 = read_number()
                c2 = tangent_control_point(loop[0][1], loop[0][0], s2)
                loop.append([last_curve[-1], c1, c2, loop[0][0]])
                break
            c1 = tangent_control_point(last_curve[2], last_curve[3], s1)
            x2 = read_number()
            y2 = read_number()
            x3 = read_number()
            y3 = read_number()
            loop.append([last_curve[-1], c1, (x2, y2), (x3, y3)])
        if len(tokens) > 0:
            raise ValueError(f"unexpected trailing tokens: {tokens}")
        return loop

    def encode_loops(self, loops: List[List[List[Tuple[float, float]]]]) -> List[int]:
        res = []
        for loop in loops:
            res.extend(self.encode_loop(loop))
        res.append(self.end_token)
        return res

    def decode_loops(
        self, tokens: List[int]
    ) -> Tuple[List[List[List[Tuple[float, float]]]], bool]:
        """
        Decode the loops encoded by the tokens.

        If the tokens were truncated, then the final (incomplete) loop is
        skipped, and the second return argument is False.
        """
        complete = self.end_token in tokens
        if complete:
            tokens = tokens[: tokens.index(self.end_token)]
        loops = []
        while len(tokens):
            if self.start_token in tokens[1:]:
                next_start = tokens[1:].index(self.start_token) + 1
            elif not complete:
                break
            else:
                next_start = len(tokens)
            loops.append(self.decode_loop(tokens[:next_start]))
            tokens = tokens[next_start:]
        return loops, complete

    def token_name(self, t: int) -> str:
        if t < 0 or t > self.num_delimiters + self.num_bins * 2:
            raise ValueError(f"token out of range: {t}")
        if t == self.start_token:
            return "<start>"
        elif t == self.end_token:
            return "<end>"
        elif t == self.close_token:
            return "<close>"
        elif t < self.num_delimiters + self.num_bins:
            i = t - self.num_delimiters
            lower, upper = self.bin_starts[i], self.bin_ends[i]
            return f"<significant bin [{lower:.03f}, {upper:.03f}]>"
        else:
            i = t - self.num_delimiters - self.num_bins
            return f"<fractional bin [{self.small_bins[i]:.03f}]>"

    def format_tokens(self, t: List[int]) -> str:
        return " ".join(self.token_name(x) for x in t)