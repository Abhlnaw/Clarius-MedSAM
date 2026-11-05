"""Byte-preservation tests for the decompression-only MATLAB preparation stage."""
import io
import struct
import unittest

from tools.prepare_clarius_frames import select_frame


def recording(stamps, rid=1, size=1):
    frames = [(stamp, bytes([10+i, 255-i, 0, 64])) for i,stamp in enumerate(stamps)]
    return (struct.pack('<5I',rid,len(frames),2,2,size) +
            b''.join(struct.pack('<Q',stamp)+payload for stamp,payload in frames)), frames


class FramePreparationTests(unittest.TestCase):
    def test_selected_samples_are_bit_exact(self):
        raw, frames = recording([123, 456, 789])
        selected, metadata = select_frame(io.BytesIO(raw),1,450)
        self.assertEqual(selected,struct.pack('<5I',1,1,2,2,1)+struct.pack('<Q',456)+frames[1][1])
        self.assertEqual(metadata['selected_frame'],2)
        self.assertEqual(metadata['original_frames'],3)

    def test_timestamp_precision_above_double_integer_range(self):
        stamp = 2**53+123
        raw,_ = recording([stamp])
        selected,metadata = select_frame(io.BytesIO(raw),1,stamp)
        self.assertEqual(metadata['timestamp_raw'],str(stamp))
        self.assertEqual(struct.unpack('<Q',selected[20:28])[0],stamp)

    def test_first_frame_and_nearest_endpoints(self):
        raw,_ = recording([100,200,300])
        for target,expected in [(None,1),(0,1),(190,2),(900,3)]:
            _,metadata = select_frame(io.BytesIO(raw),1,target)
            self.assertEqual(metadata['selected_frame'],expected)

    def test_reject_truncated_or_wrong_format(self):
        raw,_ = recording([123])
        for bad in [raw[:10],raw[:24],raw[:-1],struct.pack('<5I',1,0,2,2,1)]:
            with self.assertRaises(ValueError):select_frame(io.BytesIO(bad),1)
        with self.assertRaises(ValueError):select_frame(io.BytesIO(raw),2)

    def test_reject_nonmonotonic_timestamps(self):
        raw,_ = recording([100,99])
        with self.assertRaisesRegex(ValueError,'Non-monotonic'):
            select_frame(io.BytesIO(raw),1,200)


if __name__ == '__main__':
    unittest.main()
