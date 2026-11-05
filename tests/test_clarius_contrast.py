"""Numerical checks for display contrast without segmentation or synthesis."""
import unittest

import numpy as np

from clarius_contrast import rf_envelope, rf_legacy, rf_window, envelope_tone, mild_clahe


class ContrastTests(unittest.TestCase):
    def test_hilbert_runs_along_depth_per_scanline(self):
        phase=2*np.pi*np.arange(256)*8/256
        rf=np.sin(phase)[:,None]*np.array([[1.,2.,4.]])
        envelope=rf_envelope(rf)
        np.testing.assert_allclose(envelope,np.ones((256,1))*[1,2,4],atol=1e-12)

    def test_known_amplitude_ratios_map_to_expected_db(self):
        envelope=np.array([[1.,10.,100.,1000.]])
        pixels,settings=rf_window(envelope,60,100)
        np.testing.assert_array_equal(pixels,[[0,85,170,255]])
        self.assertEqual(settings['black_relative_db'],-60)

    def test_tone_is_identity_monotonic_and_preserves_black(self):
        gray=np.arange(256,dtype=np.uint8)[None,:]
        np.testing.assert_array_equal(envelope_tone(gray),gray)
        bright=envelope_tone(gray,gamma=.8)
        self.assertEqual(bright[0,0],0)
        self.assertEqual(bright[0,-1],255)
        self.assertTrue((bright>=gray).all())
        self.assertTrue((np.diff(bright.astype(int))>=0).all())

    def test_spatial_grid_and_input_are_unchanged(self):
        gray=np.arange(120,dtype=np.uint8).reshape(12,10);copy=gray.copy()
        mapped=envelope_tone(gray,5,210,.9)
        np.testing.assert_array_equal(gray,copy)
        self.assertEqual(mapped.shape,gray.shape)
        self.assertEqual(mapped.dtype,np.uint8)
        self.assertEqual(mapped[0,0],0)

    def test_zero_signal_and_constant_image(self):
        zeros=np.zeros((64,5))
        for pixels in (rf_envelope(zeros),rf_legacy(zeros),rf_window(zeros)[0]):
            self.assertFalse(np.any(pixels))
            self.assertTrue(np.isfinite(pixels).all())
        constant=np.full((10,20),75,dtype=np.uint8)
        np.testing.assert_array_equal(mild_clahe(constant),constant)

    def test_reject_invalid_inputs(self):
        for data in (np.ones(5),np.empty((0,2)),np.array([[np.nan]])):
            with self.assertRaises(ValueError):rf_envelope(data)
        for kwargs in ({'dynamic_range_db':0},{'gamma':0},{'white_percentile':101}):
            with self.assertRaises(ValueError):rf_window(np.ones((2,2)),**kwargs)
        with self.assertRaises(ValueError):rf_window(-np.ones((2,2)))
        with self.assertRaises(ValueError):envelope_tone(np.ones((2,2)))
        with self.assertRaises(ValueError):envelope_tone(np.ones((2,2),np.uint8),20,10)


if __name__=='__main__':unittest.main()
