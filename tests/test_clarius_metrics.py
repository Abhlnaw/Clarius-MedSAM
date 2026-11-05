import unittest
import numpy as np
from clarius_metrics import evaluate_mask


class ReferenceMetricTests(unittest.TestCase):
    def test_identity_and_empty_identity(self):
        mask=np.zeros((40,60),bool);mask[10:30,15:45]=True
        for reference in [mask,np.zeros_like(mask)]:
            result=evaluate_mask(reference,reference)
            for key in ['ssim','roi_mean_ssim','dice','iou']:self.assertAlmostEqual(result[key],1.)

    def test_constant_opposites_match_closed_form_ssim(self):
        result=evaluate_mask(np.zeros((30,40),bool),np.ones((30,40),bool))
        self.assertAlmostEqual(result['ssim'],.01**2/(1+.01**2),places=10)
        self.assertEqual(result['dice'],0);self.assertEqual(result['iou'],0)

    def test_known_shift_overlap_and_ssim_symmetry(self):
        first=np.zeros((40,60),bool);first[15:25,20:30]=True
        second=np.zeros_like(first);second[15:25,21:31]=True
        result=evaluate_mask(first,second)
        self.assertAlmostEqual(result['dice'],.9);self.assertAlmostEqual(result['iou'],9/11)
        self.assertAlmostEqual(result['ssim'],evaluate_mask(second,first)['ssim'])
        self.assertLess(result['ssim'],1)

    def test_shared_background_inflates_full_ssim_but_not_common_roi_mean(self):
        first=np.zeros((40,60),bool);first[12:28,18:40]=True
        second=first.copy();second[12:28,35:40]=False
        domain=np.zeros_like(first);domain[8:32,12:48]=True
        original=evaluate_mask(first,second,domain)
        padded=evaluate_mask(np.pad(first,30),np.pad(second,30),np.pad(domain,30))
        self.assertGreater(padded['ssim'],original['ssim'])
        self.assertAlmostEqual(padded['roi_mean_ssim'],original['roi_mean_ssim'],places=10)

    def test_invalid_arrays_and_hiding_foreground_are_rejected(self):
        mask=np.ones((20,20),bool)
        for prediction in [np.full((20,20),2),np.ones((20,19)),np.full((20,20),np.nan)]:
            with self.assertRaises(ValueError):evaluate_mask(prediction,mask)
        with self.assertRaises(ValueError):evaluate_mask(mask,mask,np.zeros_like(mask))
        with self.assertRaises(ValueError):evaluate_mask(np.ones((5,5)),np.ones((5,5)))


if __name__=='__main__':unittest.main()
