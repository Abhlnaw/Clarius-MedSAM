import unittest
import numpy as np
import torch
from clarius_localizer import INPUT_HW, LiverBoxNet, decode_box, grouped_folds, mask_box, pad_box


class LocalizerTests(unittest.TestCase):
    def test_mask_box_exclusive_edges_and_hole(self):
        mask=np.zeros((100,200),bool);mask[10:80,20:180]=True;mask[30:50,60:100]=False
        np.testing.assert_allclose(mask_box(mask),[.1,.1,.9,.8])

    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError):mask_box(np.zeros((10,10),bool))

    def test_decoded_boxes_have_ordered_edges_and_gradients(self):
        logits=torch.tensor([[-5.,1.,2.,-2.],[2.,-1.,-1.,3.]],requires_grad=True)
        boxes=decode_box(logits)
        self.assertTrue(bool((boxes[:,2:]>boxes[:,:2]).all()))
        self.assertTrue(bool(((boxes>=0)&(boxes<=1)).all()))
        boxes.sum().backward();self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_group_split_and_single_outer_prediction(self):
        cases=[dict(id=f'{g}_{i}',exam=str(g)) for g in range(13) for i in range(3)]
        folds=grouped_folds(cases)
        tested=[]
        for split in folds:
            groups={role:{cases[i]['exam'] for i in indices} for role,indices in split.items()}
            self.assertFalse(groups['train']&groups['test'])
            self.assertFalse(groups['train']&groups['validation'])
            self.assertFalse(groups['test']&groups['validation'])
            tested+=split['test']
        self.assertEqual(sorted(tested),list(range(len(cases))))
        self.assertEqual(folds,grouped_folds(cases))

    def test_padding_is_image_fraction_and_clipped(self):
        np.testing.assert_allclose(pad_box([.01,.1,.95,.8],.03),[0,.07,.98,.83])

    def test_network_output_and_mean_initialization(self):
        model=LiverBoxNet().eval();model.initialize_box([.05,.1,.95,.7])
        with torch.inference_mode():box=model(torch.zeros(1,1,*INPUT_HW))[0].numpy()
        np.testing.assert_allclose(box,[.05,.1,.95,.7],atol=1e-6)


if __name__=='__main__':unittest.main()
