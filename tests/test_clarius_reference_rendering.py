import importlib.util
from pathlib import Path
import unittest
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from propagate_reference_contours import advect

spec=importlib.util.spec_from_file_location('reference_export',Path(__file__).resolve().parents[1]/'tools/create_reference_masks.py')
export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)


class ReferenceRenderingTests(unittest.TestCase):
    def test_review_is_rgb_ultrasound_at_example_proportions_and_labels_stay_native(self):
        image=np.full((1536,192),128,np.uint8)
        mask=np.zeros(image.shape,bool);mask[200:800,20:170]=True
        original=mask.copy()
        preview=export.render_reference_overlay(image,mask,longest_edge=1418)
        self.assertEqual(preview.mode,'RGB');self.assertEqual(preview.size,(1418,853))
        values=np.asarray(preview)
        np.testing.assert_array_equal(values[0,0],[128,128,128])
        self.assertTrue(((values[...,0]>200)&(values[...,1]<100)).any())
        np.testing.assert_array_equal(mask,original)

    def test_authored_vertices_and_holes_rasterize_in_correct_axes(self):
        shape=(100,20);display=(1000,600)
        outer=export.rasterize([[100,120],[900,120],[900,480],[100,480]],shape,display)
        hole=export.rasterize([[400,240],[600,240],[600,360],[400,360]],shape,display)
        result=outer&~hole
        self.assertTrue(result[30,5]);self.assertFalse(result[50,10]);self.assertFalse(result[10,5])

    def test_closed_spline_preserves_authored_vertices(self):
        vertices=np.array([[10,10],[90,10],[90,90],[10,90]],float)
        curve=export.smooth_contour(vertices,samples=400)
        np.testing.assert_allclose(curve[::100],vertices,atol=1e-10)

    def test_forward_flow_moves_contours_in_correct_display_coordinates(self):
        path=np.array([[30.,30.],[60.,30.],[60.,60.],[30.,60.]])
        flow=np.zeros((2,100,200));flow[0]=-2;flow[1]=4
        actual=advect([path],flow,(1000,600))[0]
        np.testing.assert_allclose(actual,path+np.array([20,-12]))
        np.testing.assert_array_equal(path,[[30,30],[60,30],[60,60],[30,60]])


if __name__=='__main__':unittest.main()
