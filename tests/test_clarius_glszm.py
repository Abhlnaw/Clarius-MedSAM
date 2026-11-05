"""Independent graph traversal, published example, and dense local references."""
import unittest
import numpy as np

from clarius_glszm import glszm_matrix, glszm_features, local_glszm_features, LOCAL_PROPERTIES
from clarius_glcm import analysis_windows


def graph_reference(gray, mask, levels):
    quantized = np.minimum((np.where(mask, gray, 0) * levels).astype(int), levels - 1)
    visited = np.zeros(mask.shape, bool)
    zones = []
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        queue = [(y, x)]; visited[y, x] = True; size = 0
        while queue:
            cy, cx = queue.pop(); size += 1
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                        if mask[ny, nx] and not visited[ny, nx] and quantized[ny, nx] == quantized[y, x]:
                            visited[ny, nx] = True; queue.append((ny, nx))
        zones.append((quantized[y, x], size))
    matrix = np.zeros((levels, max(s for _, s in zones)), int)
    for g, s in zones:matrix[g, s - 1] += 1
    return matrix


class GLSZMTests(unittest.TestCase):
    def test_published_pyradiomics_matrix(self):
        image = np.array([[5,2,5,4,4], [3,3,3,1,3], [2,1,1,1,3], [4,2,2,2,3], [3,5,3,3,2]])
        actual = glszm_matrix((image - .5) / 5, np.ones(image.shape, bool), levels=5)
        expected = [[0,0,0,1,0], [1,0,0,0,1], [1,0,1,0,1], [1,1,0,0,0], [3,0,0,0,0]]
        np.testing.assert_array_equal(actual, expected)

    def test_random_masked_matrices_match_independent_graph_traversal(self):
        rng = np.random.default_rng(123)
        for _ in range(15):
            image = rng.random((9, 11)); mask = rng.random(image.shape) > .3
            np.testing.assert_array_equal(glszm_matrix(image, mask, 4), graph_reference(image, mask, 4))

    def test_constant_roi_known_features(self):
        features = glszm_features(np.full((3, 4), .5), np.ones((3, 4), bool), levels=4)
        self.assertEqual(len(features), 16)
        self.assertAlmostEqual(features['small_area_emphasis'], 1/144)
        self.assertAlmostEqual(features['large_area_emphasis'], 144)
        self.assertEqual(features['gray_level_non_uniformity_normalized'], 1)
        self.assertEqual(features['size_zone_non_uniformity_normalized'], 1)
        self.assertEqual(features['zone_percentage'], 1/12)
        self.assertEqual(features['gray_level_variance'], 0)
        self.assertEqual(features['zone_variance'], 0)
        self.assertEqual(features['zone_entropy'], 0)
        self.assertAlmostEqual(features['low_gray_level_zone_emphasis'], 1/9)
        self.assertEqual(features['high_gray_level_zone_emphasis'], 9)
        self.assertAlmostEqual(features['small_area_low_gray_level_emphasis'], 1/1296)
        self.assertAlmostEqual(features['small_area_high_gray_level_emphasis'], 9/144)
        self.assertEqual(features['large_area_low_gray_level_emphasis'], 16)
        self.assertEqual(features['large_area_high_gray_level_emphasis'], 1296)

    def test_nonconstant_feature_values_from_published_matrix(self):
        image=np.array([[5,2,5,4,4],[3,3,3,1,3],[2,1,1,1,3],[4,2,2,2,3],[3,5,3,3,2]])
        f=glszm_features((image-.5)/5,np.ones(image.shape,bool),levels=5)
        self.assertAlmostEqual(f['gray_level_non_uniformity'],27/11)
        self.assertAlmostEqual(f['gray_level_non_uniformity_normalized'],27/121)
        self.assertAlmostEqual(f['size_zone_non_uniformity'],43/11)
        self.assertAlmostEqual(f['size_zone_non_uniformity_normalized'],43/121)
        self.assertAlmostEqual(f['zone_percentage'],11/25)
        self.assertAlmostEqual(f['gray_level_variance'],204/121)
        self.assertAlmostEqual(f['zone_variance'],310/121)
        self.assertAlmostEqual(f['zone_entropy'],-8/11*np.log2(1/11)-3/11*np.log2(3/11))

    def test_diagonal_pixels_are_connected_and_mask_splits_zones(self):
        image = np.eye(4)
        mask = image.astype(bool)
        matrix = glszm_matrix(image, mask, 2)
        self.assertEqual(matrix[1, 3], 1)
        mask[2, 2] = False
        matrix = glszm_matrix(image, mask, 2)
        self.assertEqual(matrix[1, 1], 1)
        self.assertEqual(matrix[1, 0], 1)

    def test_global_rotation_invariance_and_conservation(self):
        rng = np.random.default_rng(2)
        image = rng.random((8, 11)); mask = rng.random(image.shape) > .2
        matrix = glszm_matrix(image, mask, 4)
        np.testing.assert_array_equal(matrix, glszm_matrix(np.rot90(image), np.rot90(mask), 4))
        self.assertEqual(int((matrix * np.arange(1, matrix.shape[1]+1)).sum()), int(mask.sum()))

    def test_dense_local_maps_match_independent_clipped_window_features(self):
        rng = np.random.default_rng(7)
        image = rng.random((7, 9)); mask = np.ones(image.shape, bool); mask[2:5, 4] = False
        actual, support = local_glszm_features(image, mask, (5, 5), levels=4, step=(1, 1))
        np.testing.assert_array_equal(support, mask)
        for y,x in zip(*np.nonzero(mask)):
            for i,(h,w) in enumerate(analysis_windows((5,5))):
                crop = (slice(max(0,y-h//2),min(7,y+h//2+1)),slice(max(0,x-w//2),min(9,x+w//2+1)))
                # Reference counts from independent graph traversal; deriving
                # SAE/LAE directly catches using global instead of clipped zones.
                matrix = graph_reference(image[crop],mask[crop],4)
                size = np.arange(1,matrix.shape[1]+1)
                self.assertAlmostEqual(actual[y,x,i*8], (matrix/size**2).sum()/matrix.sum(), places=6)
                self.assertAlmostEqual(actual[y,x,i*8+1], (matrix*size**2).sum()/matrix.sum(), places=5)
        self.assertFalse(actual[~mask].any())

    def test_lattice_nodes_equal_dense_and_outside_values_do_not_leak(self):
        rng=np.random.default_rng(8);image=rng.random((13,11));mask=np.ones(image.shape,bool);mask[5:8,5]=False
        dense,_=local_glszm_features(image,mask,(7,5),levels=4,step=(1,1))
        sparse,support=local_glszm_features(image,mask,(7,5),levels=4,step=(4,2))
        for y in [0,4,8,12]:
            np.testing.assert_allclose(sparse[y,::2],dense[y,::2],rtol=1e-6)
        image[~mask]=np.nan
        other,_=local_glszm_features(image,mask,(7,5),levels=4,step=(4,2))
        np.testing.assert_array_equal(other,sparse)
        self.assertTrue(np.isfinite(sparse).all())
        np.testing.assert_array_equal(support,mask)

    def test_single_pixel_and_invalid_inputs(self):
        maps,support=local_glszm_features(np.array([[1.]]),np.array([[True]]),(3,3),step=(1,1))
        self.assertTrue(support[0,0]);self.assertTrue(np.isfinite(maps).all())
        for kwargs in [dict(levels=1),dict(step=(0,1)),dict(window=(2,3))]:
            options=dict(window=(3,3));options.update(kwargs)
            with self.assertRaises(ValueError):local_glszm_features(np.ones((3,3)),np.ones((3,3),bool),**options)
        with self.assertRaises(ValueError):glszm_matrix(np.array([[np.nan]]),np.array([[True]]))
        with self.assertRaises(ValueError):glszm_matrix(np.ones((3,3)),np.zeros((3,3),bool))


if __name__ == '__main__':unittest.main()
