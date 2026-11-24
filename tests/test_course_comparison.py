"""Checks for shared corruptions and bounded, variable-width view-stream state."""

import unittest
import numpy as np
from data.course_protocols import noisy_observations,noisy_stream_view,mask_hash
from data.missing import make_mask
from models.view_stream import ViewEncoder,ViewBaseline


class CourseProtocolTests(unittest.TestCase):
    def test_hidden_values_do_not_influence_observed_noise(self):
        rng=np.random.default_rng(7)
        views=[rng.normal(size=(30,4)).astype(np.float32) for _ in range(2)]
        mask=make_mask(30,2,0.6,1)
        poison=[x.copy() for x in views]
        for j,x in enumerate(poison): x[~mask[:,j]]=np.nan
        a=noisy_observations(views,mask,1,1.0)
        b=noisy_observations(poison,mask,1,1.0)
        for j in range(2): np.testing.assert_array_equal(a[j][mask[:,j]],b[j][mask[:,j]])
        self.assertEqual(mask_hash(mask),mask_hash(mask.copy()))

    def test_drift_has_no_effect_before_its_declared_view(self):
        x=np.random.default_rng(7).normal(size=(30,4)).astype(np.float32)
        np.testing.assert_array_equal(x,noisy_stream_view(x,1,2,3,'view_noise'))
        self.assertFalse(np.array_equal(x,noisy_stream_view(x,1,3,3,'view_noise')))
        np.testing.assert_array_equal(noisy_stream_view(x,1,3,3,'view_noise'),noisy_stream_view(x,1,3,3,'view_noise'))

    def test_encoder_supports_changing_view_width_without_retaining_old_views(self):
        encoder=ViewEncoder(16,1)
        rng=np.random.default_rng(5)
        for width in [24,76,6]:
            result=encoder.update(rng.normal(size=(30,width)).astype(np.float32))
            self.assertEqual(result.shape,(30,16))
            self.assertEqual(encoder.reference.shape,(30,16))
        self.assertEqual(set(encoder.__dict__),{'components','seed','forgetting','reference'})

    def test_sliding_view_history_is_bounded(self):
        model=ViewBaseline('sliding_window',3,1,{'n_init':2,'max_iter':20,'tol':1e-4},window=2)
        rng=np.random.default_rng(2)
        for width in [24,15,6,8]: model.update(rng.normal(size=(30,width)).astype(np.float32))
        self.assertEqual(len(model.history),2)


if __name__=='__main__': unittest.main()
