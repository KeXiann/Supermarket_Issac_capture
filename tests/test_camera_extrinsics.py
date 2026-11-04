import copy
import json
from pathlib import Path
import unittest

import numpy as np

from camera_extrinsics import load_extrinsics,validate_extrinsics,with_operator_overrides
from camera_profile import resolve_camera_profile


class EstimatedCameraMount(unittest.TestCase):
    def test_matches_pinned_unscaled_repository_mount_not_modified_local_worktree(self):
        c=load_extrinsics();source=Path(__file__).resolve().parents[1]/c['source']['snapshot_path']
        prior=json.loads(source.read_text())
        self.assertEqual(c['T_parent_camera_cv'],prior['left']['mount']['body_T_camera_cv'])
        self.assertEqual(prior['baseline_m'],.06555722207234115)
        self.assertAlmostEqual(c['pitch_down_deg'],41.10635710810979)
        self.assertFalse(c['calibrated'])
        self.assertTrue(c['do_not_use_as_ground_truth_hand_eye_calibration'])

    def test_camera_axes_point_right_down_and_forward_down_in_robot_frame(self):
        m=validate_extrinsics(load_extrinsics())
        from upstream_camera import load_contract
        source=np.asarray(load_contract()['stereo']['left']['mount']['body_T_camera_cv'])
        np.testing.assert_array_equal(m,source)
        self.assertLess(m[1,0],-.99999)
        self.assertLess(m[2,1],0)
        self.assertGreater(m[0,2],0);self.assertLess(m[2,2],0)
        np.testing.assert_allclose(m[:3,3],[.10292323492068217,.022820609792258507,.40395833708452433],atol=1e-12)

    def test_bad_or_wrong_frame_transform_cannot_become_calibrated(self):
        for kind in ('reflection','scale','nonfinite','bottomrow','pelvis','calibrated'):
            c=copy.deepcopy(load_extrinsics())
            if kind=='reflection': c['T_parent_camera_cv'][0]=[-v for v in c['T_parent_camera_cv'][0]]
            elif kind=='scale':c['T_parent_camera_cv'][0][2]*=2
            elif kind=='nonfinite':c['T_parent_camera_cv'][0][3]=float('nan')
            elif kind=='bottomrow':c['T_parent_camera_cv'][3][3]=0
            elif kind=='pelvis':c['parent_frame']='pelvis'
            else:c['calibrated']=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_extrinsics(c)

    def test_new_scene_profiles_share_physical_mount_and_keep_old_scene_configs_available(self):
        root=Path(__file__).resolve().parents[1]
        for scene in (None,root/'assets/scenes/pico_supermarket_basket_31f4a73.usda'):
            current=resolve_camera_profile(scene)
            self.assertEqual(current['extrinsics']['status'],'estimated_not_calibrated')
            self.assertEqual(current['mount_position_m'],load_extrinsics()['translation_m'])
            self.assertAlmostEqual(current['pitch_down_deg'],41.10635710810979)
        self.assertIsNone(resolve_camera_profile(extrinsics_config='virtual')['extrinsics'])

    def test_operator_changes_are_separately_labelled_and_keep_source_estimate_unchanged(self):
        original=load_extrinsics()
        modified=with_operator_overrides(original,mount=[.11,.02,.4],pitch=35.)
        self.assertEqual(modified['status'],'operator_override_not_calibrated')
        np.testing.assert_allclose(validate_extrinsics(modified)[:3,3],[.11,.02,.4])
        self.assertAlmostEqual(modified['pitch_down_deg'],35.)
        self.assertEqual(original['status'],'estimated_not_calibrated')
        self.assertEqual(original['T_parent_camera_cv'],load_extrinsics()['T_parent_camera_cv'])

    def test_mount_source_uses_current_commit(self):
        from upstream_contract import COMMIT
        self.assertEqual(load_extrinsics()['source']['commit'], COMMIT)
