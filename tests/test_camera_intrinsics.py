import copy
import math
import unittest

from camera_intrinsics import (load_rectified_intrinsics, validate_intrinsics, usd_projection_parameters,
                               centered_render_plan, RectifiedOutputMapper, verify_native_projection)
from camera_profile import resolve_camera_profile


class RectifiedCameraIntrinsics(unittest.TestCase):
    def test_uses_supplied_output_K_not_scaled_raw_HD_or_VGA(self):
        c = load_rectified_intrinsics()
        self.assertEqual(c['K_px'], [[354.2246895731583, 0., 325.25666235264373],
                                   [0., 354.2246895731583, 180.2094012432459], [0., 0., 1.]])
        self.assertEqual(c['distortion_coefficients'], [])
        self.assertFalse(c['extrinsics_verified'])
        self.assertEqual(len(c['calibration_sha256']), 64)

    def test_rejects_wrong_size_instead_of_silently_reusing_K(self):
        for size in ((1280, 720), (640, 480), (320, 180)):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, 'explicit resize/crop'):
                validate_intrinsics(load_rectified_intrinsics(), *size)

    def test_raw_distortion_cannot_be_applied_to_rectified_output(self):
        c = load_rectified_intrinsics()
        c['distortion_model'] = 'opencv_fisheye'
        c['distortion_coefficients'] = [-.0323137, .00481129, .0257629, -.0213337]
        with self.assertRaisesRegex(ValueError, 'raw fisheye'):
            validate_intrinsics(c, 640, 360)

    def test_bad_matrices_are_rejected_before_rendering(self):
        for row, col, value in ((0, 0, 0), (1, 1, -1), (0, 1, .1), (2, 2, 2), (0, 2, math.nan)):
            c = copy.deepcopy(load_rectified_intrinsics()); c['K_px'][row][col] = value
            with self.subTest(row=row, col=col, value=value), self.assertRaises(ValueError):
                validate_intrinsics(c, 640, 360)

    def test_nominal_mm_focal_does_not_change_pixel_intrinsics(self):
        c = load_rectified_intrinsics()
        for focal in (11.5, 14., 30.):
            p = usd_projection_parameters(c, 640, 360, focal)
            self.assertAlmostEqual(640*focal/p['horizontal_aperture_mm'], c['K_px'][0][0])
            self.assertAlmostEqual(360*focal/p['vertical_aperture_mm'], c['K_px'][1][1])

    def test_scene_profile_keeps_mount_and_allows_explicit_historical_projection(self):
        calibrated = resolve_camera_profile(pitch=28.)
        historical = resolve_camera_profile(pitch=28., intrinsics_config='virtual')
        self.assertEqual(calibrated['mount_position_m'], historical['mount_position_m'])
        self.assertEqual(calibrated['pitch_down_deg'], historical['pitch_down_deg'])
        self.assertIsNotNone(calibrated['intrinsics'])
        self.assertIsNone(historical['intrinsics'])

    def test_guard_band_covers_all_output_pixels_without_black_border(self):
        import numpy as np
        c = load_rectified_intrinsics(); plan = centered_render_plan(c,640,360)
        mapper = RectifiedOutputMapper(plan)
        image = np.full((plan['render_height'],plan['render_width'],3),127,np.uint8)
        output = mapper.apply(image)
        self.assertEqual(output.shape,(360,640,3))
        self.assertTrue(np.all(output==127))
        k = c['K_px']
        self.assertAlmostEqual(plan['render_K_px'][0][2]-plan['output_to_render_offset_px'][0],k[0][2])
        self.assertAlmostEqual(plan['render_K_px'][1][2]-plan['output_to_render_offset_px'][1],k[1][2])

    def test_depth_remap_does_not_interpolate_invalid_or_millimeter_values(self):
        import numpy as np
        plan = centered_render_plan(load_rectified_intrinsics(),640,360)
        image = np.zeros((plan['render_height'],plan['render_width']),np.uint16)
        image[:,::2] = 2000
        output = RectifiedOutputMapper(plan).apply(image,depth=True)
        self.assertEqual(output.dtype,np.uint16)
        self.assertEqual(set(np.unique(output)),{0,2000})

    def test_actual_renderer_projection_must_match_output_K(self):
        import numpy as np
        c = load_rectified_intrinsics(); plan = centered_render_plan(c,640,360)
        projection = np.eye(4)
        projection[0,0] = 2*c['K_px'][0][0]/plan['render_width']
        projection[1,1] = 2*c['K_px'][1][1]/plan['render_height']
        info = {'intrinsics': c, 'render_plan': plan}
        result = verify_native_projection({'cameraProjection': projection.flatten()}, info)
        self.assertTrue(result['verified'])
        np.testing.assert_allclose(result['output_K_px'], c['K_px'])
        projection[0,0] *= .99
        with self.assertRaises(AssertionError):
            verify_native_projection({'cameraProjection': projection.flatten()}, info)
        projection[2,0] = .02
        with self.assertRaisesRegex(ValueError,'centered'):
            verify_native_projection({'cameraProjection': projection.flatten()}, info)
