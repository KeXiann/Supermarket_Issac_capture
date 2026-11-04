"""Exact saved-image intrinsics; raw stereo distortion is reference material only."""
import hashlib
import json
import math
from pathlib import Path


DEFAULT_CALIBRATION = Path(__file__).resolve().parents[1] / 'assets/upstream/7e5fd9d0ae2f/configs/collection/camera_stereo.json'


def load_rectified_intrinsics(path=DEFAULT_CALIBRATION):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    document = json.loads(raw)
    if 'left' in document and document['left'].get('image_type') == 'rectified_pinhole':
        # Read the upstream document directly: no second copy of K/raw stereo calibration.
        eye = document['left']
        output = {'width': eye['width'], 'height': eye['height'], 'K_px': eye['K'],
                  'rectified': not any(eye['distortion']), 'ready_to_apply_to_simulation': True,
                  'optical_frame': document['optical_frame']}
        profile, source = eye['profile'], {'snapshot_path': str(path),
            'raw_calibration_source': document['raw_calibration']['source']}
        metric_verified = document['calibration_reference']['metric_accuracy_verified']
    else:
        output = document['saved_rectified_output']
        profile, source = document['profile'], document['source']
        metric_verified = document['depth']['metric_accuracy_verified']
    if not output.get('rectified') or not output.get('ready_to_apply_to_simulation'):
        raise ValueError('Calibration must explicitly provide the saved rectified image K')
    result = {'profile': profile, 'width': output['width'], 'height': output['height'],
              'K_px': output.get('K_px'), 'distortion_model': 'none_rectified', 'distortion_coefficients': [],
              'reference_optical_frame': output['optical_frame'],
              'calibration_path': str(path), 'calibration_sha256': hashlib.sha256(raw).hexdigest(),
              'source': source, 'extrinsics_verified': False,
              'metric_depth_accuracy_verified': metric_verified,
              'baseline_m': document.get('baseline_m'),
              'raw_calibration': document.get('raw_calibration'),
              'right': document.get('right')}
    validate_intrinsics(result, result['width'], result['height'])
    return result


def validate_intrinsics(intrinsics, width, height):
    if (width, height) != (intrinsics['width'], intrinsics['height']):
        raise ValueError(f"Rectified K is for {intrinsics['width']}x{intrinsics['height']}, not {width}x{height}; explicit resize/crop calibration required")
    k = intrinsics.get('K_px')
    if not isinstance(k, (list, tuple)) or len(k) != 3 or any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in k):
        raise ValueError('K must be a finite zero-skew 3x3 pinhole matrix')
    if (not all(math.isfinite(float(v)) for row in k for v in row)
            or k[0][0] <= 0 or k[1][1] <= 0 or k[0][1] != 0 or k[1][0] != 0
            or list(k[2]) != [0, 0, 1] or width < 16 or height < 16):
        raise ValueError('K must be a finite zero-skew 3x3 pinhole matrix with positive focal lengths')
    if intrinsics.get('distortion_model') != 'none_rectified' or intrinsics.get('distortion_coefficients') != []:
        raise ValueError('Saved rectified images must not reuse raw fisheye distortion')
    return k


def usd_projection_parameters(intrinsics, width, height, focal_length_mm):
    """USD film offsets preserve the same OpenCV right/down pixel convention."""
    k = validate_intrinsics(intrinsics, width, height)
    if not math.isfinite(focal_length_mm) or focal_length_mm <= 0:
        raise ValueError('Nominal USD focal length must be positive and finite')
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    return {'horizontal_aperture_mm': width * focal_length_mm / fx,
            'vertical_aperture_mm': height * focal_length_mm / fy,
            'horizontal_aperture_offset_mm': (width / 2 - cx) * focal_length_mm / fx,
            'vertical_aperture_offset_mm': (cy - height / 2) * focal_length_mm / fy}


def centered_render_plan(intrinsics, width, height):
    """Guard-band rendering plus a pixel-center remap preserves off-center K.

    The installed RTX path ignores off-axis USD/lens principal points in an
    actual zero-distortion render. Do not rely on those metadata values alone.
    """
    k = validate_intrinsics(intrinsics, width, height)
    if not math.isclose(k[0][0], k[1][1], rel_tol=0, abs_tol=1e-9):
        raise ValueError('This renderer path requires square pixels; non-square fx/fy needs separate validation')
    rw = width + 2*(math.ceil(abs(k[0][2]-width/2))+1)
    rh = height + 2*(math.ceil(abs(k[1][2]-height/2))+1)
    return {'method': 'centered_guard_band_render_then_subpixel_remap',
            'render_width': rw, 'render_height': rh, 'output_width': width, 'output_height': height,
            'render_K_px': [[k[0][0],0.,(rw-1)/2], [0.,k[1][1],(rh-1)/2], [0.,0.,1.]],
            'output_to_render_offset_px': [(rw-1)/2-k[0][2], (rh-1)/2-k[1][2]],
            'pixel_convention': 'OpenCV integer pixel centers',
            'raw_distortion_applied': False}


def verify_native_projection(camera_parameters, camera_info):
    """Read RTX's actual projection, then verify the remapped output pixel K."""
    import numpy as np
    plan = camera_info['render_plan']
    projection = np.asarray(camera_parameters['cameraProjection'], dtype=float).reshape(4, 4)
    if not np.isfinite(projection).all() or not np.allclose(projection[2, :2], 0., atol=1e-7):
        raise ValueError('RTX must provide a finite centered native projection')
    size = np.array([plan['render_width'], plan['render_height']], dtype=float)
    measured = np.eye(3)
    measured[0, 0], measured[1, 1] = np.abs(projection.diagonal()[:2]) * size / 2
    measured[:2, 2] = (size - 1) / 2
    np.testing.assert_allclose(measured, plan['render_K_px'], rtol=1e-5, atol=1e-4,
                               err_msg='RTX native projection differs from the requested K')
    native = measured.copy()
    measured[:2, 2] -= plan['output_to_render_offset_px']
    expected = np.asarray(camera_info['intrinsics']['K_px'], dtype=float)
    np.testing.assert_allclose(measured, expected, rtol=1e-5, atol=1e-4,
                               err_msg='Actual rectified output projection differs from upstream K')
    return {'verified': True, 'source': 'RTX camera_params cameraProjection after warmup',
            'native_render_K_px': native.tolist(), 'output_K_px': measured.tolist(),
            'max_K_error_px': float(np.max(np.abs(measured - expected))),
            'robot_extrinsics_verified': False}


class RectifiedOutputMapper:
    def __init__(self, plan):
        import numpy as np
        self.plan = plan
        u,v = np.meshgrid(np.arange(plan['output_width'],dtype=np.float32),
                          np.arange(plan['output_height'],dtype=np.float32))
        dx,dy = plan['output_to_render_offset_px']
        self.map_x, self.map_y = u+dx, v+dy
        if (self.map_x.min()<0 or self.map_y.min()<0 or self.map_x.max()>plan['render_width']-1
                or self.map_y.max()>plan['render_height']-1):
            raise ValueError('Guard-band render does not cover the requested calibrated image')

    def apply(self, image, *, depth=False):
        import cv2
        if image.shape[:2] != (self.plan['render_height'],self.plan['render_width']):
            raise ValueError('Rendered image dimensions differ from calibrated remap plan')
        return cv2.remap(image,self.map_x,self.map_y,cv2.INTER_NEAREST if depth else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0)
