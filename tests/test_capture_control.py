from project_paths import SONIC_ROOT
"""Recording gestures cannot collide with calibration; bounds gate all rows."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(SONIC_ROOT))
from gear_sonic.utils.teleop.capture_control import RecordingChords, RecordingControl
from recording_window import recording_window, recording_mask
from capture_session import initialize, _finalize_artifacts


class CaptureControlTests(unittest.TestCase):
    def test_recording_chords_do_not_change_a_simultaneous_trigger_grasp(self):
        from gear_sonic.utils.teleop.revo2_hand_control import controller_hand_command
        for trigger in (0., .35, 1.):
            for side in (True, False):
                expected = controller_hand_command(trigger, 0., side)
                c = RecordingChords()
                self.assertIsNone(c.update(False,False,False,False,1.,now=1.))
                np.testing.assert_array_equal(controller_hand_command(trigger,1.,side),expected)
                self.assertEqual(c.update(True,False,False,False,1.,now=1.3),'toggle')
                np.testing.assert_array_equal(controller_hand_command(trigger,1.,side),expected)
                c.update(False,False,False,False,1.,now=2.)
                c.update(False,False,False,False,1.,now=2.21)
                self.assertEqual(c.update(True,False,False,False,1.,now=2.22),'toggle')
                np.testing.assert_array_equal(controller_hand_command(trigger,1.,side),expected)

    def test_hold_and_release_debounce(self):
        c = RecordingChords()
        self.assertEqual(c.update(True, False, False, False, .8, now=1), 'toggle')
        self.assertIsNone(c.update(True, False, False, False, .8, now=4))
        c.update(False, False, False, False, .8, now=5)
        self.assertIsNone(c.update(True, False, False, False, .8, now=5.1))
        c.update(False, False, False, False, .8, now=6)
        c.update(False, False, False, False, .8, now=6.21)
        self.assertEqual(c.update(True, False, False, False, .8, now=6.22), 'toggle')

    def test_calibration_and_mode_chords_are_exclusive(self):
        for buttons in ((1,1,1,1), (1,0,1,0), (0,1,0,1), (1,1,0,0)):
            c = RecordingChords()
            self.assertIsNone(c.update(*buttons, .9, now=1))
            self.assertIsNone(c.update(1,0,0,0,.9,now=2))
        self.assertIsNone(RecordingChords().update(1,0,0,0,.5))
        self.assertEqual(RecordingChords().update(0,1,0,0,.8), 'discard')

    def test_explicit_window_survives_close_and_excludes_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            initialize(run, manual_recording=True)
            c=RecordingControl(run/'recording_control.json')
            self.assertIsNone(c.handle('toggle', pose_ready=False, now_ns=100))
            with self.assertRaises(ValueError): recording_window(run)
            self.assertEqual(c.handle('toggle', pose_ready=True, now_ns=200), 'started')
            self.assertEqual(c.handle('toggle', pose_ready=True, now_ns=400), 'stopped')
            self.assertIsNone(c.handle('toggle',pose_ready=True,now_ns=500))
            np.testing.assert_array_equal(recording_mask(run,[100,200,300,400,500]),[0,1,1,0,0])
            with self.assertRaises(FileExistsError): RecordingControl(run/'recording_control.json')
            self.assertFalse(list(run.glob('*.tmp')))

    def test_abort_never_publishes_training_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp); initialize(run, manual_recording=True)
            c=RecordingControl(run/'recording_control.json')
            c.handle('toggle',pose_ready=True,now_ns=100)
            c.handle('discard',pose_ready=True,now_ns=200)
            with patch('capture_session.inspect',return_value=({'status':'incomplete','checks':[]},{})), \
                 patch('capture_session.publish_mentor_submission') as publish:
                result=_finalize_artifacts(run, unittest.mock.Mock())
            self.assertEqual(result['status'],'discarded');publish.assert_not_called()
            self.assertFalse((run/'data').exists()); self.assertFalse((run/'completion.json').exists())
            self.assertTrue((run/'discarded.json').exists())
            with self.assertRaises(ValueError):recording_mask(run,[100,150])

    def test_historical_runs_have_no_implicit_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);initialize(run)
            self.assertIsNone(recording_window(run))
            np.testing.assert_array_equal(recording_mask(run,[100,200]),[1,1])

    def test_manual_final_alignment_uses_selected_clock_and_mode_rows(self):
        # Raw integrity covers three samples; the exported window has two.
        # This catches accidental reuse of the full-run mode array at close.
        from capture_session import finalize
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp); initialize(run,profile='bridge',manual_recording=True)
            c=RecordingControl(run/'recording_control.json')
            c.handle('toggle',pose_ready=True,now_ns=2_000_000_000)
            c.handle('toggle',pose_ready=True,now_ns=3_000_000_000)
            (run/'pico_replay_archive.npz').write_bytes(b'archive fixture')
            (run/'prepared').mkdir()
            (run/'prepared/quality_report.json').write_text('{"status":"development_checks_pass"}')
            (run/'sonic_state').mkdir()
            (run/'sonic_state/encoder_mode.csv').write_text('time_monotonic_ms,encoder_mode_0\n1000,1\n2000,2\n2020,2\n')
            raw={'state_monotonic_ns':np.array([1_000_000_000,2_000_000_000,2_020_000_000]),
                 'time':np.array([0.,1.,1.02])}
            selected={k:v[1:] for k,v in raw.items()}
            detail={'robot':({},raw,None,None),'modes':np.array([1,2,2]),
                    'human':{'packet_time':np.array([1.,2.,2.02])}}
            with patch('capture_session.inspect',return_value=({'status':'complete','checks':[]},detail)), \
                 patch('prepare_trajectory.load_robot',return_value=({},selected,np.array([1,2]),0)):
                self.assertEqual(finalize(run)['status'],'complete')
            with np.load(run/'prepared/alignment.npz') as a:
                np.testing.assert_array_equal(a['human_valid'],[True,True])
                self.assertEqual(len(a['host_monotonic_s']),2)


if __name__=='__main__': unittest.main()
