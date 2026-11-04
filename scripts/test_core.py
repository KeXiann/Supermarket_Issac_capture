"""Small CPU-only regression suite; no simulator, external checkout or weights."""
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'bridge'), str(ROOT/'tests')]
names = ['test_bridge_core', 'test_bridge_recording', 'test_simulation_timing',
         'test_startup_posture', 'test_owned_process_cleanup', 'test_replay_clock', 'test_world_replay_clock']
names = [name for name in names if (ROOT/'tests'/f'{name}.py').exists()]
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
raise SystemExit(not result.wasSuccessful())
