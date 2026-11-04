import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

source = Path.home()/"repo/GR00T-WholeBodyControl/gear_sonic/utils/teleop/simulation_stop.py"
spec = importlib.util.spec_from_file_location("simulation_stop", source)
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class SimulationStopTests(unittest.TestCase):
    def test_old_or_partial_requests_cannot_stop_new_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'stop.json'
            module.atomic_json(path,{'token':'old','requested_monotonic':time.monotonic()})
            gate=module.SimulationStopGate(path)
            self.assertFalse(gate.requested())
            path.write_text('{')
            self.assertFalse(gate.requested())

    def test_acknowledgement_matches_this_request_and_stopped_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'stop.json';gate=module.SimulationStopGate(path)
            module.atomic_json(module.acknowledgement_path(path),{'token':'old','physics_stopped':True})
            def bridge():
                deadline=time.monotonic()+1.
                while time.monotonic()<deadline:
                    if gate.requested():
                        gate.acknowledge(66.14);return
                    time.sleep(.005)
            worker=threading.Thread(target=bridge);worker.start()
            self.assertTrue(module.request_simulation_stop(path,'Manager done',timeout=1.))
            worker.join()
            ack=json.loads(module.acknowledgement_path(path).read_text())
            self.assertEqual(ack['simulation_seconds'],66.14)
            self.assertEqual(ack['token'],gate.token)

    def test_missing_bridge_ack_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(module.request_simulation_stop(Path(tmp)/'stop.json','done',timeout=.02))
