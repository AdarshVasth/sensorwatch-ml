import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

from app import DemoHandler
from model import benchmark_data, evaluate, normal_readings, score


class ModelTests(unittest.TestCase):
    def test_holdout_has_both_classes_and_no_training_duplicates(self):
        holdout, labels = benchmark_data()
        self.assertEqual(holdout.shape, (260, 3))
        self.assertEqual(np.bincount(labels).tolist(), [200, 60])
        baseline = set(map(tuple, normal_readings(42, 600)))
        self.assertTrue(baseline.isdisjoint(map(tuple, holdout)))

    def test_fault_ranks_above_typical_reading(self):
        baseline, baseline_flag = score([62, 2, 8])
        fault, fault_flag = score([77, 3.4, 11])
        self.assertFalse(baseline_flag)
        self.assertTrue(fault_flag)
        self.assertGreater(fault, baseline)

    def test_rejects_invalid_inputs(self):
        for reading in ([62, 2], [62, -1, 8], ["bad", 2, 8], ["nan", 2, 8],
                        [62, 2, float("inf")]):
            with self.subTest(reading=reading), self.assertRaises(ValueError):
                score(reading)

    def test_benchmark_counts_and_ranking(self):
        results = evaluate()
        self.assertEqual(results["True negatives"] + results["False positives"], 200)
        self.assertEqual(results["True positives"] + results["False negatives"], 60)
        self.assertGreater(results["AUROC"], 0.9)


class DemoTests(unittest.TestCase):
    def test_http_entry_points(self):
        with ThreadingHTTPServer(("127.0.0.1", 0), DemoHandler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            origin = f"http://127.0.0.1:{server.server_port}"
            try:
                for path, expected in (
                    ("/", "Synthetic benchmark"),
                    ("/score?reading=62&reading=2&reading=8", "Within baseline"),
                    ("/score?reading=77&reading=3.4&reading=11", "Review suggested"),
                ):
                    with self.subTest(path=path), urlopen(origin + path, timeout=5) as response:
                        self.assertEqual(response.status, 200)
                        self.assertIn(expected, response.read().decode())
                for path, status in (("/score?reading=nan", 400), ("/missing", 404)):
                    with self.subTest(path=path), self.assertRaises(HTTPError) as error:
                        urlopen(origin + path, timeout=5)
                    self.assertEqual(error.exception.code, status)
                    error.exception.close()
            finally:
                server.shutdown()
                worker.join()


if __name__ == "__main__":
    unittest.main()
