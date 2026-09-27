"""Input validation, safe report rendering, and benchmark reproducibility."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark import generate
from match import check_inputs
from review import render_html


class WorkflowTests(unittest.TestCase):
    def test_generated_records_are_deterministic_and_have_valid_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            generate(a, 60, 42)
            generate(b, 60, 42)
            for p in a.iterdir():
                self.assertEqual(hashlib.sha256(p.read_bytes()).digest(), hashlib.sha256((b / p.name).read_bytes()).digest())
            check_inputs(a)
            targets = set()
            for source in (2, 3):
                with (a / f'test_source{source}.tsv').open() as f:
                    targets.update(row['entity_id'] for row in csv.DictReader(f, delimiter='\t'))
            assigned = set()
            with (a / 'ground_truth.tsv').open() as f:
                labels = list(csv.DictReader(f, delimiter='\t'))
            self.assertEqual(len(labels), 60)
            for row in labels:
                links = set(filter(None, row['matched_entity_ids'].split(',')))
                self.assertTrue(links <= targets)
                self.assertFalse(links & assigned)
                assigned.update(links)

    def test_duplicate_input_ids_fail_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'data'
            generate(data, 12, 42)
            path = data / 'test_source1.tsv'
            with path.open('a') as f:
                f.write(path.read_text().splitlines()[1] + '\n')
            with self.assertRaisesRegex(ValueError, 'duplicate entity ID'):
                check_inputs(data)

    def test_report_cannot_break_out_of_json_script(self):
        attack = '</script><script>window.injected=true</script>&<img src=x onerror=alert(1)>'
        report = {'record': attack}
        html = render_html(report)
        data = html.split('<script type="application/json" id="report-data">')[1].split('</script>')[0]
        self.assertNotIn('<', data)
        self.assertEqual(json.loads(data), report)
        self.assertNotIn(attack, html)


if __name__ == '__main__':
    unittest.main()
