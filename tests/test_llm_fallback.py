import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

import jd_analyzer_llm


class TestLLMFallback(unittest.TestCase):
    @mock.patch("jd_analyzer_llm.check_ollama_available", return_value=True)
    @mock.patch("jd_analyzer_llm.requests.post")
    def test_ollama_unavailable_falls_back_to_nlp(self, mock_post, mock_check):
        # Simulate Ollama API failure
        mock_post.side_effect = Exception("Connection refused")

        jd_text = "Senior Python developer needed with 4-6 years of experience in web applications."

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "jd_meta.yaml"

            # Call hybrid analyzer - should catch the exception and fall back
            config = jd_analyzer_llm.analyze_jd_hybrid(
                jd_text,
                output_path,
                use_llm=True,
                job_title="Senior Python Developer",
                requisition_id="TEST-REQ-123",
            )

            # Ollama was attempted once
            mock_post.assert_called_once()
            mock_check.assert_called()

            # Fallback should have produced a valid YAML config on disk
            self.assertTrue(output_path.exists())
            loaded = yaml.safe_load(output_path.read_text(encoding="utf-8"))

            # Basic structural checks
            self.assertIn("job", loaded)
            self.assertIn("must_haves", loaded)
            self.assertIn("nice_to_haves", loaded)

            # The returned config should match what was written
            self.assertEqual(loaded["job"]["title"], config["job"]["title"])
            self.assertEqual(loaded["job"]["requisition_id"], config["job"]["requisition_id"])


if __name__ == "__main__":
    unittest.main()


