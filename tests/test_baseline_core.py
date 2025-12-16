import unittest

from baseline import (
    normalize_text,
    extract_years_experience,
    score_must_have_coverage,
)


class TestBaselineCore(unittest.TestCase):
    def test_normalize_text_lower_and_strip(self):
        self.assertEqual(normalize_text("  Hello World  "), "hello world")

    def test_extract_years_experience_single_and_range(self):
        text = "I have 5 years of experience and previously had 4-6 years in similar roles."
        years = extract_years_experience(text)
        # Current implementation picks the highest value found (6.0 here)
        self.assertAlmostEqual(years, 6.0, places=1)

    def test_extract_years_experience_none(self):
        self.assertEqual(extract_years_experience("No explicit years mentioned"), 0.0)

    def test_score_must_have_coverage_all_met(self):
        text = "Python and SQL developer with cloud experience."
        must_haves = ["python", "sql"]
        score, missing, found = score_must_have_coverage(text, must_haves)
        self.assertEqual(score, 1.0)
        self.assertEqual(missing, [])
        self.assertCountEqual(found, must_haves)

    def test_score_must_have_coverage_partial(self):
        text = "Python developer."
        must_haves = ["python", "sql"]
        score, missing, found = score_must_have_coverage(text, must_haves)
        self.assertAlmostEqual(score, 0.5)
        self.assertEqual(missing, ["sql"])
        self.assertEqual(found, ["python"])


if __name__ == "__main__":
    unittest.main()


