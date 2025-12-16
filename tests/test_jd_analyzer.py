import unittest

from jd_analyzer import detect_domain, extract_experience_requirements


class TestJDAnalyzer(unittest.TestCase):
    def test_detect_domain_software(self):
        jd = "Senior Python developer needed, experience with Django and web applications."
        self.assertEqual(detect_domain(jd), "software")

    def test_extract_experience_range(self):
        jd = "4-6 years of experience required"
        exp = extract_experience_requirements(jd)
        self.assertEqual(exp["min_years"], 4)
        self.assertEqual(exp["max_years"], 6)


if __name__ == "__main__":
    unittest.main()


