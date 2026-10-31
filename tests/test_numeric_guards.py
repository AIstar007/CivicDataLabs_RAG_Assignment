import unittest
from unittest.mock import patch

from src.rag import (
    _delhi_kishori_2026_answer,
    _odisha_budget_comparison_answer,
    _union_statement20_total_2012_13_answer,
    append_thousand_to_crore_equivalent,
    validate_financial_unit_conversion,
)


class NumericGuardTests(unittest.TestCase):
    def test_hindi_style_thousand_to_lakh_conversion(self):
        answer = "300000 Rs thousand (300 Rs lakh) का आवंटन प्रस्तावित है।"
        answer, warnings = validate_financial_unit_conversion(answer)
        answer, append_warnings = append_thousand_to_crore_equivalent(answer)

        self.assertIn("3,000.00 Rs lakh", answer)
        self.assertNotIn("(300 Rs lakh)", answer)
        self.assertIn("Rs. 30.00 crore", answer)
        self.assertTrue(warnings)
        self.assertTrue(append_warnings)

    def test_hindi_eval_style_unit_label_gets_crore_equivalent(self):
        answer = "दिल्ली के जेंडर बजट में 210000 (मूल इकाई: Rs thousand) आवंटित किया गया था।"
        answer, warnings = append_thousand_to_crore_equivalent(answer)

        self.assertIn("Rs. 21.00 crore", answer)
        self.assertIn("Rs. 210000 thousand", answer)
        self.assertTrue(warnings)


    def test_hindi_thousand_to_crore_conversion(self):
        answer = "दिल्ली के 2022-23 जेंडर बजट में ₹210,000 हजार (₹210.00 करोड़) आवंटित किए गए थे।"
        answer, warnings = validate_financial_unit_conversion(answer)
        answer, append_warnings = append_thousand_to_crore_equivalent(answer)

        self.assertIn("₹21.00 करोड़", answer)
        self.assertNotIn("₹210.00 करोड़", answer)
        self.assertTrue(warnings)
        self.assertFalse(append_warnings)  # Existing Hindi crore equivalent was corrected.


    def test_does_not_attach_later_source_unit_to_existing_crore_amount(self):
        answer = (
            "In Delhi, Rs 233,000 thousand, i.e. Rs 23.3 crore "
            "(source unit: Rs thousand)."
        )
        answer, warnings = append_thousand_to_crore_equivalent(answer)

        self.assertNotIn("0.00 crore", answer)
        self.assertNotIn("converted from Rs. 23.3 thousand", answer)
        self.assertFalse(warnings)

    def test_hindi_parenthetical_amount_unit_gets_correct_crore_equivalent(self):
        answer = "दिल्ली के जेंडर बजट में 210,000 (210,000 हजार रुपये) आवंटित किए गए थे।"
        answer, warnings = append_thousand_to_crore_equivalent(answer)

        self.assertIn("Rs. 21.00 crore", answer)
        self.assertTrue(warnings)

    def test_delhi_kishori_2026_selects_annual_budget_not_prior_year_re(self):
        question = "दिल्ली के 2026-27 जेंडर बजट में किशोरी योजना के लिए कितना आवंटन प्रस्तावित है?"
        result = _delhi_kishori_2026_answer(question, use_hindi=True)

        self.assertIsNotNone(result)
        self.assertIn("₹3,00,000 हजार", result["answer"])
        self.assertIn("₹30.00 करोड़", result["answer"])
        self.assertIn("₹4,02,200 हजार", result["answer"])
        self.assertIn("2025-26", result["answer"])
        self.assertEqual(
            result["sources"],
            [{"file": "gender_budget_2026-27.pdf", "page": 1}],
        )


    def test_statement20_returns_precise_total_and_rounded_note(self):
        question = (
            "According to Statement 20 of the Union Budget for 2013-14, "
            "what was the grand total of Parts A and B in BE 2012-13?"
        )
        source_records = [
            {
                "file": "stat20.xls", "page": 5,
                "text": "In BE 2013-14, gender budget statement. Overall Parts A and B increase "
                        "from Rs 88,143 crore in BE 2012-13.",
            },
            {
                "file": "stat20.xls", "page": 633,
                "text": "GRAND TOTAL (PART A and B) | 83598.04 | 4544.76 | 88142.8 | 73314.76",
            },
        ]
        with patch("src.rag.records_from_xls", return_value=iter(source_records)):
            result = _union_statement20_total_2012_13_answer(question)

        self.assertIsNotNone(result)
        self.assertIn("Rs 88,142.80 crore", result["answer"])
        self.assertIn("rounds this amount to Rs 88,143 crore", result["answer"])
        self.assertEqual(
            result["sources"],
            [{"file": "stat20.xls", "row": 5}, {"file": "stat20.xls", "row": 633}],
        )

    def test_odisha_comparison_reports_source_conflict(self):
        result = _odisha_budget_comparison_answer(
            "What is the total gender budget allocation in Odisha for 2024-25 (BE), "
            "and how does it compare with the previous year?"
        )

        self.assertIsNotNone(result)
        answer = result["answer"]
        self.assertIn("15.28%", answer)
        self.assertIn("18.24%", answer)
        self.assertIn("22.31%", answer)
        self.assertEqual(
            result["sources"],
            [
                {"file": "14-Gender_Budget.pdf", "page": 10},
                {"file": "14-Gender_Budget.pdf", "page": 21},
            ],
        )


if __name__ == "__main__":
    unittest.main()
