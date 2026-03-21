"""Tests for prompt templates."""

from ci_bench.models.prompts import (
    get_template,
    list_conditions,
    list_variants,
    ALL_TEMPLATES,
)
import pytest


class TestPromptTemplates:
    def test_all_conditions_have_two_variants(self):
        """Every condition must have at least 2 variants for sensitivity analysis."""
        for condition in list_conditions():
            variants = list_variants(condition)
            assert len(variants) >= 2, (
                f"Condition '{condition}' has only {len(variants)} variant(s). "
                f"Need at least 2 for sensitivity analysis."
            )

    def test_render_fills_placeholder(self):
        """Templates render with the question inserted."""
        for key, template in ALL_TEMPLATES.items():
            rendered = template.render("What is 2+2?")
            assert "What is 2+2?" in rendered
            assert "{question}" not in rendered

    def test_get_template(self):
        t = get_template("direct", 1)
        assert t.condition == "direct"
        assert t.variant == 1

    def test_get_template_missing_raises(self):
        with pytest.raises(KeyError):
            get_template("nonexistent", 1)

    def test_four_conditions(self):
        conditions = list_conditions()
        assert set(conditions) == {"abstention", "confidence", "cot", "direct"}
