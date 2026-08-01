"""Regression tests for prompt-versioned routing cache (issue #1).

Stdlib unittest, no test dependency:

    python -m unittest test_prompt_cache -v
"""
import unittest
from unittest.mock import patch

import prompt
import route


class PromptFingerprint(unittest.TestCase):
    def test_is_stable_across_calls(self):
        self.assertEqual(prompt.fingerprint(), prompt.fingerprint())

    def test_changes_when_rubric_changes(self):
        before = prompt.fingerprint()
        with patch.object(prompt, "SYSTEM", prompt.SYSTEM + "\nAn extra rule."):
            self.assertNotEqual(before, prompt.fingerprint())
        self.assertEqual(before, prompt.fingerprint())

    def test_changes_when_examples_change(self):
        before = prompt.fingerprint()
        extra = prompt.FEWSHOT + [("a new situation", {"action": "mute"})]
        with patch.object(prompt, "FEWSHOT", extra):
            self.assertNotEqual(before, prompt.fingerprint())


class CacheIdentity(unittest.TestCase):
    """The bug: editing the rubric left the cache name unchanged, so the next run
    replayed decisions made under the old rubric."""

    def _cache_name(self):
        with patch.object(route.anthropic, "Anthropic"), patch.object(
            route, "api_key", return_value="test-key"
        ):
            return route.Router(model="test-model").cache.path.name

    def test_name_carries_the_rubric_fingerprint(self):
        self.assertIn(prompt.fingerprint(), self._cache_name())

    def test_editing_the_rubric_invalidates_the_cache(self):
        before = self._cache_name()
        with patch.object(prompt, "SYSTEM", prompt.SYSTEM + "\nAn extra rule."):
            after = self._cache_name()
        self.assertNotEqual(before, after)

    def test_same_rubric_keeps_the_same_cache(self):
        self.assertEqual(self._cache_name(), self._cache_name())

    def test_model_still_separates_caches(self):
        with patch.object(route.anthropic, "Anthropic"), patch.object(
            route, "api_key", return_value="test-key"
        ):
            a = route.Router(model="model-a").cache.path.name
            b = route.Router(model="model-b").cache.path.name
        self.assertNotEqual(a, b)

    def test_no_cache_flag_disables_caching(self):
        with patch.object(route.anthropic, "Anthropic"), patch.object(
            route, "api_key", return_value="test-key"
        ):
            self.assertIsNone(route.Router(use_cache=False).cache)


if __name__ == "__main__":
    unittest.main()
