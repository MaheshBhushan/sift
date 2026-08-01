"""Both API clients share one retry budget (issue #26).

The router was given a raised budget so a 110-message run survives a rate-limit
window; the vision pre-pass runs first and gates that run, so it has to survive
the same window or the fix only moved the failure one stage earlier.

Stdlib unittest, no test dependency:

    python -m unittest test_api_retries -v
"""
import unittest
from unittest import mock

import common
import media
import route


class SharedRetryBudget(unittest.TestCase):
    """Driven through a stub SDK: no API key, no network."""

    def setUp(self):
        # Each module imported api_key by name, so patch it where it is looked up.
        for target in (media, route):
            patcher = mock.patch.object(target, "api_key", return_value="test-key")
            patcher.start()
            self.addCleanup(patcher.stop)

    def _max_retries(self, build):
        with mock.patch("anthropic.Anthropic") as ctor:
            build()
        return ctor.call_args.kwargs.get("max_retries")

    def test_the_budget_is_above_the_sdk_default(self):
        self.assertGreater(common.API_MAX_RETRIES, 2)

    def test_the_vision_client_uses_it(self):
        self.assertEqual(self._max_retries(media._client), common.API_MAX_RETRIES)

    def test_the_router_client_uses_it(self):
        budget = self._max_retries(lambda: route.Router(use_cache=False))
        self.assertEqual(budget, common.API_MAX_RETRIES)

    def test_both_clients_agree(self):
        router = self._max_retries(lambda: route.Router(use_cache=False))
        self.assertEqual(self._max_retries(media._client), router)


if __name__ == "__main__":
    unittest.main()
