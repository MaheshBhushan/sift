"""Retry behaviour of Router._ask_retrying. Stdlib unittest -- run with:

    python -m unittest discover -s code
"""
import json
import unittest
from unittest import mock

import anthropic
import httpx

from route import BAD_JSON_ATTEMPTS, MAX_ATTEMPTS, MalformedResponse, Router


def _router():
    """A Router without __init__ -- no API key, no client, no cache needed."""
    return object.__new__(Router)


def _status_error(cls, status):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    return cls("boom", response=response, body=None)


def _rate_limit():
    return _status_error(anthropic.RateLimitError, 429)


def _overloaded():
    return _status_error(anthropic.InternalServerError, 529)


def _connection():
    return anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )


class AskRetryingTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("route.time.sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def test_returns_first_result_without_sleeping(self):
        router = _router()
        router._ask = mock.Mock(return_value={"action": "notify"})

        self.assertEqual(router._ask_retrying({}), {"action": "notify"})
        self.assertEqual(router._ask.call_count, 1)
        self.sleep.assert_not_called()

    def test_recovers_from_transient_api_errors(self):
        """The regression: these used to propagate and kill the run."""
        for error in (_rate_limit(), _overloaded(), _connection()):
            with self.subTest(error=type(error).__name__):
                router = _router()
                router._ask = mock.Mock(side_effect=[error, {"action": "notify"}])

                self.assertEqual(router._ask_retrying({}), {"action": "notify"})
                self.assertEqual(router._ask.call_count, 2)

    def test_recovers_from_bad_json(self):
        router = _router()
        router._ask = mock.Mock(
            side_effect=[json.JSONDecodeError("nope", "", 0), {"action": "notify"}]
        )

        self.assertEqual(router._ask_retrying({}), {"action": "notify"})
        self.assertEqual(router._ask.call_count, 2)

    def test_gives_up_on_malformed_responses_after_one_retry(self):
        """A truncated response repeats; four attempts just bill four calls."""
        router = _router()
        router._ask = mock.Mock(side_effect=MalformedResponse("no JSON in response"))

        with self.assertRaises(MalformedResponse):
            router._ask_retrying({})
        self.assertEqual(router._ask.call_count, BAD_JSON_ATTEMPTS)
        self.assertLess(router._ask.call_count, MAX_ATTEMPTS)

    def test_does_not_retry_a_plain_value_error(self):
        """Only a malformed *response* is retryable; a ValueError from building
        the request is a bug to surface, not bad luck to wait out."""
        router = _router()
        router._ask = mock.Mock(side_effect=ValueError("pack is not serializable"))

        with self.assertRaises(ValueError):
            router._ask_retrying({})
        self.assertEqual(router._ask.call_count, 1)

    def test_backs_off_exponentially_between_attempts(self):
        router = _router()
        router._ask = mock.Mock(side_effect=[_rate_limit(), _rate_limit(), {"action": "notify"}])

        router._ask_retrying({})
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [1, 2])

    def test_raises_after_max_attempts(self):
        router = _router()
        router._ask = mock.Mock(side_effect=_rate_limit())

        with self.assertRaises(anthropic.RateLimitError):
            router._ask_retrying({})
        self.assertEqual(router._ask.call_count, MAX_ATTEMPTS)
        # No sleep after the final failure.
        self.assertEqual(self.sleep.call_count, MAX_ATTEMPTS - 1)

    def test_does_not_retry_non_transient_api_errors(self):
        """A 401 fails identically every time; retrying just burns the backoff."""
        for cls, status in ((anthropic.AuthenticationError, 401), (anthropic.BadRequestError, 400)):
            with self.subTest(status=status):
                router = _router()
                router._ask = mock.Mock(side_effect=_status_error(cls, status))

                with self.assertRaises(cls):
                    router._ask_retrying({})
                self.assertEqual(router._ask.call_count, 1)


class CallCounters(unittest.TestCase):
    """`calls` is billed requests; `asked` is messages sent. A retried message
    makes them diverge, and main.py subtracts `asked` -- not `calls` -- from the
    row count, which is what used to be able to report negative cache hits."""

    def _router_with_responses(self, *responses):
        router = _router()
        router.calls = 0
        router.asked = 0
        router.model = "test-model"
        router.cache = None
        router.client = mock.Mock()
        router.client.messages.create = mock.Mock(side_effect=responses)
        return router

    @staticmethod
    def _resp(text):
        block = mock.Mock(type="text")
        block.text = text
        return mock.Mock(content=[block], stop_reason="max_tokens")

    def test_an_unparseable_response_was_still_billed(self):
        router = self._router_with_responses(self._resp("sorry, no JSON here"))

        with mock.patch("route.prompt.user_message", return_value="u"), self.assertRaises(
            MalformedResponse
        ):
            router._ask({"message": {"message_id": "m1"}})

        self.assertEqual(router.calls, 1)

    def test_a_retried_message_bills_twice_but_counts_as_one_message(self):
        router = self._router_with_responses(
            self._resp("no JSON"), self._resp('{"action": "notify"}')
        )

        with mock.patch("route.prompt.user_message", return_value="u"), mock.patch(
            "route.time.sleep"
        ), mock.patch("route.rules.apply", return_value=({"action": "notify"}, [])), mock.patch(
            "route.rules.reason_override", side_effect=lambda d, f: d
        ):
            router.route({"message": {"message_id": "m1"}})

        self.assertEqual(router.calls, 2)
        self.assertEqual(router.asked, 1)


if __name__ == "__main__":
    unittest.main()
