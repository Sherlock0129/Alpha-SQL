import unittest

from alphasql.jev.client import JevAPIError, JevClient, JevTimeoutError
from alphasql.jev.models import (
    ChoiceQuestion,
    ChoiceResult,
    JevResponseError,
    NoulQuestion,
    NoulResult,
    ScoreQuestion,
    ScoreResult,
)
from alphasql.jev.runtime import JevConfigurationError, JevSettings


class JevClientTest(unittest.TestCase):
    def setUp(self):
        self.settings = JevSettings(api_key=None)

    def test_parses_all_typed_answers_and_usage(self):
        def transport(state, questions):
            self.assertEqual(state["question"], "test")
            return {
                "model": "mock-jev",
                "usage": {"input_tokens": 12, "output_tokens": 3},
                "answers": {
                    "choice": {
                        "choice": "a", "probabilities": {"a": 0.8, "b": 0.2},
                        "confidence": 0.7,
                    },
                    "score": {
                        "score": 2, "probabilities": [0.1, 0.2, 0.7],
                        "confidence": 0.6,
                    },
                    "noul": {"noul": 0.75},
                },
            }

        response = JevClient(self.settings, transport=transport).ask(
            {"question": "test"},
            {
                "choice": ChoiceQuestion({"a": "A", "b": "B"}),
                "score": ScoreQuestion(["bad", "okay", "good"]),
                "noul": NoulQuestion("Is this relevant?"),
            },
        )
        self.assertIsInstance(response.answers["choice"], ChoiceResult)
        self.assertIsInstance(response.answers["score"], ScoreResult)
        self.assertIsInstance(response.answers["noul"], NoulResult)
        self.assertEqual(response.usage.input_tokens, 12)
        self.assertEqual(response.answers["choice"].usage.output_tokens, 3)
        self.assertAlmostEqual(response.answers["noul"].probabilities["true"], 0.75)

    def test_invalid_response_is_not_silently_accepted(self):
        client = JevClient(self.settings, transport=lambda state, questions: {"answers": {}})
        with self.assertRaisesRegex(JevResponseError, "omitted"):
            client.ask({}, {"needed": NoulQuestion("Needed?")})

    def test_timeout_is_wrapped_clearly(self):
        def timeout(state, questions):
            raise TimeoutError("mock timeout")

        with self.assertRaisesRegex(JevTimeoutError, "timed out"):
            JevClient(self.settings, transport=timeout).ask(
                {}, {"needed": NoulQuestion("Needed?")}
            )

    def test_api_error_is_wrapped_without_being_swallowed(self):
        def fail(state, questions):
            raise RuntimeError("mock service failure")

        with self.assertRaisesRegex(JevAPIError, "mock service failure"):
            JevClient(self.settings, transport=fail).ask(
                {}, {"needed": NoulQuestion("Needed?")}
            )

    def test_no_key_fails_before_sdk_or_network(self):
        client = JevClient(self.settings)
        with self.assertRaisesRegex(JevConfigurationError, "no network call"):
            client.ask({}, {"needed": NoulQuestion("Needed?")})


if __name__ == "__main__":
    unittest.main()
