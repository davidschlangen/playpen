import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from playpen.cli import evaluate


class EvaluateTest(unittest.TestCase):

    def setUp(self):
        self.model_spec = MagicMock()
        self.model_spec.model_name = "test-model"
        self.gen_args = {"temperature": 0.0, "max_tokens": 300}
        self.results_dir = Path("/tmp/test-results")

        self.mock_evaluate_suite = patch("playpen.cli.evaluate_suite").start()
        self.mock_evaluate_suite.return_value = 0.5

        self.mock_get_suite_game_map = patch("playpen.cli.get_suite_game_map").start()
        self.mock_get_suite_game_map.return_value = {"clem": ["wordle"], "static": []}

        self.mock_store = patch("playpen.cli.store_eval_score").start()

    def tearDown(self):
        patch.stopall()

    # ------------------------------------------------------------------ #
    # Error cases                                                          #
    # ------------------------------------------------------------------ #

    def test_no_suite_no_game_raises_value_error(self):
        with self.assertRaises(ValueError):
            evaluate(None, self.model_spec, self.gen_args, self.results_dir, None, False)

    # ------------------------------------------------------------------ #
    # Suite selection / game_selector derivation                          #
    # ------------------------------------------------------------------ #

    def test_suite_clem_sets_clem_game_selector(self):
        evaluate("clem", self.model_spec, self.gen_args, self.results_dir, None, False)
        self.mock_get_suite_game_map.assert_called_once()
        called_with = self.mock_get_suite_game_map.call_args[0][0]
        self.assertIn("{'benchmark':['2.0']}", called_with)

    def test_suite_all_sets_both_game_selectors(self):
        evaluate("all", self.model_spec, self.gen_args, self.results_dir, None, False)
        self.mock_get_suite_game_map.assert_called_once()
        called_with = self.mock_get_suite_game_map.call_args[0][0]
        self.assertIn("{'benchmark':['2.0']}", called_with)
        self.assertIn("{'benchmark':['static_1.0']}", called_with)

    def test_both_suite_and_game_selector_does_not_raise(self):
        evaluate("clem", self.model_spec, self.gen_args, self.results_dir, "wordle", False)
        self.mock_get_suite_game_map.assert_called_once_with("wordle")

    def test_game_selector_without_suite_does_not_raise(self):
        evaluate(None, self.model_spec, self.gen_args, self.results_dir, "wordle", False)
        self.mock_get_suite_game_map.assert_called_once_with("wordle")

    # ------------------------------------------------------------------ #
    # skip_gameplay / dataset_name                                         #
    # ------------------------------------------------------------------ #

    def test_skip_gameplay_passes_none_as_dataset_name(self):
        evaluate("clem", self.model_spec, self.gen_args, self.results_dir, None, True)
        self.assertIsNone(self.mock_evaluate_suite.call_args[0][5])

    def test_no_skip_gameplay_passes_instances(self):
        evaluate("clem", self.model_spec, self.gen_args, self.results_dir, None, False)
        self.assertEqual(self.mock_evaluate_suite.call_args[0][5], "instances")

    # ------------------------------------------------------------------ #
    # store_eval_score calls                                               #
    # ------------------------------------------------------------------ #

    def test_clemscore_is_stored_after_eval(self):
        evaluate("clem", self.model_spec, self.gen_args, self.results_dir, None, False)
        expected_file = self.results_dir / f"{self.model_spec.model_name}.val.json"
        self.mock_store.assert_called_once_with(expected_file, "clemscore", 0.5)

    def test_suites_with_games_each_store_clemscore(self):
        self.mock_get_suite_game_map.return_value = {"clem": ["wordle"], "static": ["mmlu_pro"]}
        evaluate("all", self.model_spec, self.gen_args, self.results_dir, None, False)
        self.assertEqual(self.mock_store.call_count, 2)
        stored_keys = [c[0][1] for c in self.mock_store.call_args_list]
        self.assertEqual(stored_keys, ["clemscore", "statscore"])

    def test_empty_suite_does_not_store_score(self):
        self.mock_get_suite_game_map.return_value = {"clem": [], "static": []}
        evaluate("clem", self.model_spec, self.gen_args, self.results_dir, None, False)
        self.mock_store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
