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
        self.mock_game_spec = MagicMock()

        self.mock_evaluate_suite = patch("playpen.cli.evaluate_suite").start()
        self.mock_evaluate_suite.return_value = 0.5

        self.mock_game_spec_cls = patch("playpen.cli.GameSpec").start()
        self.mock_game_spec_cls.from_dict.return_value = self.mock_game_spec

        self.mock_store = patch("playpen.cli.store_eval_score").start()

    def tearDown(self):
        patch.stopall()

    # ------------------------------------------------------------------ #
    # Error cases                                                          #
    # ------------------------------------------------------------------ #

    def test_no_suite_error_game_selected(self):
        with self.assertRaises(ValueError):
            evaluate(None, self.model_spec, self.gen_args, self.results_dir, "wordle", False)

    def test_no_suite_error_no_game_selected(self):
        with self.assertRaises(ValueError):
            evaluate(None, self.model_spec, self.gen_args, self.results_dir, None, False)

    def test_game_selector_not_in_clem_suite_error(self):
        with self.assertRaises(ValueError):
            evaluate("clem", self.model_spec, self.gen_args, self.results_dir, "mmlu_pro", False)

    def test_game_selector_not_in_static_suite_error(self):
        with self.assertRaises(ValueError):
            evaluate("static", self.model_spec, self.gen_args, self.results_dir, "wordle", False)

    # ------------------------------------------------------------------ #
    # Suite selection                                                      #
    # ------------------------------------------------------------------ #

    def evaluate_suite_clem(self):
        evaluate("clem", self.model_spec, self.gen_args, self.results_dir, None, False)

        self.mock_evaluate_suite.assert_called_once()
        self.assertEqual(self.mock_evaluate_suite.call_args[0][0], "clem")

    def evaluate_suite_static(self):
        evaluate("static", self.model_spec, self.gen_args, self.results_dir, None, False)

        self.mock_evaluate_suite.assert_called_once()
        self.assertEqual(self.mock_evaluate_suite.call_args[0][0], "static")

    def evaluate_suite_all(self):
        evaluate("all", self.model_spec, self.gen_args, self.results_dir, None, False)

        self.assertEqual(self.mock_evaluate_suite.call_count, 2)
        suite_names = [c[0][0] for c in self.mock_evaluate_suite.call_args_list]
        self.assertIn("clem", suite_names)
        self.assertIn("static", suite_names)

    # ------------------------------------------------------------------ #
    # Benchmark alias auto-detection                                       #
    # ------------------------------------------------------------------ #

    def test_clem_benchmark_alias_auto_sets_clem_suite(self):
        evaluate(None, self.model_spec, self.gen_args, self.results_dir,
                 "{'benchmark':['2.0']}", False)

        self.mock_evaluate_suite.assert_called_once()
        self.assertEqual(self.mock_evaluate_suite.call_args[0][0], "clem")

    def test_static_benchmark_alias_auto_sets_static_suite(self):
        evaluate(None, self.model_spec, self.gen_args, self.results_dir,
                 "{'benchmark':['static_1.0']}", False)

        self.mock_evaluate_suite.assert_called_once()
        self.assertEqual(self.mock_evaluate_suite.call_args[0][0], "static")

    # ------------------------------------------------------------------ #
    # game_selector handling                                               #
    # ------------------------------------------------------------------ #

    def test_all_suite_ignores_game_selector(self):
        """suite='all' must set game_selector to None, so _game_selector = GameSpec.from_dict(...)."""
        evaluate("all", self.model_spec, self.gen_args, self.results_dir, "wordle", False)

        self.assertEqual(self.mock_evaluate_suite.call_count, 2)
        for c in self.mock_evaluate_suite.call_args_list:
            self.assertNotEqual(c[0][4], "wordle")

    def test_specific_game_selector_is_passed_through_for_clem_suite(self):
        evaluate("clem", self.model_spec, self.gen_args, self.results_dir, "wordle", False)

        self.mock_evaluate_suite.assert_called_once()
        self.assertEqual(self.mock_evaluate_suite.call_args[0][4], "wordle")

    def test_specific_game_selector_is_passed_through_for_static_suite(self):
        evaluate("static", self.model_spec, self.gen_args, self.results_dir, "mmlu_pro", False)

        self.mock_evaluate_suite.assert_called_once()
        self.assertEqual(self.mock_evaluate_suite.call_args[0][4], "mmlu_pro")

    def test_benchmark_alias_game_selector_ignored_when_suite_is_set(self):
        """If suite='clem' and game_selector is the clem alias, game_selector is dropped."""
        evaluate("clem", self.model_spec, self.gen_args, self.results_dir,
                 "{'benchmark':['2.0']}", False)

        self.mock_evaluate_suite.assert_called_once()
        self.assertEqual(self.mock_evaluate_suite.call_args[0][4], self.mock_game_spec)

    # ------------------------------------------------------------------ #
    # skip_gameplay / dataset_name                                         #
    # ------------------------------------------------------------------ #

    def test_skip_gameplay_passes_none_as_dataset_name(self):
        evaluate("clem", self.model_spec, self.gen_args, self.results_dir, None, True)

        self.assertIsNone(self.mock_evaluate_suite.call_args[0][5])

    def test_no_skip_gameplay_passes_instances_for_clem(self):
        evaluate("clem", self.model_spec, self.gen_args, self.results_dir, None, False)

        self.assertEqual(self.mock_evaluate_suite.call_args[0][5], "instances")

    def test_no_skip_gameplay_passes_instances_static_for_static(self):
        evaluate("static", self.model_spec, self.gen_args, self.results_dir, None, False)

        self.assertEqual(self.mock_evaluate_suite.call_args[0][5], "instances-static")

    # ------------------------------------------------------------------ #
    # store_eval_score calls                                               #
    # ------------------------------------------------------------------ #

    def test_clemscore_is_stored_after_clem_eval(self):
        evaluate("clem", self.model_spec, self.gen_args, self.results_dir, None, False)

        expected_file = self.results_dir / f"{self.model_spec.model_name}.val.json"
        self.mock_store.assert_called_once_with(expected_file, "clemscore", 0.5)

    def test_statscore_is_stored_after_static_eval(self):
        evaluate("static", self.model_spec, self.gen_args, self.results_dir, None, False)

        expected_file = self.results_dir / f"{self.model_spec.model_name}.val.json"
        self.mock_store.assert_called_once_with(expected_file, "statscore", 0.5)

    def test_both_scores_are_stored_for_all_suite(self):
        evaluate("all", self.model_spec, self.gen_args, self.results_dir, None, False)

        self.assertEqual(self.mock_store.call_count, 2)
        stored_keys = [c[0][1] for c in self.mock_store.call_args_list]
        self.assertIn("clemscore", stored_keys)
        self.assertIn("statscore", stored_keys)


if __name__ == "__main__":
    unittest.main()
