from __future__ import annotations

import pytest

import app.services.recipes_plan as rp


def _patch(monkeypatch: pytest.MonkeyPatch, pantry: dict[str, float]) -> None:
    # Make canonicalization deterministic for tests (no profiles / synonyms involved)
    monkeypatch.setattr(rp, "_canonicalize_ingredient_name", lambda s: s.strip().lower())

    # Override DB lookup
    monkeypatch.setattr(
        rp,
        "_get_pantry_set",
        lambda: {k.strip().lower(): float(v) for k, v in pantry.items()},
    )


def _recipe_from_items(*items: str) -> dict:
    # Match your function’s normalized shape: {"item" or "raw"}
    return {"ingredients": [{"item": it} for it in items]}


def test_all_available_no_missing_or_partial(monkeypatch: pytest.MonkeyPatch):
    _patch(monkeypatch, {"milk": 1.0, "eggs": 2.0, "butter": 10.0})

    recipe = _recipe_from_items("milk", "eggs", "butter")
    missing, partial = rp.diff_recipe_against_pantry(recipe)

    assert missing == []
    assert partial == []


def test_missing_when_not_in_pantry(monkeypatch: pytest.MonkeyPatch):
    _patch(monkeypatch, {"flour": 1.0, "salt": 1.0})

    recipe = _recipe_from_items("flour", "salt", "baking powder")
    missing, partial = rp.diff_recipe_against_pantry(recipe)

    assert missing == ["baking powder"]
    assert partial == []


def test_missing_when_qty_zero_or_negative(monkeypatch: pytest.MonkeyPatch):
    _patch(monkeypatch, {"rice": 0.0, "beans": -1.0})

    recipe = _recipe_from_items("rice", "beans")
    missing, partial = rp.diff_recipe_against_pantry(recipe)

    assert set(missing) == {"rice", "beans"}
    assert partial == []


def test_partial_when_qty_between_zero_and_one(monkeypatch: pytest.MonkeyPatch):
    _patch(monkeypatch, {"rice": 0.5})

    recipe = _recipe_from_items("rice")
    missing, partial = rp.diff_recipe_against_pantry(recipe)

    assert missing == []
    assert partial == ["rice"]


def test_ingredient_parsing_prefers_item_then_raw(monkeypatch: pytest.MonkeyPatch):
    _patch(monkeypatch, {"olive oil": 1.0})

    recipe = {"ingredients": [{"raw": "olive oil"}]}
    missing, partial = rp.diff_recipe_against_pantry(recipe)

    assert missing == []
    assert partial == []


def test_string_ingredient_entries_supported(monkeypatch: pytest.MonkeyPatch):
    _patch(monkeypatch, {"salt": 1.0})

    recipe = {"ingredients": ["salt"]}
    missing, partial = rp.diff_recipe_against_pantry(recipe)

    assert missing == []
    assert partial == []


def test_skips_blank_items(monkeypatch: pytest.MonkeyPatch):
    _patch(monkeypatch, {"salt": 1.0})

    recipe = {"ingredients": [{"item": ""}, {"item": "   "}, {"raw": None}, "   ", "salt"]}
    missing, partial = rp.diff_recipe_against_pantry(recipe)

    assert missing == []
    assert partial == []


def test_deduping_and_stable_order(monkeypatch: pytest.MonkeyPatch):
    # milk is partial, eggs missing; repeats should not duplicate outputs
    _patch(monkeypatch, {"milk": 0.5, "eggs": 0.0})

    recipe = _recipe_from_items("milk", "milk", "eggs", "milk", "eggs")
    missing, partial = rp.diff_recipe_against_pantry(recipe)

    assert missing == ["eggs"]
    assert partial == ["milk"]


def test_case_and_whitespace_normalization_via_test_canon(monkeypatch: pytest.MonkeyPatch):
    _patch(monkeypatch, {"olive oil": 1.0})

    recipe = _recipe_from_items("  Olive Oil  ")
    missing, partial = rp.diff_recipe_against_pantry(recipe)

    assert missing == []
    assert partial == []
