"""Tests for the medical expert web search tool."""

from services.medical_expert_agent.tools.search_web import (
    DEFAULT_INCLUDE_DOMAINS,
    DEFAULT_MAX_RESULTS,
    _medical_web_include_domains,
    medical_web_max_results,
    search_web,
)


def test_search_web_returns_empty_when_tavily_key_missing(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    assert search_web("diabetes clinical guidance") == []


# ---------------------------------------------------------------------------
# medical_web_max_results env config
# ---------------------------------------------------------------------------


class TestMedicalWebMaxResults:
    def test_default_when_no_env(self, monkeypatch):
        monkeypatch.delenv("MARGE_WEB_RAG_MAX_RESULTS", raising=False)
        monkeypatch.delenv("MEDICAL_WEB_SEARCH_MAX_RESULTS", raising=False)
        assert medical_web_max_results() == DEFAULT_MAX_RESULTS

    def test_marge_env_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("MARGE_WEB_RAG_MAX_RESULTS", "5")
        monkeypatch.setenv("MEDICAL_WEB_SEARCH_MAX_RESULTS", "2")
        assert medical_web_max_results() == 5

    def test_legacy_env_used_when_marge_unset(self, monkeypatch):
        monkeypatch.delenv("MARGE_WEB_RAG_MAX_RESULTS", raising=False)
        monkeypatch.setenv("MEDICAL_WEB_SEARCH_MAX_RESULTS", "4")
        assert medical_web_max_results() == 4

    def test_clamped_to_upper_bound(self, monkeypatch):
        monkeypatch.setenv("MARGE_WEB_RAG_MAX_RESULTS", "999")
        assert medical_web_max_results() == 5

    def test_clamped_to_lower_bound(self, monkeypatch):
        monkeypatch.setenv("MARGE_WEB_RAG_MAX_RESULTS", "0")
        assert medical_web_max_results() == 1

    def test_invalid_string_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MARGE_WEB_RAG_MAX_RESULTS", "not-a-number")
        assert medical_web_max_results() == DEFAULT_MAX_RESULTS

    def test_requested_arg_caps_configured_value(self, monkeypatch):
        monkeypatch.setenv("MARGE_WEB_RAG_MAX_RESULTS", "5")
        # Caller asks for 2 → return 2 (≤ configured 5)
        assert medical_web_max_results(2) == 2

    def test_requested_arg_does_not_exceed_configured(self, monkeypatch):
        monkeypatch.setenv("MARGE_WEB_RAG_MAX_RESULTS", "2")
        # Caller asks for 5 but env caps at 2 → return 2
        assert medical_web_max_results(5) == 2


# ---------------------------------------------------------------------------
# _medical_web_include_domains env config
# ---------------------------------------------------------------------------


class TestMedicalWebIncludeDomains:
    def test_default_when_no_env(self, monkeypatch):
        monkeypatch.delenv("MEDICAL_WEB_SEARCH_INCLUDE_DOMAINS", raising=False)
        assert _medical_web_include_domains() == list(DEFAULT_INCLUDE_DOMAINS)

    def test_default_includes_medlineplus_and_pubmed(self):
        assert "medlineplus.gov" in DEFAULT_INCLUDE_DOMAINS
        assert "pubmed.ncbi.nlm.nih.gov" in DEFAULT_INCLUDE_DOMAINS

    def test_comma_separated_override(self, monkeypatch):
        monkeypatch.setenv(
            "MEDICAL_WEB_SEARCH_INCLUDE_DOMAINS", "who.int,nice.org.uk"
        )
        assert _medical_web_include_domains() == ["who.int", "nice.org.uk"]

    def test_semicolon_separated_override(self, monkeypatch):
        monkeypatch.setenv(
            "MEDICAL_WEB_SEARCH_INCLUDE_DOMAINS", "ada.org;cdc.gov"
        )
        assert _medical_web_include_domains() == ["ada.org", "cdc.gov"]

    def test_mixed_separators_and_whitespace(self, monkeypatch):
        monkeypatch.setenv(
            "MEDICAL_WEB_SEARCH_INCLUDE_DOMAINS",
            "  who.int , nice.org.uk ; cdc.gov  ",
        )
        assert _medical_web_include_domains() == [
            "who.int",
            "nice.org.uk",
            "cdc.gov",
        ]

    def test_empty_segments_filtered_out(self, monkeypatch):
        monkeypatch.setenv(
            "MEDICAL_WEB_SEARCH_INCLUDE_DOMAINS", ",,who.int,,,"
        )
        assert _medical_web_include_domains() == ["who.int"]

    def test_empty_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MEDICAL_WEB_SEARCH_INCLUDE_DOMAINS", "")
        assert _medical_web_include_domains() == list(DEFAULT_INCLUDE_DOMAINS)
