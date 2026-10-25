from pathlib import Path

import pytest
from pydantic import ValidationError

from voxa_shared.kb.chunking import approx_tokens, chunk_faq, chunk_text
from voxa_shared.kb.retrieve import or_query
from voxa_shared.redact import redact
from voxa_shared.tenant import TenantConfig, load_config

ACME = Path(__file__).parents[2] / "samples/acme/tenant.yaml"


def test_acme_config_valid_and_roundtrips():
    c = load_config(ACME.read_text())
    again = TenantConfig.model_validate(c.model_dump(mode="json"))
    assert again.fingerprint() == c.fingerprint()
    assert {t.name for t in c.tools} == {"order_status", "create_ticket"}
    assert [t for t in c.tools if t.side_effect][0].name == "create_ticket"


@pytest.mark.parametrize("patch", [
    {"tenant_id": "Bad-Id"}, {"schema_version": 99}, {"extra_field": 1},
    {"turn": {"vad_threshold": 5}}, {"identity": {"org_name": ""}},
])
def test_invalid_configs_rejected(patch):
    base = load_config(ACME.read_text()).model_dump(mode="json")
    base.update(patch)
    with pytest.raises(ValidationError):
        TenantConfig.model_validate(base)


def test_tool_validation():
    base = load_config(ACME.read_text()).model_dump(mode="json")
    base["tools"][1]["name"] = "order_status"
    with pytest.raises(ValidationError, match="duplicate"):
        TenantConfig.model_validate(base)
    base["tools"][1]["name"] = "x_tool"; base["tools"][1]["url"] = None
    with pytest.raises(ValidationError, match="needs url"):
        TenantConfig.model_validate(base)


def test_chunk_sizes_and_overlap():
    text = " ".join(f"Sentence number {i} talks about returns and refunds for item {i}." for i in range(300))
    chunks = chunk_text(text)
    assert len(chunks) > 3
    assert all(approx_tokens(c) <= 500 for c in chunks)
    assert all(approx_tokens(c) >= 250 for c in chunks[:-1])
    # overlap: last sentence of chunk i appears at the start of chunk i+1
    assert chunks[0].split(". ")[-1][:30] in chunks[1]


def test_faq_chunks_atomic():
    assert chunk_faq([("Q1?", "A1."), ("", "x")]) == ["Q: Q1?\nA: A1."]


def test_or_query_drops_stopwords():
    assert or_query("What is your return policy for headphones?") == "return | policy | headphones"


def test_redact():
    s = redact("mail me at jo@acme.com or call 415-555-1234, card 4111 1111 1111 1111, ssn 123-45-6789")
    assert "jo@" not in s and "415" not in s and "4111" not in s and "123-45" not in s
