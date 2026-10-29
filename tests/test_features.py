from src.features import extract_features, normalize_request

def test_features_are_stable_and_numeric():
    f = extract_features("GET /search?q=hello HTTP/1.1")
    assert "request_length" in f
    assert "sql_keyword_hits" in f
    assert all(isinstance(v, (int, float)) for v in f.values())

def test_normalization_decodes_percent_encoding():
    assert "script" in normalize_request("%3Cscript%3Ealert(1)%3C/script%3E")

def test_attack_indicators():
    sqli = extract_features("GET /?id=1 UNION SELECT password FROM users")
    xss = extract_features("GET /?q=%3Cscript%3Ealert(1)%3C/script%3E")
    assert sqli["has_union_select"] == 1
    assert xss["has_script_tag"] == 1
