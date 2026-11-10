import sys, os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from batch_risk_control.tables_utils import save_table


def test_save_table_writes_both_formats(tmp_path):
    df = pd.DataFrame({"method": ["a", "b"], "coverage": [0.901, 0.850], "width": [10.0, 12.5]})
    stem = tmp_path / "sub" / "mytable"
    save_table(df, stem, caption="Test caption", label="tab:test")
    assert (tmp_path / "sub" / "mytable.csv").exists()
    assert (tmp_path / "sub" / "mytable.tex").exists()

    csv_back = pd.read_csv(f"{stem}.csv")
    assert list(csv_back["method"]) == ["a", "b"]

    tex = (tmp_path / "sub" / "mytable.tex").read_text(encoding="utf-8")
    assert "Test caption" in tex
    assert "tab:test" in tex
    assert r"\toprule" in tex and r"\bottomrule" in tex
    assert "0.901" in tex  # float_format default applied


def test_save_table_escapes_underscores(tmp_path):
    df = pd.DataFrame({"method_name": ["dro_plugin"], "value": [1.0]})
    stem = tmp_path / "esc"
    save_table(df, stem, caption="c", label="l")
    tex = (tmp_path / "esc.tex").read_text(encoding="utf-8")
    assert r"dro\_plugin" in tex
    assert r"method\_name" in tex


def test_save_table_with_notes(tmp_path):
    df = pd.DataFrame({"x": [1.0]})
    stem = tmp_path / "n"
    save_table(df, stem, caption="c", label="l", notes="A footnote.")
    tex = (tmp_path / "n.tex").read_text(encoding="utf-8")
    assert "A footnote." in tex
    assert "tablenotes" in tex


def test_save_table_preserves_underscore_in_math_mode_header(tmp_path):
    df = pd.DataFrame({r"Final $\alpha_t$": [1.0], "plain_header": [2.0]})
    stem = tmp_path / "math"
    save_table(df, stem, caption="c", label="l")
    tex = (tmp_path / "math.tex").read_text(encoding="utf-8")
    assert r"Final $\alpha_t$" in tex  # subscript inside $...$ must survive unescaped
    assert r"plain\_header" in tex     # plain-text header still gets escaped
