import re
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sanice import Sanice
from sanice.core import _cnpj_valido, _cpf_valido

LANGS = ["pt", "en", "zh", "hi"]


@pytest.fixture
def clf_df():
    rng = np.random.default_rng(0)
    n = 200
    renda = rng.normal(5000, 1500, n)
    idade = rng.integers(18, 70, n)
    cidade = rng.choice(["SP", "RJ", "MG"], n)
    alvo = (renda + idade * 20 + rng.normal(0, 300, n) > 5600).astype(int)
    return pd.DataFrame({"renda": renda, "idade": idade, "cidade": cidade, "pago": alvo})


@pytest.fixture
def reg_df():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 100, 150)
    return pd.DataFrame({"x": x, "y": 3 * x + rng.normal(0, 2, 150)})


def test_i18n_languages_have_the_same_keys_and_placeholders():
    base = Sanice.I18N["en"]
    for lang in LANGS:
        assert set(Sanice.I18N[lang]) == set(base), lang
        for key, texto in Sanice.I18N[lang].items():
            assert set(re.findall(r"\{(\w+)", texto)) == set(re.findall(r"\{(\w+)", base[key])), (lang, key)


def test_aliases_are_complete_and_unique_per_language():
    for idx in range(3):
        nomes = [alias[idx] for alias in Sanice.METHOD_ALIASES.values()]
        assert len(nomes) == len(set(nomes))
    for metodo in Sanice.METHOD_ALIASES:
        assert callable(getattr(Sanice, metodo))


def test_every_language_exposes_every_command(clf_df):
    for idx, lang in enumerate(["en", "zh", "hi"]):
        app = Sanice(clf_df, lang=lang)
        for alias in Sanice.METHOD_ALIASES.values():
            assert callable(getattr(app, alias[idx])), (lang, alias[idx])


def test_new_messages_are_translated(clf_df, caplog):
    for lang, trecho in [("pt", "duplicadas"), ("en", "duplicate"), ("zh", "重复"), ("hi", "duplicate rows hataye")]:
        caplog.clear()
        Sanice(clf_df, lang=lang).remover_duplicados()
        assert trecho in caplog.text


def test_reads_brazilian_csv_without_arguments(tmp_path):
    caminho = tmp_path / "vendas.CSV"
    caminho.write_bytes("município;valor\nSão Paulo;10\nBelém;20\n".encode("latin1"))
    df = Sanice(str(caminho)).pegar_dataframe()
    assert list(df.columns) == ["município", "valor"]
    assert df["município"].tolist() == ["São Paulo", "Belém"]


def test_reads_with_explicit_options(tmp_path):
    caminho = tmp_path / "d.csv"
    caminho.write_text("a|b\n1,5|2\n", encoding="utf-8")
    df = Sanice(str(caminho), sep="|", decimal=",", encoding="utf-8").pegar_dataframe()
    assert df["a"].iloc[0] == 1.5


def test_reads_excel_sheet(tmp_path):
    caminho = tmp_path / "d.xlsx"
    with pd.ExcelWriter(caminho) as escritor:
        pd.DataFrame({"a": [1]}).to_excel(escritor, sheet_name="um", index=False)
        pd.DataFrame({"b": [2]}).to_excel(escritor, sheet_name="dois", index=False)
    assert list(Sanice(str(caminho), aba="dois").pegar_dataframe().columns) == ["b"]


def test_unsupported_format_is_reported(caplog):
    app = Sanice("dados.xyz")
    assert app.df is None
    assert "Formato não suportado" in caplog.text


def test_criar_coluna_error_keeps_the_chain(clf_df, caplog):
    app = Sanice(clf_df, lang="en")
    assert app.criar_coluna("z", "renda +") is app
    assert "Failed to create column" in caplog.text


def test_scale_excludes_target_and_prediction_matches(clf_df):
    app = Sanice(clf_df)
    app.escalonar(excluir=["pago"]).auto_ml(alvo="pago")
    assert "pago" not in app.colunas_escala
    assert set(app.df["pago"].unique()) == {0, 1}
    app.prever()
    assert (app.df["previsao"] == app.df["pago"]).mean() > 0.8


def test_auto_ml_makes_model_ready_without_saving(clf_df):
    app = Sanice(clf_df).auto_ml(alvo="pago")
    assert app.modelo_ativo is not None
    app.prever("pred")
    assert "pred" in app.df.columns
    assert app.metricas["accuracy"] > 0.8


def test_save_load_predict_on_new_data(clf_df, tmp_path):
    caminho = str(tmp_path / "m.pkl")
    Sanice(clf_df).escalonar(excluir="pago").auto_ml(alvo="pago", salvar_modelo=caminho)
    novos = clf_df.drop(columns=["pago"]).head(20)
    app = Sanice(novos).carregar_ia(caminho).prever()
    assert len(app.df["previsao"]) == 20
    assert app.metricas


def test_prediction_with_a_single_category_row(clf_df):
    app = Sanice(clf_df).auto_ml(alvo="pago")
    X = app._preparar(pd.DataFrame([{"renda": 9000, "idade": 30, "cidade": "RJ"}]), escalar=True)
    assert X.loc[0, "cidade_RJ"] == 1.0
    assert X.loc[0, "cidade_SP"] == 0.0


def test_task_type_detection(clf_df, reg_df):
    assert Sanice(clf_df).auto_ml(alvo="pago").classes == [0, 1]
    assert Sanice(reg_df).auto_ml(alvo="y").classes is None


def test_balanced_and_cross_validation_and_metric(clf_df):
    app = Sanice(clf_df).auto_ml(alvo="pago", balancear=True, validacao_cruzada=3, metrica="f1")
    assert app.modelo_ativo is not None


def test_regression_metrics(reg_df):
    app = Sanice(reg_df).auto_ml(alvo="y", tipo="regressao", metrica="rmse")
    assert set(app.metricas) == {"r2", "rmse", "mae"}
    assert app.metricas["r2"] > 0.9


def test_nulls_do_not_break_training(clf_df):
    clf_df.loc[::7, "renda"] = np.nan
    clf_df.loc[::11, "cidade"] = None
    app = Sanice(clf_df).auto_ml(alvo="pago")
    assert app.modelo_ativo is not None


def test_predict_proba_importance_and_evaluate(clf_df, capsys):
    app = Sanice(clf_df).auto_ml(alvo="pago").prever_proba()
    assert {"prob_0", "prob_1"} <= set(app.df.columns)
    assert np.allclose(app.df["prob_0"] + app.df["prob_1"], 1)
    app.importancia(top=2).avaliar()
    saida = capsys.readouterr().out
    assert "renda" in saida or "idade" in saida
    assert "Acurácia" in saida
    assert app.importancias is not None


def test_predict_without_model_is_reported(clf_df, caplog):
    app = Sanice(clf_df, lang="en")
    assert app.prever() is app
    assert "Train with .auto_ml()" in caplog.text


def test_duplicates_rename_nulls_info_sample(capsys):
    df = pd.DataFrame({"a": [1, 1, 2, None], "b": ["x", "x", "y", "z"]})
    app = Sanice(df)
    app.remover_duplicados()
    assert len(app.df) == 3
    app.renomear({"a": "valor", "nao_existe": "x"})
    assert "valor" in app.df.columns
    app.nulos().informacoes()
    saida = capsys.readouterr().out
    assert "valor" in saida
    app.amostra(n=2)
    assert len(app.df) == 2


def test_validate_documents():
    assert _cpf_valido("529.982.247-25")
    assert not _cpf_valido("111.111.111-11")
    assert not _cpf_valido("529.982.247-24")
    assert _cnpj_valido("11.222.333/0001-81")
    assert not _cnpj_valido("11.222.333/0001-82")
    df = pd.DataFrame({
        "cpf": ["529.982.247-25", "123", None],
        "email": ["a@b.com", "x", "c@d.org"],
        "cep": ["01310-100", "123", "30140071"],
    })
    app = Sanice(df).validar("cpf", "CPF").validar("email", "EMAIL").validar("cep", "CEP")
    assert app.df["cpf_valido"].tolist() == [True, False, False]
    assert app.df["email_valido"].tolist() == [True, False, True]
    assert app.df["cep_valido"].tolist() == [True, False, True]
    app.validar("cpf", "CPF", remover=True)
    assert len(app.df) == 1


def test_validate_unknown_rule(caplog):
    Sanice(pd.DataFrame({"a": [1]}), lang="en").validar("a", "XYZ")
    assert "Unknown validation rule" in caplog.text


def test_new_transform_rules():
    df = pd.DataFrame({
        "cpf": [12345678909.0, None],
        "cep": [1310100, 30140071],
        "uf": [" sp", "ba"],
        "data": ["31/12/2024", "01/02/2025"],
        "nome": ["ana", None],
    })
    app = Sanice(df)
    app.transformar("cpf", "CPF").transformar("cep", "CEP").transformar("uf", "UF")
    app.transformar("data", "DATE").transformar("nome", "UPPER")
    assert app.df["cpf"].iloc[0] == "12345678909"
    assert pd.isna(app.df["cpf"].iloc[1])
    assert app.df["cep"].tolist() == ["01310100", "30140071"]
    assert app.df["uf"].tolist() == ["SP", "BA"]
    assert app.df["data"].iloc[0] == pd.Timestamp("2024-12-31")
    assert app.df["nome"].iloc[0] == "ANA"
    assert pd.isna(app.df["nome"].iloc[1])
    app.transformar("uf", "REGIAO")
    assert app.df["uf"].tolist() == ["Sudeste", "Nordeste"]


def test_unknown_rule_is_reported(caplog):
    Sanice(pd.DataFrame({"a": [1]}), lang="en").transformar("a", "NADA")
    assert "unknown" in caplog.text


def test_extract_date_and_lags():
    df = pd.DataFrame({"data": ["01/03/2024", "02/03/2024", "03/03/2024", "04/03/2024"], "v": [1.0, 2.0, 3.0, 4.0]})
    app = Sanice(df).extrair_data("data").criar_atrasos("v", atrasos=[1], medias=[2])
    assert app.df["data_mes"].tolist() == [3, 3, 3, 3]
    assert app.df["data_dia"].tolist() == [1, 2, 3, 4]
    assert pd.isna(app.df["v_atraso1"].iloc[0])
    assert app.df["v_atraso1"].iloc[1] == 1.0
    assert app.df["v_media2"].iloc[3] == 2.5


def test_detect_anomalies_and_segments():
    rng = np.random.default_rng(3)
    valores = np.append(rng.normal(100, 5, 200), [900, 950])
    df = pd.DataFrame({"valor": valores, "outro": np.append(rng.normal(0, 1, 200), [30, 40])})
    app = Sanice(df).detectar_anomalias(contaminacao=0.02)
    assert app.df["anomalia"].iloc[-1] and app.df["anomalia"].iloc[-2]
    app.segmentar(colunas=["valor", "outro"], grupos=2)
    assert app.df["segmento"].nunique() == 2


def test_undo_restores_previous_state():
    app = Sanice(pd.DataFrame({"a": [1, 2, 3]}))
    app.filtrar("a > 1")
    assert len(app.df) == 2
    app.desfazer()
    assert len(app.df) == 3
    app.desfazer()


def test_undo_empty_is_reported(caplog):
    Sanice(pd.DataFrame({"a": [1]}), lang="en").desfazer()
    assert "Nothing to undo" in caplog.text


def test_group_and_pivot_use_the_chosen_language(capsys, caplog):
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    app = Sanice(df, lang="en")
    app.group_by("g", "v", "soma")
    app.pivot_table("g", None, "v")
    assert "Grouped by g" in caplog.text


def test_remove_nulls_accepts_english_strategy():
    app = Sanice(pd.DataFrame({"a": [1, None]}))
    app.remover_nulos("fill", 9)
    assert app.df["a"].tolist() == [1, 9]


def test_save_json_and_reports(clf_df, tmp_path):
    app = Sanice(clf_df).corrigir_colunas().auto_ml(alvo="pago")
    app.salvar(str(tmp_path / "d.json"))
    assert pd.read_json(tmp_path / "d.json").shape[0] == len(clf_df)
    app.salvar_relatorio(str(tmp_path / "r.html"))
    conteudo = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "Relatório Sanice" in conteudo and "Passos executados" in conteudo
    app.salvar_relatorio(str(tmp_path / "r.xlsx"))
    assert set(pd.ExcelFile(tmp_path / "r.xlsx").sheet_names) == {"Resumo", "Passos", "Modelo", "Dados"}


def test_unknown_save_format_is_reported(caplog):
    Sanice(pd.DataFrame({"a": [1]}), lang="en").salvar("x.abc")
    assert "Unsupported format" in caplog.text


def test_mongo_missing_dependency_is_reported(caplog):
    with patch.dict(sys.modules, {"pymongo": None}):
        Sanice(pd.DataFrame({"a": [1]}), lang="en").exportar_mongo("mongodb://x", "d", "c")
    assert "Install pymongo" in caplog.text


def test_api_serves_predictions(clf_df):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    app = Sanice(clf_df).escalonar(excluir="pago").auto_ml(alvo="pago")
    capturado = {}
    with patch("uvicorn.run", side_effect=lambda api, **kw: capturado.update(api=api, **kw)):
        app.servir_api(host="0.0.0.0", porta=9000)
    assert capturado["host"] == "0.0.0.0" and capturado["port"] == 9000
    cliente = TestClient(capturado["api"])
    resposta = cliente.post("/predict", json={"renda": 9000, "idade": 40, "cidade": "SP"})
    assert resposta.status_code == 200
    assert resposta.json()["predicao"] in (0, 1)
    assert set(resposta.json()["probabilidades"]) == {"0", "1"}
    assert cliente.post("/predict", json={"renda": 9000}).status_code == 422


def test_api_without_model_is_reported(clf_df, caplog):
    Sanice(clf_df, lang="en").servir_api()
    assert "Train with .auto_ml()" in caplog.text


def test_cli_lists_new_commands(capsys):
    from sanice.core import cli
    sys.argv = ["sanice", "help"]
    cli()
    saida = capsys.readouterr().out
    assert "detectar_anomalias" in saida and "salvar_relatorio" in saida
