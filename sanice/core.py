# Copyright 2025 w.Sanice
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import html
import logging
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import unidecode
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    get_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sqlalchemy import create_engine

logger = logging.getLogger('Sanice')
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

REGIOES = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
    "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}

PESOS_CNPJ = {
    12: [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2],
    13: [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2],
}


def _so_digitos(valor, tamanho=None):
    if isinstance(valor, float) and valor.is_integer():
        valor = int(valor)
    digitos = re.sub(r"\D", "", str(valor))
    if tamanho and not isinstance(valor, str):
        digitos = digitos.zfill(tamanho)
    return digitos


def _cpf_valido(valor):
    d = _so_digitos(valor, 11)
    if len(d) != 11 or d == d[0] * 11:
        return False
    for n in (9, 10):
        soma = sum(int(d[i]) * (n + 1 - i) for i in range(n))
        if (soma * 10 % 11) % 10 != int(d[n]):
            return False
    return True


def _cnpj_valido(valor):
    d = _so_digitos(valor, 14)
    if len(d) != 14 or d == d[0] * 14:
        return False
    for n in (12, 13):
        resto = sum(int(d[i]) * PESOS_CNPJ[n][i] for i in range(n)) % 11
        if (0 if resto < 2 else 11 - resto) != int(d[n]):
            return False
    return True


def _email_valido(valor):
    return re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", str(valor).strip()) is not None


def _cep_valido(valor):
    return len(_so_digitos(valor, 8)) == 8


VALIDADORES = {
    "CPF": _cpf_valido,
    "CNPJ": _cnpj_valido,
    "EMAIL": _email_valido,
    "CEP": _cep_valido,
}


def _mutante(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self._snapshot()
        return func(self, *args, **kwargs)
    return wrapper


class Sanice:
    """
    S.A.N.I.C.E.
    (Sistema Automatizado de Normalização, Inteligência Computacional e Estatística)
    -----------------------------------------------------------------------------
    Framework para limpar, transformar e modelar dados automaticamente.
    """
    I18N = {
        "pt": {
            "auto_date": "[SMART] Data detectada e convertida na coluna: '{col}'",
            "auto_mem": "[SMART] Memória otimizada! {n} colunas convertidas para 'category'.",
            "mongo_ok": "[MONGO] Dados exportados para coleção '{col}' com sucesso.",
            "sql_read_ok": "[SQL] Li {rows} linhas da consulta SQL.",
            "load_ok": "[CARREGAR] Dados carregados: {rows} linhas x {cols} colunas.",
            "load_err": "[ERRO] Falha ao carregar: {e}",
            "view": "\n[VISUALIZAR] {header}:",
            "clean_cols": "[LIMPEZA] Nomes das colunas padronizados.",
            "clean_txt": "[LIMPEZA] Texto da coluna '{col}' normalizado.",
            "drop_null": "[REMOVER] {qtd} linhas com nulos removidas.",
            "fill_null": "[PREENCHER] Nulos preenchidos com '{val}'.",
            "date_conv": "[DATA] Coluna '{col}' convertida para data.",
            "col_add": "[CRIAR] Coluna '{col}' criada.",
            "filter": "[FILTRO] '{query}': {before} -> {after} linhas.",
            "sort": "[ORDENAR] Ordenado por {cols}.",
            "join": "[UNIR] Tabelas unidas ({how}). Linhas: {before} -> {after}",
            "plot_err": "[ERRO] Falha ao plotar: {e}",
            "stats": "\n[ESTATÍSTICAS] Resumo Estatístico:",
            "types": "\n[TIPOS] Tipos de Dados:",
            "save": "[SALVAR] Arquivo salvo em: {path}",
            "ml_start": "\n[AUTO-ML] Iniciando treinamento para prever: '{target}'...",
            "ml_ignore_date": "   [INFO] Ignorando colunas de data crua: {cols}",
            "ml_feats": "   - Features processadas: {n} colunas (após encoding).",
            "ml_success_clf": "   Modelo Classificador treinado!",
            "ml_success_reg": "   Modelo Regressor treinado!",
            "ml_acc": "   Acurácia: {score:.2%}",
            "ml_r2": "   R² Score: {score:.4f}",
            "ml_saved": "   Modelo salvo em: {path}",
            "ia_loaded": "[MODELO] Modelo carregado! Espera {n} colunas.",
            "pred_done": "[PREVISÃO] Previsões geradas na coluna '{col}'.",
            "err_load_ia": "[ERRO] Treine com .auto_ml() ou use .carregar_ia() antes de prever!",
            "sql_ok": "[SQL] Tabela '{tb}' exportada com sucesso para o banco.",
            "sql_err": "[SQL] Erro ao exportar: {e}",
            "scale_ok": "[ESCALA] Dados normalizados usando '{method}'.",
            "outlier_rem": "[OUTLIERS] {qtd} outliers removidos (Método IQR).",
            "api_start": "[API] Servidor iniciado em http://{host}:{porta}/docs",
            "trans_money": "[TRANSFORMAR] '{col}' convertida para Moeda (float).",
            "trans_num": "[TRANSFORMAR] '{col}' limpa (apenas dígitos).",
            "trans_email": "[TRANSFORMAR] '{col}' normalizada para E-mail.",
            "trans_date": "[TRANSFORMAR] '{col}' convertida para Data.",
            "trans_err": "[ERRO] Regra '{rule}' desconhecida ou falha.",
            "select_ok": "[SELEÇÃO] Mantidas {n} colunas.",
            "select_warn": "[AVISO] Colunas não encontradas e ignoradas: {cols}",
            "help_title": "\n[AJUDA] Comandos disponíveis em '{lang}':\n",
            "ml_tourn": "   [AUTO-ML] Avaliando {n} modelos (Linear, RF, Gradient)...",
            "ml_win": "   [RESULTADO] Melhor modelo: {name} | {metric}: {score:.4f}",
            "ml_fail": "   [ERRO] Falha no modelo {name}: {e}",
            "fmt_err": "[ERRO] Formato não suportado: '{fonte}'.",
            "load_enc": "[CARREGAR] Codificação: {enc} | Separador: '{sep}'",
            "col_err": "[ERRO] Falha ao criar a coluna: {e}",
            "filter_err": "[ERRO] Consulta inválida: {e}",
            "save_err": "[ERRO] Falha ao salvar: {e}",
            "group_err": "[ERRO] Falha ao agrupar: {e}",
            "group_header": "Agrupamento por {cols} ({op})",
            "pivot_err": "[ERRO] Falha na tabela dinâmica: {e}",
            "pivot_header": "Tabela Dinâmica",
            "mongo_err": "[MONGO] Erro ao exportar: {e}",
            "mongo_missing": "[MONGO] Instale o pymongo: pip install pymongo",
            "ml_err": "[ERRO] Falha no treinamento: {e}",
            "ml_no_target": "[ERRO] Informe o alvo: .auto_ml(alvo='coluna')",
            "ml_kind_clf": "   Tipo detectado: classificação",
            "ml_kind_reg": "   Tipo detectado: regressão",
            "ml_balance": "   Classes balanceadas por peso.",
            "ml_cv": "   Validação cruzada com {k} partes.",
            "ml_metrics": "   Métricas no teste: {texto}",
            "target_scaled": "[AVISO] O alvo '{alvo}' foi escalado. Use .escalonar(excluir=['{alvo}']).",
            "model_load_err": "[ERRO] Falha ao carregar o modelo: {e}",
            "pred_err": "[ERRO] Falha na previsão: {e}",
            "proba_unsupported": "[ERRO] Este modelo não calcula probabilidades.",
            "proba_done": "[PROBABILIDADE] Colunas criadas: {cols}",
            "scale_skip": "[AVISO] Escala não aplicada. Faltam as colunas: {cols}",
            "api_missing": "[API] Instale as dependências: pip install sanice[api]",
            "api_err": "[API] Erro: {e}",
            "dup_removed": "[DUPLICADOS] {qtd} linhas duplicadas removidas.",
            "rename_ok": "[RENOMEAR] {n} colunas renomeadas.",
            "null_title": "\n[NULOS] Porcentagem de nulos por coluna:",
            "null_none": "[NULOS] Nenhum valor nulo encontrado.",
            "info_title": "\n[INFO] {rows} linhas x {cols} colunas | Memória: {mem:.2f} MB",
            "sample_ok": "[AMOSTRA] {n} linhas sorteadas.",
            "validate_ok": "[VALIDAR] '{col}' ({rule}): {invalidos} inválidos de {total}.",
            "validate_unknown": "[ERRO] Regra de validação desconhecida: '{rule}'.",
            "date_extract": "[DATA] Colunas criadas a partir de '{col}': {cols}",
            "lag_ok": "[ATRASOS] Colunas criadas para '{col}': {cols}",
            "anom_ok": "[ANOMALIAS] {qtd} anomalias marcadas na coluna '{col}'.",
            "seg_ok": "[SEGMENTOS] {n} segmentos criados na coluna '{col}'.",
            "no_numeric": "[ERRO] Nenhuma coluna numérica disponível.",
            "imp_title": "\n[IMPORTÂNCIA] Variáveis que mais pesam no modelo:",
            "imp_unsupported": "[ERRO] Este modelo não informa a importância das variáveis.",
            "no_metrics": "[AVISO] Nenhuma métrica disponível. Treine com .auto_ml() primeiro.",
            "eval_title": "\n[AVALIAÇÃO] {nome}:",
            "metric_accuracy": "Acurácia",
            "metric_f1": "F1",
            "metric_r2": "R²",
            "metric_rmse": "RMSE",
            "metric_mae": "MAE",
            "report_ok": "[RELATÓRIO] Relatório salvo em: {path}",
            "rep_title": "Relatório Sanice",
            "rep_steps": "Passos executados",
            "rep_summary": "Resumo estatístico",
            "rep_model": "Modelo",
            "rep_data": "Primeiras linhas",
            "undo_ok": "[DESFAZER] Última operação desfeita.",
            "undo_empty": "[AVISO] Nada para desfazer.",
            "log_level": "[LOG] Sanice configurado para o nível: {nivel}",
            "trans_text": "[TRANSFORMAR] '{col}' padronizada.",
            "trans_region": "[TRANSFORMAR] '{col}' convertida de UF para região.",
        },
        "en": {
            "auto_date": "[SMART] Date detected and converted in column: '{col}'",
            "auto_mem": "[SMART] Memory optimized! {n} columns converted to 'category'.",
            "mongo_ok": "[MONGO] Data exported to collection '{col}' successfully.",
            "sql_read_ok": "[SQL] Read {rows} rows from SQL query.",
            "load_ok": "[LOAD] Data loaded: {rows} rows x {cols} cols.",
            "load_err": "[ERROR] Failed to load: {e}",
            "view": "\n[VIEW] {header}:",
            "clean_cols": "[CLEAN] Column names standardized.",
            "clean_txt": "[CLEAN] Text in column '{col}' normalized.",
            "drop_null": "[DROP] {qtd} rows with nulls removed.",
            "fill_null": "[FILL] Nulls filled with '{val}'.",
            "date_conv": "[DATE] Column '{col}' converted to datetime.",
            "col_add": "[ADD] Column '{col}' created.",
            "filter": "[FILTER] '{query}': {before} -> {after} rows.",
            "sort": "[SORT] Sorted by {cols}.",
            "join": "[JOIN] Tables merged ({how}). Rows: {before} -> {after}",
            "plot_err": "[ERROR] Failed to plot: {e}",
            "stats": "\n[STATS] Statistical Summary:",
            "types": "\n[TYPES] Data Types:",
            "save": "[SAVE] File saved at: {path}",
            "ml_start": "\n[AUTO-ML] Starting training to predict: '{target}'...",
            "ml_ignore_date": "   [INFO] Ignoring raw date columns: {cols}",
            "ml_feats": "   - Features processed: {n} cols (after encoding).",
            "ml_success_clf": "   Classifier Model trained!",
            "ml_success_reg": "   Regressor Model trained!",
            "ml_acc": "   Accuracy: {score:.2%}",
            "ml_r2": "   R² Score: {score:.4f}",
            "ml_saved": "   Model saved at: {path}",
            "ia_loaded": "[MODEL] Model loaded! Expects {n} columns.",
            "pred_done": "[PREDICT] Predictions generated in column '{col}'.",
            "err_load_ia": "[ERROR] Train with .auto_ml() or use .load_ai() before predicting!",
            "sql_ok": "[SQL] Table '{tb}' successfully exported to database.",
            "sql_err": "[SQL] Error exporting: {e}",
            "scale_ok": "[SCALE] Data normalized using '{method}'.",
            "outlier_rem": "[OUTLIERS] {qtd} outliers removed (IQR Method).",
            "api_start": "[API] Server started at http://{host}:{porta}/docs",
            "trans_money": "[TRANSFORM] '{col}' converted to Currency (float).",
            "trans_num": "[TRANSFORM] '{col}' cleaned (digits only).",
            "trans_email": "[TRANSFORM] '{col}' normalized to E-mail.",
            "trans_date": "[TRANSFORM] '{col}' converted to Date.",
            "trans_err": "[ERROR] Rule '{rule}' unknown or failed.",
            "help_title": "\n[HELP] Available commands in '{lang}':\n",
            "select_ok": "[SELECT] Kept {n} columns.",
            "select_warn": "[WARN] Columns not found and ignored: {cols}",
            "ml_tourn": "   [AUTO-ML] Evaluating {n} models (Linear, RF, Gradient)...",
            "ml_win": "   [RESULT] Best model: {name} | {metric}: {score:.4f}",
            "ml_fail": "   [ERROR] Model {name} failed: {e}",
            "fmt_err": "[ERROR] Unsupported format: '{fonte}'.",
            "load_enc": "[LOAD] Encoding: {enc} | Separator: '{sep}'",
            "col_err": "[ERROR] Failed to create column: {e}",
            "filter_err": "[ERROR] Invalid query: {e}",
            "save_err": "[ERROR] Failed to save: {e}",
            "group_err": "[ERROR] Failed to group: {e}",
            "group_header": "Grouped by {cols} ({op})",
            "pivot_err": "[ERROR] Pivot table failed: {e}",
            "pivot_header": "Pivot Table",
            "mongo_err": "[MONGO] Export error: {e}",
            "mongo_missing": "[MONGO] Install pymongo: pip install pymongo",
            "ml_err": "[ERROR] Training failed: {e}",
            "ml_no_target": "[ERROR] Set the target: .auto_ml(target='column')",
            "ml_kind_clf": "   Detected type: classification",
            "ml_kind_reg": "   Detected type: regression",
            "ml_balance": "   Classes balanced by weight.",
            "ml_cv": "   Cross-validation with {k} folds.",
            "ml_metrics": "   Test metrics: {texto}",
            "target_scaled": "[WARN] Target '{alvo}' was scaled. Use .scale_data(exclude=['{alvo}']).",
            "model_load_err": "[ERROR] Failed to load the model: {e}",
            "pred_err": "[ERROR] Prediction failed: {e}",
            "proba_unsupported": "[ERROR] This model does not compute probabilities.",
            "proba_done": "[PROBABILITY] Columns created: {cols}",
            "scale_skip": "[WARN] Scaling skipped. Missing columns: {cols}",
            "api_missing": "[API] Install the dependencies: pip install sanice[api]",
            "api_err": "[API] Error: {e}",
            "dup_removed": "[DUPLICATES] {qtd} duplicate rows removed.",
            "rename_ok": "[RENAME] {n} columns renamed.",
            "null_title": "\n[NULLS] Percentage of nulls per column:",
            "null_none": "[NULLS] No null values found.",
            "info_title": "\n[INFO] {rows} rows x {cols} cols | Memory: {mem:.2f} MB",
            "sample_ok": "[SAMPLE] {n} rows drawn.",
            "validate_ok": "[VALIDATE] '{col}' ({rule}): {invalidos} invalid out of {total}.",
            "validate_unknown": "[ERROR] Unknown validation rule: '{rule}'.",
            "date_extract": "[DATE] Columns created from '{col}': {cols}",
            "lag_ok": "[LAGS] Columns created for '{col}': {cols}",
            "anom_ok": "[ANOMALIES] {qtd} anomalies flagged in column '{col}'.",
            "seg_ok": "[SEGMENTS] {n} segments created in column '{col}'.",
            "no_numeric": "[ERROR] No numeric columns available.",
            "imp_title": "\n[IMPORTANCE] Variables that weigh most in the model:",
            "imp_unsupported": "[ERROR] This model does not report variable importance.",
            "no_metrics": "[WARN] No metrics available. Train with .auto_ml() first.",
            "eval_title": "\n[EVALUATION] {nome}:",
            "metric_accuracy": "Accuracy",
            "metric_f1": "F1",
            "metric_r2": "R²",
            "metric_rmse": "RMSE",
            "metric_mae": "MAE",
            "report_ok": "[REPORT] Report saved at: {path}",
            "rep_title": "Sanice Report",
            "rep_steps": "Steps performed",
            "rep_summary": "Statistical summary",
            "rep_model": "Model",
            "rep_data": "First rows",
            "undo_ok": "[UNDO] Last operation undone.",
            "undo_empty": "[WARN] Nothing to undo.",
            "log_level": "[LOG] Sanice configured to level: {nivel}",
            "trans_text": "[TRANSFORM] '{col}' standardized.",
            "trans_region": "[TRANSFORM] '{col}' converted from state code to region.",
        },
        "zh": {
            "auto_date": "[智能] 检测到日期并已转换列：'{col}'",
            "auto_mem": "[智能] 内存已优化！{n} 列已转换为 'category'。",
            "mongo_ok": "[MONGO] 数据已成功导出到集合 '{col}'。",
            "sql_read_ok": "[SQL] 从 SQL 查询中读取了 {rows} 行。",
            "load_ok": "[加载] 数据已加载：{rows} 行 x {cols} 列。",
            "load_err": "[错误] 加载失败：{e}",
            "view": "\n[查看] {header}：",
            "clean_cols": "[清洗] 列名已标准化。",
            "clean_txt": "[清洗] 列 '{col}' 的文本已规范化。",
            "drop_null": "[移除] 已移除 {qtd} 行空值。",
            "fill_null": "[填充] 空值已填充为 '{val}'。",
            "date_conv": "[日期] 列 '{col}' 已转换为日期格式。",
            "col_add": "[创建] 列 '{col}' 已创建。",
            "filter": "[过滤] '{query}': {before} -> {after} 行。",
            "sort": "[排序] 按 {cols} 排序。",
            "join": "[合并] 表格已合并 ({how})。行数: {before} -> {after}",
            "plot_err": "[错误] 绘图失败：{e}",
            "stats": "\n[统计] 统计摘要：",
            "types": "\n[类型] 数据类型：",
            "save": "[保存] 文件已保存至：{path}",
            "ml_start": "\n[自动机器学习] 开始训练预测：'{target}'...",
            "ml_ignore_date": "   [信息] 忽略原始日期列：{cols}",
            "ml_feats": "   - 特征处理：{n} 列（编码后）。",
            "ml_success_clf": "   分类模型训练完成！",
            "ml_success_reg": "   回归模型训练完成！",
            "ml_acc": "   准确率：{score:.2%}",
            "ml_r2": "   R² 分数：{score:.4f}",
            "ml_saved": "   模型已保存至：{path}",
            "ia_loaded": "[模型] 模型已加载！需要 {n} 列。",
            "pred_done": "[预测] 预测结果已生成在 '{col}' 列。",
            "err_load_ia": "[错误] 预测前请先使用 .auto_ml() 训练或 .load_ai() 加载模型！",
            "sql_ok": "[SQL] 表 '{tb}' 已成功导出到数据库。",
            "sql_err": "[SQL] 导出错误：{e}",
            "scale_ok": "[缩放] 数据已使用 '{method}' 标准化。",
            "outlier_rem": "[异常值] 已移除 {qtd} 个异常值 (IQR 方法)。",
            "api_start": "[API] 服务器已启动 http://{host}:{porta}/docs",
            "trans_money": "[转换] '{col}' 已转换为货币 (float)。",
            "trans_num": "[转换] '{col}' 已清洗 (仅数字)。",
            "trans_email": "[转换] '{col}' 已标准化为电子邮件。",
            "trans_date": "[转换] '{col}' 已转换为日期。",
            "trans_err": "[错误] 规则 '{rule}' 未知或失败。",
            "help_title": "\n[帮助] '{lang}' 可用命令：\n",
            "select_ok": "[选择] 保留了 {n} 列。",
            "select_warn": "[警告] 未找到并已忽略的列：{cols}",
            "ml_tourn": "   [AUTO-ML] 正在评估 {n} 个模型...",
            "ml_win": "   [结果] 最佳模型: {name} ({metric}: {score:.4f})",
            "ml_fail": "   [错误] 模型 {name} 失败: {e}",
            "fmt_err": "[错误] 不支持的格式：'{fonte}'。",
            "load_enc": "[加载] 编码：{enc} | 分隔符：'{sep}'",
            "col_err": "[错误] 创建列失败：{e}",
            "filter_err": "[错误] 查询无效：{e}",
            "save_err": "[错误] 保存失败：{e}",
            "group_err": "[错误] 分组失败：{e}",
            "group_header": "按 {cols} 分组 ({op})",
            "pivot_err": "[错误] 透视表失败：{e}",
            "pivot_header": "透视表",
            "mongo_err": "[MONGO] 导出错误：{e}",
            "mongo_missing": "[MONGO] 请安装 pymongo：pip install pymongo",
            "ml_err": "[错误] 训练失败：{e}",
            "ml_no_target": "[错误] 请指定目标列：.auto_ml(target='列名')",
            "ml_kind_clf": "   检测到的类型：分类",
            "ml_kind_reg": "   检测到的类型：回归",
            "ml_balance": "   已按权重平衡类别。",
            "ml_cv": "   使用 {k} 折交叉验证。",
            "ml_metrics": "   测试集指标：{texto}",
            "target_scaled": "[警告] 目标列 '{alvo}' 已被缩放。请使用 .scale_data(exclude=['{alvo}'])。",
            "model_load_err": "[错误] 加载模型失败：{e}",
            "pred_err": "[错误] 预测失败：{e}",
            "proba_unsupported": "[错误] 此模型无法计算概率。",
            "proba_done": "[概率] 已创建列：{cols}",
            "scale_skip": "[警告] 未应用缩放，缺少列：{cols}",
            "api_missing": "[API] 请安装依赖：pip install sanice[api]",
            "api_err": "[API] 错误：{e}",
            "dup_removed": "[重复] 已移除 {qtd} 行重复数据。",
            "rename_ok": "[重命名] 已重命名 {n} 列。",
            "null_title": "\n[空值] 各列空值百分比：",
            "null_none": "[空值] 未发现空值。",
            "info_title": "\n[信息] {rows} 行 x {cols} 列 | 内存：{mem:.2f} MB",
            "sample_ok": "[抽样] 已随机抽取 {n} 行。",
            "validate_ok": "[验证] '{col}' ({rule})：{total} 条中有 {invalidos} 条无效。",
            "validate_unknown": "[错误] 未知的验证规则：'{rule}'。",
            "date_extract": "[日期] 已从 '{col}' 创建列：{cols}",
            "lag_ok": "[滞后] 已为 '{col}' 创建列：{cols}",
            "anom_ok": "[异常] 已在列 '{col}' 中标记 {qtd} 个异常。",
            "seg_ok": "[分群] 已在列 '{col}' 中创建 {n} 个分群。",
            "no_numeric": "[错误] 没有可用的数值列。",
            "imp_title": "\n[重要性] 对模型影响最大的变量：",
            "imp_unsupported": "[错误] 此模型不提供变量重要性。",
            "no_metrics": "[警告] 暂无指标，请先使用 .auto_ml() 训练。",
            "eval_title": "\n[评估] {nome}：",
            "metric_accuracy": "准确率",
            "metric_f1": "F1",
            "metric_r2": "R²",
            "metric_rmse": "RMSE",
            "metric_mae": "MAE",
            "report_ok": "[报告] 报告已保存至：{path}",
            "rep_title": "Sanice 报告",
            "rep_steps": "执行的步骤",
            "rep_summary": "统计摘要",
            "rep_model": "模型",
            "rep_data": "前几行数据",
            "undo_ok": "[撤销] 已撤销上一步操作。",
            "undo_empty": "[警告] 没有可撤销的操作。",
            "log_level": "[日志] Sanice 日志级别已设置为：{nivel}",
            "trans_text": "[转换] '{col}' 已标准化。",
            "trans_region": "[转换] '{col}' 已从州代码转换为地区。",
        },
        "hi": {
            "auto_date": "[SMART] '{col}' mein date mili aur convert ho gayi.",
            "auto_mem": "[SMART] Memory bachayi gayi! {n} columns 'category' ban gaye.",
            "mongo_ok": "[MONGO] Data '{col}' collection mein export ho gaya.",
            "sql_read_ok": "[SQL] SQL query se {rows} rows padhe gaye.",
            "load_ok": "[LOAD] Data load ho gaya: {rows} rows x {cols} cols.",
            "load_err": "[ERROR] Load karne mein fail: {e}",
            "view": "\n[DEKHE] {header}:",
            "clean_cols": "[SAFAI] Column ke naam standardize kiye gaye.",
            "clean_txt": "[SAFAI] Column '{col}' ka text theek kiya gaya.",
            "drop_null": "[HATAYE] {qtd} rows null hataye gaye.",
            "fill_null": "[BHARE] Nulls ko '{val}' se bhara gaya.",
            "date_conv": "[TARIKH] Column '{col}' date mein badla gaya.",
            "col_add": "[BANAYE] Column '{col}' banaya gaya.",
            "filter": "[FILTER] '{query}': {before} -> {after} rows.",
            "sort": "[SORT] {cols} ke hisaab se sort kiya.",
            "join": "[JODE] Tables jode gaye ({how}). Rows: {before} -> {after}",
            "plot_err": "[ERROR] Plot karne mein fail: {e}",
            "stats": "\n[STATS] Sankhyiki Saar:",
            "types": "\n[TYPES] Data Types:",
            "save": "[SAVE] File save kiya gaya: {path}",
            "ml_start": "\n[AUTO-ML] Training shuru: '{target}'...",
            "ml_ignore_date": "   [INFO] Raw date columns ignore kar rahe hain: {cols}",
            "ml_feats": "   - Features processed: {n} cols (encoding ke baad).",
            "ml_success_clf": "   Model train ho gaya (Classifier)!",
            "ml_success_reg": "   Model train ho gaya (Regressor)!",
            "ml_acc": "   Accuracy: {score:.2%}",
            "ml_r2": "   R² Score: {score:.4f}",
            "ml_saved": "   Model save kiya gaya: {path}",
            "ia_loaded": "[MODEL] Model load ho gaya! {n} columns chahiye.",
            "pred_done": "[PREDICT] Predictions '{col}' column mein likhe gaye.",
            "err_load_ia": "[ERROR] Predict karne se pehle .auto_ml() se train karein ya .load_ai() use karein!",
            "sql_ok": "[SQL] Table '{tb}' database mein export ho gaya.",
            "sql_err": "[SQL] Export mein galti: {e}",
            "scale_ok": "[SCALE] Data '{method}' se normalize kiya gaya.",
            "outlier_rem": "[OUTLIERS] {qtd} outliers hataye gaye (IQR Method).",
            "api_start": "[API] Server shuru hua http://{host}:{porta}/docs par",
            "trans_money": "[BADLAV] '{col}' currency (float) mein badla gaya.",
            "trans_num": "[BADLAV] '{col}' saaf kiya gaya (keval ank).",
            "trans_email": "[BADLAV] '{col}' E-mail ke liye theek kiya gaya.",
            "trans_date": "[BADLAV] '{col}' Tarikh mein badla gaya.",
            "trans_err": "[ERROR] Rule '{rule}' galat hai ya fail ho gayi.",
            "help_title": "\n[MADAD] '{lang}' mein commands:\n",
            "select_ok": "[CHUNNA] {n} columns rakhi gayin.",
            "select_warn": "[CHETAVANI] Columns nahi mili aur ignore ki gayi: {cols}",
            "ml_tourn": "   [AUTO-ML] {n} models ka mulyankan (evaluation) ho raha hai...",
            "ml_win": "   [NATIJA] Behtarin model: {name} ({metric}: {score:.4f})",
            "ml_fail": "   [GALTI] Model {name} fail hua: {e}",
            "fmt_err": "[ERROR] Yeh format support nahi hota: '{fonte}'.",
            "load_enc": "[LOAD] Encoding: {enc} | Separator: '{sep}'",
            "col_err": "[ERROR] Column banane mein fail: {e}",
            "filter_err": "[ERROR] Query galat hai: {e}",
            "save_err": "[ERROR] Save karne mein fail: {e}",
            "group_err": "[ERROR] Group karne mein fail: {e}",
            "group_header": "{cols} ke hisaab se group ({op})",
            "pivot_err": "[ERROR] Pivot table fail hui: {e}",
            "pivot_header": "Pivot Table",
            "mongo_err": "[MONGO] Export mein galti: {e}",
            "mongo_missing": "[MONGO] pymongo install karein: pip install pymongo",
            "ml_err": "[ERROR] Training fail hui: {e}",
            "ml_no_target": "[ERROR] Target batayein: .auto_ml(target='column')",
            "ml_kind_clf": "   Type mila: classification",
            "ml_kind_reg": "   Type mila: regression",
            "ml_balance": "   Classes ko weight se balance kiya gaya.",
            "ml_cv": "   {k} folds ke saath cross-validation.",
            "ml_metrics": "   Test ke metrics: {texto}",
            "target_scaled": "[CHETAVANI] Target '{alvo}' scale ho gaya. .scale_kare(exclude=['{alvo}']) use karein.",
            "model_load_err": "[ERROR] Model load karne mein fail: {e}",
            "pred_err": "[ERROR] Prediction fail hui: {e}",
            "proba_unsupported": "[ERROR] Yeh model probability nahi nikalta.",
            "proba_done": "[SAMBHAVNA] Columns bane: {cols}",
            "scale_skip": "[CHETAVANI] Scale nahi hua. Columns nahi mile: {cols}",
            "api_missing": "[API] Dependencies install karein: pip install sanice[api]",
            "api_err": "[API] Galti: {e}",
            "dup_removed": "[DUPLICATE] {qtd} duplicate rows hataye gaye.",
            "rename_ok": "[NAAM] {n} columns ke naam badle gaye.",
            "null_title": "\n[NULL] Har column mein null ka pratishat:",
            "null_none": "[NULL] Koi null value nahi mili.",
            "info_title": "\n[INFO] {rows} rows x {cols} cols | Memory: {mem:.2f} MB",
            "sample_ok": "[NAMUNA] {n} rows chune gaye.",
            "validate_ok": "[JAANCH] '{col}' ({rule}): {total} mein se {invalidos} galat.",
            "validate_unknown": "[ERROR] Jaanch ka rule maloom nahi: '{rule}'.",
            "date_extract": "[TARIKH] '{col}' se columns bane: {cols}",
            "lag_ok": "[LAG] '{col}' ke liye columns bane: {cols}",
            "anom_ok": "[ANOMALY] Column '{col}' mein {qtd} anomalies mili.",
            "seg_ok": "[SEGMENT] Column '{col}' mein {n} segments bane.",
            "no_numeric": "[ERROR] Koi numeric column nahi hai.",
            "imp_title": "\n[MAHATVA] Model par sabse zyada asar karne wale variables:",
            "imp_unsupported": "[ERROR] Yeh model variable importance nahi deta.",
            "no_metrics": "[CHETAVANI] Metrics nahi hain. Pehle .auto_ml() se train karein.",
            "eval_title": "\n[MULYANKAN] {nome}:",
            "metric_accuracy": "Accuracy",
            "metric_f1": "F1",
            "metric_r2": "R²",
            "metric_rmse": "RMSE",
            "metric_mae": "MAE",
            "report_ok": "[REPORT] Report save hui: {path}",
            "rep_title": "Sanice Report",
            "rep_steps": "Kiye gaye steps",
            "rep_summary": "Sankhyiki saar",
            "rep_model": "Model",
            "rep_data": "Pehli rows",
            "undo_ok": "[UNDO] Pichla operation wapas liya gaya.",
            "undo_empty": "[CHETAVANI] Wapas karne ke liye kuch nahi hai.",
            "log_level": "[LOG] Sanice ka level set hua: {nivel}",
            "trans_text": "[BADLAV] '{col}' standardize ki gayi.",
            "trans_region": "[BADLAV] '{col}' state code se region mein badli gayi.",
        }
    }


    METHOD_ALIASES = {
        "exportar_mongo":     ["export_mongo",        "导出Mongo",    "mongo_bheje"],
        "corrigir_colunas":   ["fix_columns",         "修正列名",     "column_sudhare"],
        "limpar_texto":       ["clean_text",          "清洗文本",     "text_safai"],
        "remover_nulos":      ["remove_nulls",        "移除空值",     "null_hataye"],
        "converter_data":     ["convert_date",        "转换日期",     "date_badlo"],
        "criar_coluna":       ["create_column",       "创建列",       "column_banaye"],
        "filtrar":            ["filter_data",         "过滤数据",     "filter_kare"],
        "ordenar":            ["sort_data",           "排序数据",     "sort_kare"],
        "unir":               ["join_data",           "合并数据",     "jode"],
        "plotar":             ["plot_chart",          "绘制图表",     "graph_banaye"],
        "resumo_estatistico": ["stats_summary",       "统计摘要",     "stats_dekhe"],
        "salvar":             ["save_file",           "保存文件",     "save_kare"],
        "auto_ml":            ["train_automl",        "自动训练",     "automl_kare"],
        "carregar_ia":        ["load_ai",             "加载模型",     "ai_load_kare"],
        "prever":             ["predict",             "预测",         "bhavishya_bataye"],
        "ver":                ["view",                "查看",         "dekhe"],
        "ajuda":              ["help",                "帮助",         "madad"],
        "agrupar":            ["group_by",            "分组",         "samuh_banaye"],
        "tabela_dinamica":    ["pivot_table",         "透视表",       "pivot_banaye"],
        "exportar_sql":       ["export_sql",          "导出SQL",      "sql_export"],
        "matriz_correlacao":  ["correlation_matrix",  "相关矩阵",     "sambandh_matrix"],
        "tratar_outliers":    ["handle_outliers",     "处理异常值",   "outliers_hataye"],
        "escalonar":          ["scale_data",          "数据缩放",     "scale_kare"],
        "servir_api":         ["serve_api",           "启动API",      "api_chalu_kare"],
        "transformar":        ["transform",           "数据转换",     "badlav_kare"],
        "configurar_logs":    ["configure_logs",      "配置日志",     "log_set_kare"],
        "selecionar_colunas": ["select_columns",      "选择列",       "columns_chunne"],
        "pegar_dataframe":    ["get_dataframe",       "获取数据",     "data_lo"],
        "remover_duplicados": ["remove_duplicates",   "移除重复",     "duplicate_hataye"],
        "renomear":           ["rename_columns",      "重命名列",     "naam_badle"],
        "nulos":              ["null_report",         "空值报告",     "null_dekhe"],
        "informacoes":        ["info",                "数据信息",     "jankari"],
        "amostra":            ["sample_rows",         "抽样",         "namuna"],
        "validar":            ["validate",            "验证数据",     "jaanch_kare"],
        "extrair_data":       ["extract_date",        "提取日期",     "date_nikale"],
        "criar_atrasos":      ["create_lags",         "创建滞后",     "lag_banaye"],
        "detectar_anomalias": ["detect_anomalies",    "检测异常",     "anomaly_khoje"],
        "segmentar":          ["segment",             "客户分群",     "segment_banaye"],
        "importancia":        ["feature_importance",  "特征重要性",   "mahatva"],
        "avaliar":            ["evaluate",            "评估模型",     "model_jaanche"],
        "prever_proba":       ["predict_proba",       "预测概率",     "sambhavna_bataye"],
        "salvar_relatorio":   ["save_report",         "保存报告",     "report_save_kare"],
        "desfazer":           ["undo",                "撤销",         "vapas_kare"],
    }

    VERBOSITY_MAP = {
        "silent": logging.CRITICAL + 1,
        "error": logging.ERROR,
        "warn": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG
    }

    CURRENCY_MAP = {
        "pt": "BRL",
        "en": "USD",
        "zh": "CNY",
        "hi": "INR",
    }

    MAX_DESFAZER = 5
    SEPARADORES = [";", ",", "\t", "|"]
    CODIFICACOES = ["utf-8-sig", "cp1252", "latin1"]
    SEM_HISTORICO = {
        "view", "stats", "types", "help_title", "null_title", "null_none",
        "info_title", "imp_title", "eval_title", "load_enc",
    }
    METRICAS_CLASSIFICACAO = {"acuracia": "accuracy", "accuracy": "accuracy", "f1": "f1_macro"}
    METRICAS_REGRESSAO = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }
    ROTULO_METRICA = {
        "accuracy": "accuracy",
        "f1_macro": "f1",
        "r2": "r2",
        "neg_root_mean_squared_error": "rmse",
        "neg_mean_absolute_error": "mae",
    }

    def __init__(self, fonte_dados, lang="pt", smart_run=False, currency=None,
                 sep=None, encoding=None, aba=0, decimal=None):
        self.lang = lang
        self.df = None
        self.scaler = None
        self.colunas_escala = []
        self.escalado = False
        self.modelo_ativo = None
        self.colunas_treino = []
        self.medianas = {}
        self.alvo = None
        self.classes = None
        self.metricas = {}
        self.importancias = None
        self.tipo_modelo = None
        self.historico = []
        self._pilha = []
        moeda_padrao = self.CURRENCY_MAP.get(self.lang, "USD")
        self.currency = currency.upper() if currency else moeda_padrao

        self._setup_aliases()

        try:
            if isinstance(fonte_dados, pd.DataFrame):
                self.df = fonte_dados.copy()
            elif isinstance(fonte_dados, str):
                self.df = self._ler(fonte_dados, sep, encoding, aba, decimal)

            if self.df is not None:
                self._log("load_ok", rows=self.df.shape[0], cols=self.df.shape[1])
                if smart_run:
                    self._tentar_converter_datas()
                    self._otimizar_memoria()
        except Exception as e:
            self._erro("load_err", e=str(e))

    def _ler(self, fonte, sep, encoding, aba, decimal):
        caminho = fonte.split("?")[0].lower()
        if caminho.endswith((".csv", ".tsv", ".txt")):
            return self._ler_texto(fonte, sep, encoding, decimal)
        if caminho.endswith((".xls", ".xlsx")):
            return pd.read_excel(fonte, sheet_name=aba)
        if caminho.endswith(".parquet"):
            return pd.read_parquet(fonte)
        if caminho.endswith(".json"):
            return pd.read_json(fonte)
        self._erro("fmt_err", fonte=fonte)
        return None

    def _ler_texto(self, fonte, sep, encoding, decimal):
        codificacoes = [encoding] if encoding else self.CODIFICACOES
        falha = None
        for cod in codificacoes:
            try:
                separador = sep or self._detectar_sep(fonte, cod)
                df = pd.read_csv(fonte, sep=separador, encoding=cod, decimal=decimal or ".")
            except UnicodeDecodeError as e:
                falha = e
                continue
            if not sep or not encoding:
                self._log("load_enc", enc=cod, sep=separador.replace("\t", "\\t"))
            return df
        raise falha

    def _detectar_sep(self, fonte, codificacao):
        try:
            with open(fonte, "r", encoding=codificacao) as arquivo:
                linha = arquivo.readline()
        except OSError:
            return ","
        contagem = {s: linha.count(s) for s in self.SEPARADORES}
        melhor = max(contagem, key=contagem.get)
        return melhor if contagem[melhor] else ","

    @classmethod
    def de_sql(cls, url_conexao, query, lang="pt"):
        try:
            engine = create_engine(url_conexao)
            df = pd.read_sql(query, engine)
        except Exception as e:
            logger.error(cls._traduzir(lang, "sql_err", e=str(e)))
            return None
        instancia = cls(fonte_dados=df, lang=lang)
        instancia._log("sql_read_ok", rows=len(df))
        return instancia

    from_sql = de_sql
    从SQL = de_sql
    sql_se = de_sql

    def configurar_logs(self, nivel="info"):
        nivel_log = self.VERBOSITY_MAP.get(nivel.lower(), logging.INFO)
        logger.setLevel(nivel_log)

        if nivel.lower() != 'silent':
            print(self._traduzir(self.lang, "log_level", nivel=nivel.upper()))

        return self

    @classmethod
    def _traduzir(cls, lang, key, **kwargs):
        tabela = cls.I18N.get(lang, cls.I18N["en"])
        msg = tabela.get(key) or cls.I18N["en"].get(key, "")
        return msg.format(**kwargs) if msg else ""

    def _log(self, key, nivel=logging.INFO, **kwargs):
        msg = self._traduzir(self.lang, key, **kwargs)
        if msg:
            if nivel == logging.INFO and key not in self.SEM_HISTORICO:
                self.historico.append(msg.strip())
            logger.log(nivel, msg)
        return msg

    def _erro(self, key, **kwargs):
        return self._log(key, nivel=logging.ERROR, **kwargs)

    def _aviso(self, key, **kwargs):
        return self._log(key, nivel=logging.WARNING, **kwargs)

    def _setup_aliases(self):
        if self.lang == "pt":
            return
        idx_map = {"en": 0, "zh": 1, "hi": 2}
        idx = idx_map.get(self.lang, 0)
        for metodo_pt, aliases in self.METHOD_ALIASES.items():
            try:
                alias_name = aliases[idx]
                metodo_original = getattr(self, metodo_pt)
                setattr(self, alias_name, metodo_original)
            except IndexError:
                pass

    def _snapshot(self):
        if self.df is not None:
            self._pilha.append((self.df.copy(), self.escalado, self.scaler, list(self.colunas_escala)))
            del self._pilha[:-self.MAX_DESFAZER]

    def desfazer(self):
        if not self._pilha:
            self._aviso("undo_empty")
            return self
        self.df, self.escalado, self.scaler, self.colunas_escala = self._pilha.pop()
        self._log("undo_ok")
        return self

    def _colunas_texto(self):
        return [
            c for c in self.df.columns
            if pd.api.types.is_object_dtype(self.df[c]) or pd.api.types.is_string_dtype(self.df[c])
        ]

    def _tentar_converter_datas(self):
        for col in self._colunas_texto():
            temp = pd.to_datetime(self.df[col], errors='coerce', format='mixed')
            nao_nulos = self.df[col].dropna().shape[0]
            if nao_nulos > 0:
                validos = temp.dropna().shape[0]
                taxa_sucesso = validos / nao_nulos

                if taxa_sucesso > 0.8:
                    self.df[col] = temp
                    self._log("auto_date", col=col)

    def _otimizar_memoria(self):
        convertidas = 0
        total_linhas = len(self.df)

        for col in self._colunas_texto():
            unicos = self.df[col].nunique()
            if total_linhas > 100 and (unicos / total_linhas) < 0.5:
                self.df[col] = self.df[col].astype('category')
                convertidas += 1

        if convertidas > 0:
            self._log("auto_mem", n=convertidas)

    def ajuda(self):
        self._log("help_title", lang=self.lang)
        if self.lang == "pt":
            cmds = [m for m in self.METHOD_ALIASES.keys()]
        else:
            idx_map = {"en": 0, "zh": 1, "hi": 2}
            idx = idx_map.get(self.lang, 0)
            cmds = [aliases[idx] for aliases in self.METHOD_ALIASES.values()]
        print(", ".join([f".{c}()" for c in cmds]))
        return self

    def ver(self, linhas=5, titulo=None):
        if self.df is not None:
            header = titulo if titulo else f"Top {linhas}"
            self._log("view", header=header)
            print(self.df.head(linhas))
        return self

    def nulos(self):
        percentuais = (self.df.isna().mean() * 100).round(2)
        percentuais = percentuais[percentuais > 0].sort_values(ascending=False)
        if percentuais.empty:
            self._log("null_none")
        else:
            self._log("null_title")
            print(percentuais.to_frame("%"))
        return self

    def informacoes(self):
        memoria = self.df.memory_usage(deep=True).sum() / 1024 ** 2
        self._log("info_title", rows=self.df.shape[0], cols=self.df.shape[1], mem=memoria)
        print(self.df.dtypes)
        return self

    @_mutante
    def corrigir_colunas(self):
        if self.df is not None:
            old_cols = self.df.columns
            new_cols = [unidecode.unidecode(c).strip().lower() for c in old_cols]
            new_cols = [c.replace(" ", "_").replace("/", "_").replace("-", "_") for c in new_cols]
            new_cols = [re.sub(r'[^a-z0-9_]+', '', c) for c in new_cols]
            self.df.rename(columns={o: n for o, n in zip(old_cols, new_cols)}, inplace=True)
            self._log("clean_cols")
        return self

    @_mutante
    def renomear(self, mapa):
        existentes = {antigo: novo for antigo, novo in mapa.items() if antigo in self.df.columns}
        faltantes = set(mapa) - set(existentes)
        if faltantes:
            self._aviso("select_warn", cols=faltantes)
        self.df = self.df.rename(columns=existentes)
        self._log("rename_ok", n=len(existentes))
        return self

    @_mutante
    def limpar_texto(self, colunas):
        if isinstance(colunas, str):
            colunas = [colunas]
        for col in colunas:
            if col in self.df.columns:
                self.df[col] = [str(x).strip().title() if pd.notnull(x) else x for x in self.df[col]]
                self._log("clean_txt", col=col)
        return self

    @_mutante
    def remover_nulos(self, estrategia="apagar", preencher_com=0):
        antes = len(self.df)
        if estrategia in ("apagar", "drop"):
            self.df.dropna(inplace=True)
            self._log("drop_null", qtd=antes - len(self.df))
        elif estrategia in ("preencher", "fill"):
            self.df.fillna(preencher_com, inplace=True)
            self._log("fill_null", val=preencher_com)
        return self

    @_mutante
    def remover_duplicados(self, colunas=None):
        antes = len(self.df)
        self.df = self.df.drop_duplicates(subset=colunas)
        self._log("dup_removed", qtd=antes - len(self.df))
        return self

    @_mutante
    def amostra(self, n=100, fracao=None, semente=42):
        if fracao:
            self.df = self.df.sample(frac=fracao, random_state=semente)
        else:
            self.df = self.df.sample(n=min(n, len(self.df)), random_state=semente)
        self._log("sample_ok", n=len(self.df))
        return self

    @_mutante
    def converter_data(self, colunas, formato=None):
        if isinstance(colunas, str):
            colunas = [colunas]
        for col in colunas:
            self.df[col] = pd.to_datetime(self.df[col], format=formato, errors='coerce')
            self._log("date_conv", col=col)
        return self

    @_mutante
    def extrair_data(self, coluna, partes=("ano", "mes", "dia", "dia_semana")):
        datas = pd.to_datetime(self.df[coluna], errors="coerce", dayfirst=True)
        extratores = {
            "ano": datas.dt.year,
            "mes": datas.dt.month,
            "dia": datas.dt.day,
            "dia_semana": datas.dt.dayofweek,
        }
        criadas = []
        for parte in partes:
            if parte in extratores:
                nome = f"{coluna}_{parte}"
                self.df[nome] = extratores[parte]
                criadas.append(nome)
        self._log("date_extract", col=coluna, cols=", ".join(criadas))
        return self

    @_mutante
    def criar_atrasos(self, coluna, atrasos=(1, 2, 3), medias=(3,)):
        criadas = []
        for n in atrasos:
            nome = f"{coluna}_atraso{n}"
            self.df[nome] = self.df[coluna].shift(n)
            criadas.append(nome)
        for n in medias:
            nome = f"{coluna}_media{n}"
            self.df[nome] = self.df[coluna].shift(1).rolling(n).mean()
            criadas.append(nome)
        self._log("lag_ok", col=coluna, cols=", ".join(criadas))
        return self

    @_mutante
    def criar_coluna(self, nome_nova_col, expressao_ou_func):
        try:
            if callable(expressao_ou_func):
                self.df[nome_nova_col] = self.df.apply(expressao_ou_func, axis=1)
            elif isinstance(expressao_ou_func, str):
                self.df.eval(f"{nome_nova_col} = {expressao_ou_func}", inplace=True)

            self._log("col_add", col=nome_nova_col)
        except Exception as e:
            self._erro("col_err", e=str(e))
        return self

    @_mutante
    def filtrar(self, query_string):
        try:
            antes = len(self.df)
            self.df = self.df.query(query_string)
            self._log("filter", query=query_string, before=antes, after=len(self.df))
        except Exception as e:
            self._erro("filter_err", e=str(e))
        return self

    @_mutante
    def ordenar(self, colunas, ascendente=True):
        self.df.sort_values(by=colunas, ascending=ascendente, inplace=True)
        self._log("sort", cols=colunas)
        return self

    @_mutante
    def unir(self, outro_df, chaves, como="inner"):
        tabela_direita = outro_df.df if isinstance(outro_df, Sanice) else outro_df
        antes = len(self.df)
        self.df = pd.merge(self.df, tabela_direita, on=chaves, how=como)
        self._log("join", how=como, before=antes, after=len(self.df))
        return self

    def plotar(self, tipo="barras", x=None, y=None, hue=None, titulo=None):
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        try:
            if tipo in ["barras", "bar"]:
                sns.barplot(data=self.df, x=x, y=y, hue=hue, palette="viridis")
            elif tipo in ["linha", "line"]:
                sns.lineplot(data=self.df, x=x, y=y, hue=hue)
            elif tipo in ["scatter", "dispersao"]:
                sns.scatterplot(data=self.df, x=x, y=y, hue=hue, alpha=0.7)
            elif tipo in ["hist", "histograma"]:
                sns.histplot(data=self.df, x=x, kde=True, hue=hue)
            elif tipo in ["box", "boxplot", "caixa"]:
                sns.boxplot(data=self.df, x=x, y=y, hue=hue)
            else:
                self._erro("plot_err", e=tipo)
                plt.close()
                return self

            plt.title(titulo if titulo else tipo.title())
            plt.xlabel(x)
            plt.ylabel(y)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self._erro("plot_err", e=str(e))
        return self

    def resumo_estatistico(self):
        self._log("stats")
        print(self.df.describe().T)
        self._log("types")
        print(self.df.dtypes)
        return self

    def salvar(self, caminho):
        try:
            minusculo = caminho.lower()
            if minusculo.endswith('.csv'):
                self.df.to_csv(caminho, index=False)
            elif minusculo.endswith('.xlsx'):
                self.df.to_excel(caminho, index=False)
            elif minusculo.endswith('.parquet'):
                self.df.to_parquet(caminho)
            elif minusculo.endswith('.json'):
                self.df.to_json(caminho, orient="records", force_ascii=False)
            else:
                self._erro("fmt_err", fonte=caminho)
                return self
            self._log("save", path=caminho)
        except Exception as e:
            self._erro("save_err", e=str(e))
        return self

    def salvar_relatorio(self, caminho):
        def t(chave):
            return self._traduzir(self.lang, chave)

        try:
            resumo = self.df.describe(include="all").T
            minusculo = caminho.lower()
            if minusculo.endswith(".xlsx"):
                with pd.ExcelWriter(caminho) as escritor:
                    resumo.to_excel(escritor, sheet_name="Resumo")
                    pd.DataFrame({"#": range(1, len(self.historico) + 1), t("rep_steps"): self.historico}).to_excel(
                        escritor, sheet_name="Passos", index=False)
                    if self.metricas:
                        pd.Series(self.metricas, name="valor").to_frame().to_excel(escritor, sheet_name="Modelo")
                    self.df.head(1000).to_excel(escritor, sheet_name="Dados", index=False)
            elif minusculo.endswith(".html"):
                passos = "".join(f"<li>{html.escape(p)}</li>" for p in self.historico)
                modelo = ""
                if self.metricas:
                    linhas = "".join(
                        f"<li>{html.escape(t('metric_' + k))}: {v:.4f}</li>" for k, v in self.metricas.items())
                    modelo = f"<h2>{t('rep_model')}: {html.escape(str(self.tipo_modelo))}</h2><ul>{linhas}</ul>"
                pagina = (
                    f"<!DOCTYPE html><html lang=\"{self.lang}\"><head><meta charset=\"utf-8\">"
                    f"<title>{t('rep_title')}</title></head><body>"
                    f"<h1>{t('rep_title')}</h1>"
                    f"<h2>{t('rep_steps')}</h2><ol>{passos}</ol>"
                    f"{modelo}"
                    f"<h2>{t('rep_summary')}</h2>{resumo.to_html()}"
                    f"<h2>{t('rep_data')}</h2>{self.df.head(50).to_html(index=False)}"
                    f"</body></html>"
                )
                with open(caminho, "w", encoding="utf-8") as arquivo:
                    arquivo.write(pagina)
            else:
                self._erro("fmt_err", fonte=caminho)
                return self
            self._log("report_ok", path=caminho)
        except Exception as e:
            self._erro("save_err", e=str(e))
        return self

    def pegar_dataframe(self):
        return self.df

    @_mutante
    def selecionar_colunas(self, colunas):
        if isinstance(colunas, str):
            colunas = [colunas]
        cols_existentes = [c for c in colunas if c in self.df.columns]

        if len(cols_existentes) < len(colunas):
            faltantes = set(colunas) - set(cols_existentes)
            self._aviso("select_warn", cols=faltantes)

        self.df = self.df[cols_existentes]
        self._log("select_ok", n=len(cols_existentes))
        return self

    @staticmethod
    def _texto_seguro(serie):
        return serie.astype("string")

    @staticmethod
    def _para_objeto(serie):
        return serie.astype(object).where(serie.notna(), np.nan)

    @staticmethod
    def _inteiros(serie):
        if pd.api.types.is_float_dtype(serie) and (serie.dropna() % 1 == 0).all():
            return serie.astype("Int64")
        return serie

    @_mutante
    def transformar(self, colunas, regra):
        if isinstance(colunas, str):
            colunas = [colunas]

        r = regra.upper()

        RULES_MONEY = ["BRL", "USD", "CNY", "INR", "DINHEIRO", "MONEY", "CURRENCY", "金钱", "PAISA", "MUDRA"]
        RULES_NUM = ["CPF", "CNPJ", "CEP", "NUMEROS", "NUMBERS", "TELEFONE", "DIGITS", "数字", "ANK"]
        RULES_EMAIL = ["EMAIL", "E-MAIL", "MAIL", "邮件"]
        RULES_UPPER = ["UPPER", "MAIUSCULO", "CAPS", "大写", "BADA"]
        RULES_LOWER = ["LOWER", "MINUSCULO", "小写", "CHOTA"]
        RULES_DATE = ["DATE", "DATA", "日期", "TARIKH"]
        RULES_UF = ["UF", "ESTADO", "STATE"]
        RULES_REGIAO = ["REGIAO", "REGION", "地区", "KSHETRA"]
        TAMANHOS = {"CPF": 11, "CNPJ": 14, "CEP": 8}

        for col in colunas:
            if col not in self.df.columns:
                continue

            serie = self.df[col]

            if r in RULES_MONEY:
                moeda_a_usar = r if r in self.CURRENCY_MAP.values() else self.currency
                self.df[col] = serie.apply(lambda x: self._tratar_moeda(x, moeda_a_usar))
                self._log("trans_money", col=col)

            elif r in RULES_NUM:
                digitos = self._texto_seguro(self._inteiros(serie)).str.replace(r"\D", "", regex=True)
                if r in TAMANHOS:
                    digitos = digitos.str.zfill(TAMANHOS[r])
                self.df[col] = self._para_objeto(digitos)
                self._log("trans_num", col=col)

            elif r in RULES_EMAIL:
                texto = self._texto_seguro(serie).str.lower().str.strip()
                valido = texto.str.contains(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", na=False)
                self.df[col] = self._para_objeto(texto).where(valido, np.nan)
                self._log("trans_email", col=col)

            elif r in RULES_UPPER:
                self.df[col] = self._para_objeto(self._texto_seguro(serie).str.upper())
                self._log("trans_text", col=col)

            elif r in RULES_LOWER:
                self.df[col] = self._para_objeto(self._texto_seguro(serie).str.lower())
                self._log("trans_text", col=col)

            elif r in RULES_DATE:
                self.df[col] = pd.to_datetime(serie, errors="coerce", dayfirst=True, format="mixed")
                self._log("trans_date", col=col)

            elif r in RULES_UF:
                self.df[col] = self._para_objeto(self._texto_seguro(serie).str.strip().str.upper())
                self._log("trans_text", col=col)

            elif r in RULES_REGIAO:
                ufs = self._texto_seguro(serie).str.strip().str.upper()
                self.df[col] = self._para_objeto(ufs.map(REGIOES))
                self._log("trans_region", col=col)

            else:
                self._log("trans_err", nivel=logging.ERROR, rule=regra)

        return self

    @_mutante
    def validar(self, colunas, regra, remover=False):
        if isinstance(colunas, str):
            colunas = [colunas]
        r = regra.upper()
        if r not in VALIDADORES:
            self._erro("validate_unknown", rule=regra)
            return self
        for col in colunas:
            if col not in self.df.columns:
                continue
            mascara = self.df[col].map(lambda v: (not pd.isna(v)) and VALIDADORES[r](v)).astype(bool)
            self._log("validate_ok", col=col, rule=r, invalidos=int((~mascara).sum()), total=len(mascara))
            if remover:
                self.df = self.df[mascara].copy()
            else:
                self.df[f"{col}_valido"] = mascara
        return self

    def _tratar_moeda(self, valor, codigo_moeda):
        if pd.isna(valor):
            return np.nan

        if isinstance(valor, (int, float, np.number)) and not isinstance(valor, bool):
            return float(valor)

        s = str(valor).strip()

        if codigo_moeda == "BRL":
            s = s.replace('R$', '').replace(' ', '').replace('.', '')
            s = s.replace(',', '.')
        elif codigo_moeda == "USD":
            s = s.replace('$', '').replace(' ', '').replace(',', '')
        elif codigo_moeda == "CNY":
            s = s.replace('¥', '').replace(' ', '').replace(',', '')
        elif codigo_moeda == "INR":
            s = s.replace('₹', '').replace(' ', '').replace(',', '')
        try:
            return float(s)
        except ValueError:
            return np.nan

    def agrupar(self, colunas_agrupar, coluna_valor, operacao="soma"):
        ops = {"soma": "sum", "media": "mean", "contagem": "count", "max": "max", "min": "min"}
        op_pandas = ops.get(operacao, operacao if operacao in ops.values() else "sum")
        try:
            resultado = self.df.groupby(colunas_agrupar)[coluna_valor].agg(op_pandas).reset_index()
            self._log("view", header=self._traduzir(self.lang, "group_header", cols=colunas_agrupar, op=operacao))
            print(resultado)
        except Exception as e:
            self._erro("group_err", e=str(e))
        return self

    def tabela_dinamica(self, indice, colunas, valores, func="soma"):
        ops = {"soma": "sum", "media": "mean", "contagem": "count"}
        try:
            pivot = pd.pivot_table(self.df, values=valores, index=indice, columns=colunas,
                                   aggfunc=ops.get(func, func if func in ops.values() else "sum"))
            self._log("view", header=self._traduzir(self.lang, "pivot_header"))
            print(pivot)
        except Exception as e:
            self._erro("pivot_err", e=str(e))
        return self

    def exportar_mongo(self, uri, database, collection):
        try:
            import pymongo
        except ImportError:
            self._erro("mongo_missing")
            return self
        try:
            client = pymongo.MongoClient(uri)
            client[database][collection].insert_many(self.df.to_dict(orient="records"))
            self._log("mongo_ok", col=collection)
        except Exception as e:
            self._erro("mongo_err", e=str(e))
        return self

    def exportar_sql(self, url_conexao, nome_tabela, modo="append"):
        try:
            engine = create_engine(url_conexao)
            self.df.to_sql(nome_tabela, engine, if_exists=modo, index=False)
            self._log("sql_ok", tb=nome_tabela)
        except Exception as e:
            self._erro("sql_err", e=str(e))
        return self

    def matriz_correlacao(self):
        try:
            plt.figure(figsize=(10, 8))
            df_num = self.df.select_dtypes(include=[np.number])
            corr = df_num.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            plt.title("Matriz de Correlação")
            plt.show()
        except Exception as e:
            self._erro("plot_err", e=str(e))
        return self

    @_mutante
    def tratar_outliers(self, colunas, metodo="iqr"):
        if isinstance(colunas, str):
            colunas = [colunas]
        antes = len(self.df)

        for col in colunas:
            if col not in self.df.columns:
                continue
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            filtro = ~((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR)))
            self.df = self.df[filtro]

        self._log("outlier_rem", qtd=antes - len(self.df))
        return self

    @_mutante
    def escalonar(self, metodo="minmax", colunas=None, excluir=None, exclude=None):
        excluir = excluir or exclude or []
        if isinstance(excluir, str):
            excluir = [excluir]
        if isinstance(colunas, str):
            colunas = [colunas]
        numericas = self.df.select_dtypes(include=[np.number]).columns
        candidatas = numericas if colunas is None else [c for c in colunas if c in numericas]
        cols = [c for c in candidatas if c not in excluir]
        if not cols:
            self._erro("no_numeric")
            return self

        self.scaler = MinMaxScaler() if metodo == "minmax" else StandardScaler()
        self.df[cols] = self.scaler.fit_transform(self.df[cols])
        self.colunas_escala = list(cols)
        self.escalado = True
        self._log("scale_ok", method=metodo)
        return self

    @staticmethod
    def _arg(kwargs, *nomes, default=None):
        for nome in nomes:
            if kwargs.get(nome) is not None:
                return kwargs[nome]
        return default

    @staticmethod
    def _eh_classificacao(y, tipo):
        t = str(tipo).lower()
        if any(x in t for x in ("class", "fenlei", "binario", "分类", "vargikaran")):
            return True
        if any(x in t for x in ("regress", "huigui", "回归")):
            return False
        if y.dtype == bool or not pd.api.types.is_numeric_dtype(y):
            return True
        return bool((y.dropna() % 1 == 0).all() and y.nunique() <= 10)

    def _preparar(self, X, escalar, treino=False):
        X = X.copy()
        X = X.drop(columns=X.select_dtypes(include=["datetime", "datetimetz"]).columns)

        if escalar and self.scaler is not None and self.colunas_escala:
            faltam = [c for c in self.colunas_escala if c not in X.columns]
            if faltam:
                self._aviso("scale_skip", cols=", ".join(faltam))
            else:
                X[self.colunas_escala] = self.scaler.transform(X[self.colunas_escala])

        numericas = list(X.select_dtypes(include=[np.number]).columns)
        if treino:
            self.medianas = {c: X[c].median() for c in numericas}
        for c in numericas:
            mediana = self.medianas.get(c, 0)
            X[c] = X[c].fillna(0 if pd.isna(mediana) else mediana)

        for c in X.columns:
            if c not in numericas and not pd.api.types.is_bool_dtype(X[c]):
                X[c] = X[c].astype(object).where(X[c].notna(), "ausente")

        if treino:
            X = pd.get_dummies(X, drop_first=True)
            self.colunas_treino = X.columns.tolist()
        else:
            X = pd.get_dummies(X).reindex(columns=self.colunas_treino, fill_value=0)
        return X.astype(float)

    @staticmethod
    def _ajustar(modelo, X, y, balancear):
        if balancear:
            modelo.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        else:
            modelo.fit(X, y)

    def _pontuar(self, modelo, X_tr, y_tr, X_te, y_te, scorer, balancear, k, clf):
        if k and k >= 2:
            if clf and y_tr.value_counts().min() >= k:
                divisor = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            else:
                divisor = KFold(n_splits=k, shuffle=True, random_state=42)
            notas = []
            for i_tr, i_te in divisor.split(X_tr, y_tr):
                copia = clone(modelo)
                self._ajustar(copia, X_tr.iloc[i_tr], y_tr.iloc[i_tr], balancear)
                notas.append(scorer(copia, X_tr.iloc[i_te], y_tr.iloc[i_te]))
            return float(np.mean(notas))
        self._ajustar(modelo, X_tr, y_tr, balancear)
        return float(scorer(modelo, X_te, y_te))

    @staticmethod
    def _calcular_metricas(modelo, X_te, y_te, clf):
        previsto = modelo.predict(X_te)
        if clf:
            return {
                "accuracy": float(accuracy_score(y_te, previsto)),
                "f1": float(f1_score(y_te, previsto, average="macro")),
            }
        return {
            "r2": float(r2_score(y_te, previsto)),
            "rmse": float(np.sqrt(mean_squared_error(y_te, previsto))),
            "mae": float(mean_absolute_error(y_te, previsto)),
        }

    def auto_ml(self, alvo=None, tipo=None, salvar_modelo=None, **kwargs):
        alvo = alvo or self._arg(kwargs, "target", "目标", "mubiao", "lakshya")
        tipo = tipo or self._arg(kwargs, "type", default="auto")
        salvar_modelo = salvar_modelo or self._arg(kwargs, "save_path", "保存路径")
        teste_tam = self._arg(kwargs, "teste_tam", "test_size", default=0.2)
        balancear = self._arg(kwargs, "balancear", "balance", "平衡", "santulan", default=False)
        k = self._arg(kwargs, "validacao_cruzada", "cross_validation", "交叉验证", default=0)
        nome_metrica = self._arg(kwargs, "metrica", "metric", "指标")

        if not alvo:
            self._erro("ml_no_target")
            return self
        if alvo not in self.df.columns:
            self._erro("select_warn", cols={alvo})
            return self

        self._log("ml_start", target=alvo)
        if alvo in self.colunas_escala:
            self._aviso("target_scaled", alvo=alvo)

        try:
            df_ml = self.df.dropna(subset=[alvo])
            y = df_ml[alvo]
            X = df_ml.drop(columns=[alvo])

            clf = self._eh_classificacao(y, tipo)
            self._log("ml_kind_clf" if clf else "ml_kind_reg")

            datas = list(X.select_dtypes(include=["datetime", "datetimetz"]).columns)
            if datas:
                self._log("ml_ignore_date", cols=datas)

            X = self._preparar(X, escalar=False, treino=True)
            self._log("ml_feats", n=X.shape[1])

            estratificar = y if clf and y.value_counts().min() >= 2 else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=teste_tam, random_state=42, stratify=estratificar)
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=teste_tam, random_state=42)

            balancear = bool(balancear and clf)
            if clf:
                modelos = {
                    "LogisticRegression": LogisticRegression(max_iter=1000),
                    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "GradientBoosting": GradientBoostingClassifier(random_state=42),
                }
                padrao = "f1_macro" if balancear else "accuracy"
                mapa = self.METRICAS_CLASSIFICACAO
            else:
                modelos = {
                    "LinearRegression": LinearRegression(),
                    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
                    "GradientBoosting": GradientBoostingRegressor(random_state=42),
                }
                padrao = "r2"
                mapa = self.METRICAS_REGRESSAO
            scoring = mapa.get(str(nome_metrica).lower(), padrao)
            rotulo = self._traduzir(self.lang, "metric_" + self.ROTULO_METRICA[scoring])
            scorer = get_scorer(scoring)

            if balancear:
                self._log("ml_balance")
            if k and k >= 2:
                self._log("ml_cv", k=k)
            self._log("ml_tourn", n=len(modelos))

            melhor_score = -float("inf")
            melhor_modelo = None
            melhor_nome = ""

            for nome, modelo in modelos.items():
                try:
                    score = self._pontuar(modelo, X_train, y_train, X_test, y_test, scorer, balancear, k, clf)
                except Exception as e:
                    self._erro("ml_fail", name=nome, e=str(e))
                    continue
                if score > melhor_score:
                    melhor_score, melhor_modelo, melhor_nome = score, modelo, nome

            if melhor_modelo is None:
                self._erro("ml_err", e="no model could be trained")
                return self

            self._ajustar(melhor_modelo, X_train, y_train, balancear)
            self._log("ml_win", name=melhor_nome, metric=rotulo, score=abs(melhor_score))

            self.metricas = self._calcular_metricas(melhor_modelo, X_test, y_test, clf)
            texto = ", ".join(
                f"{self._traduzir(self.lang, 'metric_' + chave)} {valor:.4f}" for chave, valor in self.metricas.items())
            self._log("ml_metrics", texto=texto)

            self.modelo_ativo = melhor_modelo
            self.tipo_modelo = melhor_nome
            self.alvo = alvo
            self.classes = list(melhor_modelo.classes_) if clf else None
            self.importancias = None
            self._score = abs(melhor_score)

            if salvar_modelo:
                joblib.dump(self._pacote(), salvar_modelo)
                self._log("ml_saved", path=salvar_modelo)

        except Exception as e:
            self._erro("ml_err", e=str(e))

        return self

    def _pacote(self):
        return {
            "modelo": self.modelo_ativo,
            "colunas_treino": self.colunas_treino,
            "scaler": self.scaler,
            "colunas_escala": self.colunas_escala,
            "medianas": self.medianas,
            "alvo": self.alvo,
            "classes": self.classes,
            "metricas": self.metricas,
            "tipo_modelo": self.tipo_modelo,
            "score": getattr(self, "_score", None),
        }

    def carregar_ia(self, caminho_modelo):
        try:
            pacote = joblib.load(caminho_modelo)
            self.modelo_ativo = pacote["modelo"]
            self.colunas_treino = pacote["colunas_treino"]
            self.scaler = pacote.get("scaler")
            colunas_escala = pacote.get("colunas_escala")
            if colunas_escala is None and self.scaler is not None:
                colunas_escala = list(getattr(self.scaler, "feature_names_in_", []))
            self.colunas_escala = list(colunas_escala or [])
            self.medianas = pacote.get("medianas", {})
            self.alvo = pacote.get("alvo")
            self.classes = pacote.get("classes")
            self.metricas = pacote.get("metricas", {})
            self.tipo_modelo = pacote.get("tipo_modelo")
            self.escalado = False
            self._log("ia_loaded", n=len(self.colunas_treino))
        except Exception as e:
            self._erro("model_load_err", e=str(e))
        return self

    @_mutante
    def prever(self, nome_coluna_saida="previsao"):
        if self.modelo_ativo is None:
            self._erro("err_load_ia")
            return self

        try:
            X = self._preparar(self.df.drop(columns=[self.alvo], errors="ignore"), escalar=not self.escalado)
            self.df[nome_coluna_saida] = self.modelo_ativo.predict(X)
            self._log("pred_done", col=nome_coluna_saida)
        except Exception as e:
            self._erro("pred_err", e=str(e))
        return self

    @_mutante
    def prever_proba(self, prefixo="prob"):
        if self.modelo_ativo is None:
            self._erro("err_load_ia")
            return self
        if not hasattr(self.modelo_ativo, "predict_proba"):
            self._erro("proba_unsupported")
            return self

        try:
            X = self._preparar(self.df.drop(columns=[self.alvo], errors="ignore"), escalar=not self.escalado)
            probabilidades = self.modelo_ativo.predict_proba(X)
            criadas = []
            for indice, classe in enumerate(self.modelo_ativo.classes_):
                nome = f"{prefixo}_{classe}"
                self.df[nome] = probabilidades[:, indice]
                criadas.append(nome)
            self._log("proba_done", cols=", ".join(criadas))
        except Exception as e:
            self._erro("pred_err", e=str(e))
        return self

    def importancia(self, top=10):
        if self.modelo_ativo is None:
            self._erro("err_load_ia")
            return self
        if hasattr(self.modelo_ativo, "feature_importances_"):
            valores = self.modelo_ativo.feature_importances_
        elif hasattr(self.modelo_ativo, "coef_"):
            valores = np.abs(np.atleast_2d(self.modelo_ativo.coef_)).mean(axis=0)
        else:
            self._erro("imp_unsupported")
            return self
        self.importancias = pd.Series(valores, index=self.colunas_treino).sort_values(ascending=False)
        self._log("imp_title")
        print(self.importancias.head(top).round(4).to_string())
        return self

    def avaliar(self):
        if not self.metricas:
            self._aviso("no_metrics")
            return self
        self._log("eval_title", nome=self.tipo_modelo)
        for chave, valor in self.metricas.items():
            print(f"   {self._traduzir(self.lang, 'metric_' + chave)}: {valor:.4f}")
        return self

    def _colunas_numericas(self, colunas, saida):
        if isinstance(colunas, str):
            colunas = [colunas]
        if colunas is None:
            colunas = [c for c in self.df.select_dtypes(include=[np.number]).columns if c != saida]
        colunas = [c for c in colunas if c in self.df.columns]
        if not colunas:
            self._erro("no_numeric")
            return None
        X = self.df[colunas]
        return X.fillna(X.median()).fillna(0)

    @_mutante
    def detectar_anomalias(self, colunas=None, contaminacao=0.05, coluna_saida="anomalia"):
        X = self._colunas_numericas(colunas, coluna_saida)
        if X is None:
            return self
        rotulos = IsolationForest(contamination=contaminacao, random_state=42).fit_predict(X)
        self.df[coluna_saida] = rotulos == -1
        self._log("anom_ok", qtd=int((rotulos == -1).sum()), col=coluna_saida)
        return self

    @_mutante
    def segmentar(self, colunas=None, grupos=3, coluna_saida="segmento"):
        X = self._colunas_numericas(colunas, coluna_saida)
        if X is None:
            return self
        try:
            Z = StandardScaler().fit_transform(X)
            self.df[coluna_saida] = KMeans(n_clusters=grupos, n_init=10, random_state=42).fit_predict(Z)
            self._log("seg_ok", n=grupos, col=coluna_saida)
        except Exception as e:
            self._erro("ml_err", e=str(e))
        return self

    def servir_api(self, host="127.0.0.1", porta=8000):
        if self.modelo_ativo is None:
            self._erro("err_load_ia")
            return

        try:
            import uvicorn
            from fastapi import FastAPI, HTTPException
        except ImportError:
            self._erro("api_missing")
            return

        app = FastAPI(title="Sanice API", description="API gerada automaticamente pelo Sanice")

        @app.get("/")
        def home():
            return {"status": "Sanice está online", "modelo": str(type(self.modelo_ativo))}

        @app.post("/predict")
        def predict(dados: dict):
            faltam = [c for c in self.colunas_escala if c not in dados]
            if faltam:
                raise HTTPException(status_code=422, detail=f"Campos ausentes: {faltam}")
            try:
                X = self._preparar(pd.DataFrame([dados]), escalar=True)
                previsao = self.modelo_ativo.predict(X)[0]
                resposta = {"predicao": previsao.item() if hasattr(previsao, "item") else previsao}
                if hasattr(self.modelo_ativo, "predict_proba"):
                    probabilidades = self.modelo_ativo.predict_proba(X)[0]
                    resposta["probabilidades"] = {
                        str(c): float(p) for c, p in zip(self.modelo_ativo.classes_, probabilidades)}
                return resposta
            except Exception as e:
                raise HTTPException(status_code=422, detail=str(e))

        try:
            self._log("api_start", host=host, porta=porta)
            uvicorn.run(app, host=host, port=porta)
        except Exception as e:
            self._erro("api_err", e=str(e))


def cli():
    import sys
    from sanice import __version__ as VERSION

    CLI_MSGS = {
        "en": "To use inside Python:",
        "pt": "Para usar no script Python:",
        "zh": "在 Python 脚本中使用：",
        "hi": "Python script mein use karne ke liye:"
    }

    CMD_MAP = {
        "help": "en",
        "--help": "en",
        "-h": "en",
        "ajuda": "pt",
        "socorro": "pt",
        "bangzhu": "zh",
        "madad": "hi"
    }

    args = sys.argv
    comando = args[1].lower() if len(args) > 1 else ""

    if comando in ["-v", "--version", "version", "versao", "-version"]:
        print(f"Sanice v{VERSION}")
        return

    if comando in CMD_MAP:

        lang_padrao = CMD_MAP[comando]
        user_lang = args[2] if len(args) > 2 else lang_padrao

        if user_lang not in CLI_MSGS:
            user_lang = "en"
        msg = CLI_MSGS[user_lang]

        print(f"\n=== Sanice CLI Help ({user_lang.upper()}) ===")
        print(f"\n{msg}")
        print("  from sanice import Sanice")
        print(f"  app = Sanice('data.csv', lang='{user_lang}')")
        print("\nReference / Referência (PT | EN | ZH | HI):")
        print("-" * 75)

        for pt_method, aliases in Sanice.METHOD_ALIASES.items():
            en, zh, hi = aliases[0], aliases[1], aliases[2]
            print(f"  {pt_method:<20} | {en:<20} | {zh:<8} | {hi}")

        print("-" * 75)
        print(f"v{VERSION}")

    else:
        print(f"Sanice v{VERSION} installed! Try:")
        print("  sanice help     (English)")
        print("  sanice ajuda    (Português)")
