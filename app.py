import json
import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import re
import numpy as np

from io import BytesIO
from PIL import Image

from pandasai import SmartDataframe
from pandasai.llm import OpenAI as PandasAIOpenAI
import openai

from pandasai.exceptions import NoCodeFoundError, NoResultFoundError

os.makedirs("exports/charts", exist_ok=True)

# Configuração da API Key
api_key = st.secrets.get("OPEN_API_KEY")
user_question = ""
if not api_key:
    st.error("Chave OPEN_API_KEY não encontrada em st.secrets")
    st.stop()

# Cliente oficial OpenAI (para chat, embeddings, etc.)
openai.api_key = st.secrets.get("OPEN_API_KEY")

# Cliente PandasAI (para SmartDataframe)
pandasai_llm = PandasAIOpenAI(api_token=api_key, model="gpt-3.5-turbo")

st.set_page_config(page_title="Análise de CSV com IA", layout="wide")
st.title("🔍 Análise de CSV com Perguntas (EDA + Q&A)")


def normalize_response(resp):
    try:
        if isinstance(resp, (str, int, float)):
            return str(resp)
        elif isinstance(resp, pd.DataFrame):
            return resp.to_json(orient="split")
        elif isinstance(resp, pd.Series):
            return resp.to_json(orient="split")
        elif isinstance(resp, dict):
            return str(resp)
        elif isinstance(resp, list):
            return str(resp)
        else:
            # tenta usar to_string se existir
            if hasattr(resp, "to_string"):
                return resp.to_string()
            return str(resp)
    except Exception:
        return str(resp)


@st.cache_data
def load_and_clean_data(file_bytes):
    file_like_object = BytesIO(file_bytes)
    df = pd.read_csv(file_like_object, sep=",", engine="python", quoting=3)

    df.columns = (
        df.columns.astype(str)
        .str.replace('"', "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.replace('"', "", regex=False).str.strip()

    converted_cols = []
    for col in df.columns:
        try:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            if numeric_series.notna().sum() > 0:
                if (numeric_series.notna().sum() / len(df[col])) > 0.8:
                    df[col] = numeric_series
                    converted_cols.append(col)
        except Exception:
            continue

    return df, converted_cols

# --- Configuração do LLM ---
try:
    llm = PandasAIOpenAI(api_token=api_key, model="gpt-3.5-turbo")
except Exception:
    st.error(
        "Chave da OpenAI não encontrada. Configure seus segredos (secrets) no Streamlit."
    )
    st.stop()

uploaded_file = st.file_uploader("📂 Faça upload do seu arquivo CSV", type=["csv"])

def get_metadata(df):
    return {
        "colunas": df.columns.tolist(),
        "tipos": df.dtypes.astype(str).to_dict(),
        "nulos": df.isnull().sum().to_dict(),
        "exemplo_linhas": df.head(3).to_dict(orient="records"),
    }


def detect_time_column(df):
    candidates = [
        c
        for c in df.columns
        if c.lower() in ("time", "timestamp", "date", "datetime", "dt")
    ]
    if candidates:
        return candidates[0]
    # try columns that look like dates
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna().astype(str).head(5).tolist()
            # quick heuristic: contains '-' or '/' or 'T'
            if any(("-" in s or "/" in s or "T" in s) for s in sample):
                return c
    return None


def perguntar_modelo(user_question, metadata):
    system_prompt = """Você é um assistente de análise de DataFrames.
    Sempre responda com um JSON válido, descrevendo a ação.
    Exemplos:
    {"acao": "listar_tipos"}
    {"acao": "estatisticas", "colunas": ["amount"], "metricas": ["mean"]}
    {"acao": "valores_unicos", "coluna": "class"}
    """

    mensagem = f"""
    Pergunta: {user_question}

    Metadados do DataFrame:
    {json.dumps(metadata, indent=2)}
    """

    resposta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": mensagem},
        ],
        temperature=0,
    )

    return resposta.choices[0].message.content


def executar_resposta(resposta, df, max_examples=3):
    try:
        dados = json.loads(resposta)
    except Exception:
        return {"erro": "Resposta não é JSON válido", "conteudo": resposta}

    acao = dados.get("acao")

    if acao == "listar_tipos":
        tipos = {
            "inteiros": [],
            "decimais": [],
            "booleanos": [],
            "datas": [],
            "categóricas": [],
            "texto_livre": [],
        }

        for col in df.columns:
            serie = df[col]
            dtype = str(serie.dtype)

            if pd.api.types.is_integer_dtype(serie):
                tipos["inteiros"].append(col)
            elif pd.api.types.is_float_dtype(serie):
                tipos["decimais"].append(col)
            elif pd.api.types.is_bool_dtype(serie):
                tipos["booleanos"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(serie):
                tipos["datas"].append(col)
            elif pd.api.types.is_object_dtype(serie):
                n_unique = serie.nunique(dropna=True)
                if n_unique < 30 or (n_unique / len(serie)) < 0.05:
                    tipos["categóricas"].append(col)
                else:
                    tipos["texto_livre"].append(col)
            else:
                tipos["texto_livre"].append(col)

        resposta = {}
        for t, cols in tipos.items():
            if cols:
                resposta[t] = {
                    "quantidade": len(cols),
                    "colunas": cols,
                    "exemplos": {
                        col: df[col].dropna().unique()[:max_examples].tolist()
                        for col in cols
                    },
                }

        return resposta

    elif acao == "estatisticas":
        return df[dados["colunas"]].agg(dados["metricas"]).to_dict()

    elif acao == "valores_unicos":
        col = dados["coluna"]
        return {
            "coluna": col,
            "n_valores_unicos": df[col].nunique(),
            "exemplos": df[col].dropna().unique()[:max_examples].tolist(),
        }

    return {"erro": "ação não reconhecida", "dados": dados}


# Fallback: check temporal patterns
def temporal_patterns_fallback(df):
    """
    - Try to find a time column (datetime or integer time).
    - For numeric columns, compute Pearson correlation with time-as-number.
    - Report columns with |corr| >= 0.25 as showing trend; also report direction.
    - If datetime index, attempt a simple seasonal summary (monthly mean) if enough data.
    """
    time_col = detect_time_column(df)
    report = {"time_column": time_col, "trends": {}, "notes": []}

    if time_col is None:
        report["notes"].append(
            "Não foi encontrada uma coluna de tempo óbvia (procure por 'time', 'date', 'timestamp')."
        )
        return report

    # Try parse to datetime
    try:
        dt = pd.to_datetime(df[time_col], errors="coerce")
        if dt.notna().sum() / len(dt) > 0.1:  # at least some parsed
            df_time = df.copy()
            df_time["_parsed_time"] = pd.to_datetime(df_time[time_col], errors="coerce")
            df_time = df_time.dropna(subset=["_parsed_time"])
            # convert to numeric ordinal for correlation (seconds)
            time_numeric = df_time["_parsed_time"].view("int64") // 10**9
            # analyze numeric cols
            for col in df_time.select_dtypes(include="number").columns:
                if col == "_parsed_time":
                    continue
                series = df_time[col].astype(float)
                if series.dropna().shape[0] < 5:
                    continue
                corr = np.corrcoef(time_numeric, series.fillna(series.mean()))[0, 1]
                if not np.isnan(corr) and abs(corr) >= 0.25:
                    direction = "ascendente" if corr > 0 else "descendente"
                    report["trends"][col] = {
                        "corr_with_time": float(corr),
                        "direction": direction,
                    }
            # seasonal: monthly mean if dt spread > 60 days and enough points
            if (
                df_time["_parsed_time"].max() - df_time["_parsed_time"].min()
            ).days >= 60 and len(df_time) >= 20:
                df_time = df_time.set_index("_parsed_time")
                monthly = df_time.select_dtypes(include="number").resample("M").mean()
                report["notes"].append(
                    "Também foi gerado resumo mensal (médias) — pode haver padrões sazonais."
                )
                report["monthly_summary_head"] = monthly.head(6).to_dict(orient="index")
            return report
    except Exception:
        report["notes"].append(
            "Falha ao converter a coluna de tempo para datetime; será usada versão numérica quando possível."
        )

    # If not datetime, try to use numeric time
    try:
        time_numeric = pd.to_numeric(df[time_col], errors="coerce")
        if time_numeric.notna().sum() / len(time_numeric) < 0.3:
            report["notes"].append(
                "A coluna de tempo não possui valores numéricos suficientes para análise de tendência."
            )
            return report
        for col in df.select_dtypes(include="number").columns:
            if col == time_col:
                continue
            series = df[col].astype(float)
            if series.dropna().shape[0] < 5:
                continue
            corr = np.corrcoef(
                time_numeric.fillna(time_numeric.mean()), series.fillna(series.mean())
            )[0, 1]
            if not np.isnan(corr) and abs(corr) >= 0.25:
                direction = "ascendente" if corr > 0 else "descendente"
                report["trends"][col] = {
                    "corr_with_time": float(corr),
                    "direction": direction,
                }
        return report
    except Exception:
        report["notes"].append(
            "Não foi possível analisar tendências a partir da coluna de tempo."
        )
        return report


if uploaded_file:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    file_bytes = uploaded_file.getvalue()
    df, converted_cols = load_and_clean_data(file_bytes)

    if df is not None:
        st.success(f"Arquivo '{uploaded_file.name}' carregado! Shape: {df.shape}")

        if converted_cols:
            st.info(
                f"Colunas convertidas para tipo numérico: `{'`, `'.join(converted_cols)}`"
            )

        # --- Análise Exploratória (EDA) ---
        st.header("🔎 Análise Exploratória dos Dados")
        tab_preview, tab_stats, tab_types, tab_viz = st.tabs(
            ["Pré-visualização", "Estatísticas", "Tipos de Dados", "Visualizações"]
        )

        with tab_preview:
            st.dataframe(df.head())

        with tab_stats:
            st.dataframe(df.describe(include="all").T)

        with tab_types:
            df_types = df.dtypes.reset_index().rename(
                columns={"index": "Coluna", 0: "Tipo"}
            )
            df_types["Tipo"] = df_types["Tipo"].astype(str)
            st.dataframe(df_types)

        with tab_viz:
            st.subheader("📊 Distribuições das Variáveis Numéricas")
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                col_to_plot = st.selectbox(
                    "Selecione uma coluna para ver a distribuição:", numeric_cols
                )
                if col_to_plot:
                    fig, ax = plt.subplots()
                    sns.histplot(df[col_to_plot], kde=True, ax=ax)
                    st.pyplot(fig)
            else:
                st.warning(
                    "Nenhuma coluna numérica foi encontrada ou pôde ser convertida."
                )

            st.subheader("🔥 Mapa de Calor de Correlação")
            if len(numeric_cols) > 1:
                fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
                corr_cols = numeric_cols[:20]
                corr = df[corr_cols].corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
                st.pyplot(fig_corr)
            else:
                st.write(
                    "Não há colunas numéricas suficientes para gerar mapa de correlação."
                )

        # --- Chat com os Dados (Q&A) ---
        st.header("❓ Converse com seus Dados")
        user_question = st.text_input(
            "Faça uma pergunta:",
            placeholder="Ex: Qual o valor médio da coluna 'amount'?",
        )

        if user_question:
            print(f"[DEBUG] Pergunta recebida: {user_question}")
            with st.spinner("Analisando..."):
                qlow = user_question.lower()
                resultado = None
                resposta = None

                try:
                    # --- Conclusões do histórico ---
                    if "conclus" in qlow:
                        history_text = (
                            "\n".join(
                                [
                                    f"Pergunta: {q}\nResposta: {a}"
                                    for q, a in st.session_state.chat_history
                                ]
                            )
                            if st.session_state.chat_history
                            else "Nenhuma pergunta foi feita ainda."
                        )

                        summary_prompt = (
                            "Você é um analista de dados explorando um dataset em CSV.\n\n"
                            f"Histórico de interações:\n{history_text}\n\n"
                            "Escreva conclusões objetivas e específicas sobre os dados, "
                            "destacando tendências, padrões ou possíveis problemas. "
                            "Não fale de 'usuários' ou 'perguntas', apenas dos dados."
                        )
                        try:
                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "Você é um assistente de análise de dados.",
                                    },
                                    {"role": "user", "content": summary_prompt},
                                ],
                            )
                            resposta = response.choices[0].message.content
                        except Exception as e:
                            resposta = f"[Erro ao gerar conclusões: {e}]"

                        st.write("📌 **Conclusões até agora:**")
                        st.markdown(resposta)

                    # --- Tipos de dados ---
                    elif "tipo" in qlow and (
                        "dado" in qlow or "coluna" in qlow and "cluster" not in qlow
                    ):
                        metadata = get_metadata(df)
                        resposta_modelo = perguntar_modelo(qlow, metadata)
                        resultado = executar_resposta(resposta_modelo, df)

                        # Renderiza resultado
                        if not ("erro" in str(resultado).lower()):
                            resposta = resultado
                            st.write("💡 **Resposta via GPT/PandasAI:**")
                            if isinstance(resposta, (dict, list)):
                                st.json(resposta)
                            else:
                                st.write(resposta)

                    # --- Outliers ---
                    elif any(
                        word in qlow
                        for word in ["atípic", "outlier", "anomalia", "discrepante"]
                    ):
                        outlier_summary = {}
                        for col in df.select_dtypes(include=["number"]).columns:
                            q1, q3 = df[col].quantile([0.25, 0.75])
                            iqr = q3 - q1
                            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                            mask = (df[col] < lower) | (df[col] > upper)
                            n_outliers = int(mask.sum())

                            if n_outliers > 0:
                                percent = round((n_outliers / len(df[col])) * 100, 2)
                                valores = df.loc[mask, col].tolist()
                                resumo = (
                                    {
                                        "quantidade": n_outliers,
                                        "percentual": f"{percent}%",
                                        "menor_outlier": min(valores),
                                        "maior_outlier": max(valores),
                                        "mediana_outliers": float(
                                            pd.Series(valores).median()
                                        ),
                                    }
                                    if n_outliers > 20
                                    else {
                                        "quantidade": n_outliers,
                                        "percentual": f"{percent}%",
                                        "valores": valores,
                                    }
                                )
                                outlier_summary[col] = resumo

                        if outlier_summary:
                            st.write("💡 **Resumo de valores atípicos detectados:**")
                            st.json(outlier_summary)
                            st.info("Obs.: Método do IQR (Intervalo Interquartil).")
                        else:
                            st.success(
                                "✅ Não foram detectados valores atípicos relevantes."
                            )
                        resposta = outlier_summary

                    # --- Impacto dos outliers ---
                    elif any(
                        pal in qlow for pal in ["como", "impacto", "efeito", "influenc"]
                    ) and any(
                        word in qlow
                        for word in ["outlier", "atípic", "anomalia", "discrepante"]
                    ):
                        resposta = (
                            "📊 Os valores atípicos podem afetar a análise:\n\n"
                            "- **Estatísticas:** distorcem média e desvio padrão.\n"
                            "- **Modelos:** alguns algoritmos são sensíveis.\n"
                            "- **Visualizações:** comprimem e escondem padrões.\n"
                            "- **Negócio:** podem ser erros ou eventos raros importantes.\n\n"
                            "👉 **Boas práticas:** investigar origem, usar medidas robustas (mediana), transformar ou tratar conforme objetivo."
                        )
                        st.write("💡 **Resposta:**")
                        st.markdown(resposta)

                    elif any(
                        word in qlow for word in ["relacion", "dispers", "cruzad"]
                    ):
                        numeric_cols = df.select_dtypes(
                            include="number"
                        ).columns.tolist()
                        if len(numeric_cols) >= 2:
                            st.write("💡 **Gráficos de Dispersão (Pairplot)**")
                            fig = sns.pairplot(df[numeric_cols], diag_kind="kde")
                            st.pyplot(fig)

                            cat_cols = df.select_dtypes(
                                exclude="number"
                            ).columns.tolist()
                            if len(cat_cols) >= 2:
                                st.write(
                                    "💡 **Tabela Cruzada (exemplo com 2 primeiras colunas categóricas):**"
                                )
                                st.dataframe(
                                    pd.crosstab(df[cat_cols[0]], df[cat_cols[1]])
                                )
                            else:
                                st.info(
                                    "Não há colunas categóricas suficientes para gerar tabela cruzada."
                                )
                            resposta = "Gráficos de dispersão e tabela cruzada gerados."
                        else:
                            st.warning("Não há colunas numéricas suficientes.")

                    elif any(
                        word in qlow
                        for word in [
                            "média",
                            "media",
                            "soma",
                            "máximo",
                            "maximo",
                            "mínimo",
                            "minimo",
                        ]
                    ):
                        match = re.search(r"coluna\s+'?([\w_]+)'?", qlow)
                        if match:
                            col = match.group(1)
                            if col in df.columns:
                                if "média" in qlow or "media" in qlow:
                                    valor, metrica = df[col].mean(), "média"
                                elif "soma" in qlow:
                                    valor, metrica = df[col].sum(), "soma"
                                elif "máximo" in qlow or "maximo" in qlow:
                                    valor, metrica = df[col].max(), "máximo"
                                elif "mínimo" in qlow or "minimo" in qlow:
                                    valor, metrica = df[col].min(), "mínimo"
                                resposta = f"📊 O {metrica} da coluna `{col}` é: **{valor:.2f}**"
                                st.write(resposta)
                            else:
                                st.warning(f"A coluna `{col}` não foi encontrada.")
                        else:
                            st.warning(
                                "Não foi possível identificar a coluna mencionada."
                            )

                    # --- Fallback: PandasAI ---
                    else:
                        sdf = SmartDataframe(
                            df,
                            config={
                                "llm": llm,
                                "description": (
                                    "Você é um assistente de análise de dados com Pandas.\n"
                                    "Sempre gere código Python válido e termine com `result = ...`.\n"
                                    "Nunca use bibliotecas externas (os, io, sys, base64).\n"
                                    "Para gráficos, use matplotlib/seaborn e finalize com `result = fig`."
                                ),
                                "save_charts": False,
                                "enable_cache": False,
                                "enforce_code_execution_safety": False,
                                "custom_plots": True,
                            },
                        )
                        try:
                            response = sdf.chat(user_question)
                            resposta = response
                        except (NoCodeFoundError, NoResultFoundError):
                            resposta = (
                                "⚠️ O agente não conseguiu gerar código/resultados para essa pergunta. "
                                "Tente reformular ou use as opções automáticas."
                            )
                        except Exception as e:
                            st.error(f"Erro inesperado no PandasAI: {e}")
                            resposta = f"[Erro PandasAI: {e}]"

                        # Renderização de resposta do PandasAI
                        if resposta is not None:
                            st.write("💡 **Resposta:**")
                            if isinstance(resposta, (str, int, float)):
                                st.write(resposta)
                            elif isinstance(resposta, (pd.DataFrame, pd.Series)):
                                st.dataframe(
                                    resposta
                                    if isinstance(resposta, pd.DataFrame)
                                    else resposta.to_frame()
                                )
                            elif isinstance(resposta, plt.Figure):
                                # Salva em memória e mostra
                                buf = BytesIO()
                                resposta.savefig(buf, format="png", bbox_inches="tight")
                                buf.seek(0)
                                st.image(buf, caption="Gráfico gerado")
                                plt.close(resposta)
                            elif (
                                isinstance(resposta, dict)
                                and resposta.get("type") == "plot"
                            ):
                                val = resposta.get("value")
                                if isinstance(val, plt.Figure):
                                    buf = BytesIO()
                                    val.savefig(buf, format="png", bbox_inches="tight")
                                    buf.seek(0)
                                    st.image(buf, caption="Gráfico gerado")
                                    plt.close(val)
                                elif isinstance(val, str):
                                    try:
                                        st.image(val, caption="Gráfico gerado")
                                    except Exception:
                                        st.write("Formato de plot não reconhecido.")
                            elif isinstance(resposta, (list, dict)):
                                st.json(resposta)
                            else:
                                st.write(str(resposta))
                except Exception as e:
                    print(f"[DEBUG] Erro no interpretador GPT: {e}")

                # salvar historico
                if resposta is not None:
                    st.session_state.chat_history.append((user_question, str(resposta)))

else:
    st.info("Aguardando o upload de um arquivo CSV para iniciar a análise.")
