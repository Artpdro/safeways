import streamlit as st
import pandas as pd
import json
import pickle
from datetime import datetime, date, time
from preditor_ofc import AccidentPredictor

# -------------------------
# Carregar o modelo treinado
# -------------------------
@st.cache_resource
def load_model(pickle_path="modelo_acidentes.pkl"):
    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        st.error("Arquivo de modelo não encontrado: 'modelo_acidentes.pkl'. Treine/salve o modelo primeiro.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

    predictor = AccidentPredictor()

    # Compatibilidade com diferentes formatos de pickle
    predictor.modelo = data.get("modelo", data.get("model", predictor.modelo))
    predictor.encoders = data.get("encoders", getattr(data, "encoders", {}))
    predictor.feature_names = data.get("features", data.get("feature_names", getattr(data, "feature_names", [])))
    predictor.best_params = data.get("params", data.get("best_params", {}))
    # Carrega métricas de teste para exibir ao usuário
    predictor.r2_score_test = data.get("r2_test", data.get("r2", None))
    predictor.rmse_score_test = data.get("rmse_test", data.get("rmse", None))
    # Carrega o histórico (necessário para calcular lags e médias na previsão e para o contexto)
    predictor.historical_df = data.get("historical_df", None)
    predictor.treinado = True
    return predictor

# -------------------------
# Carregar opções auxiliares
# -------------------------
@st.cache_data
def load_options():
    try:
        with open("uf_options.json", "r", encoding="utf-8") as f:
            uf_options = json.load(f)
    except Exception:
        uf_options = []

    try:
        with open("municipios_por_uf.json", "r", encoding="utf-8") as f:
            municipios_por_uf = json.load(f)
    except Exception:
        municipios_por_uf = {}

    try:
        with open("condicoes_metereologicas_options.json", "r", encoding="utf-8") as f:
            condicoes = json.load(f)
    except Exception:
        # Padrão de opções de clima simplificadas (Baseado na lógica de _simplificar_clima)
        condicoes = ["Chuva", "Nublado", "Bom", "Vento", "Nevoeiro/Neblina", "Outro"]

    return uf_options, municipios_por_uf, condicoes

# -------------------------
# Função utilitária para determinar data automática
# -------------------------
def escolher_data_automatica(pred):
    """Escolhe a próxima data de previsão."""
    try:
        hist = pred.historical_df
        if isinstance(hist, pd.DataFrame) and "data" in hist.columns and not hist.empty:
            hist_datas = pd.to_datetime(hist["data"], errors='coerce')
            ultima = hist_datas.max()
            if pd.notna(ultima):
                proxima = (ultima + pd.Timedelta(days=1)).date()
                return proxima.strftime("%d/%m/%Y")
    except Exception:
        pass
    # fallback: hoje
    return date.today().strftime("%d/%m/%Y")


# -------------------------
# App
# -------------------------
st.set_page_config(page_title="Preditor de Acidentes", layout="wide")
st.title("🛣️ Preditor de Acidentes de Trânsito")

predictor = load_model()
uf_options, municipios_por_uf, condicoes_options = load_options()

if not predictor:
    st.warning("Modelo não carregado. Verifique o arquivo 'modelo_acidentes.pkl'.")
    st.stop()
else:
    # Exibe métricas de teste (para dar confiança ao usuário)
    if predictor.treinado and predictor.r2_score_test is not None:
        st.sidebar.markdown("### Info do Modelo")
        st.sidebar.info(f"R² (Teste): **{predictor.r2_score_test:.4f}**\n\nRMSE (Teste): **{predictor.rmse_score_test:.2f}**")


# -------------------------
# Interface de Input
# -------------------------
with st.container():
    st.markdown("Preencha os campos abaixo e clique em **Prever Acidentes**.")
    c1, c2, c3 = st.columns(3)

    with c1:
        uf = st.selectbox("UF", uf_options, index=uf_options.index("PE") if "PE" in uf_options else 0)
    
    with c2:
        municipios_filtrados = municipios_por_uf.get(uf, ["DESCONHECIDO"]) if municipios_por_uf else ["DESCONHECIDO"]
        municipio = st.selectbox("Município", municipios_filtrados, index=municipios_filtrados.index("RECIFE") if "RECIFE" in municipios_filtrados else 0)
    
    with c3:
        # Corrigido para permitir a escolha de qualquer horário
        horario_obj = st.time_input("Horário (HH:MM:SS)", value=time(12, 0, 0))
        horario_str = horario_obj.strftime("%H:%M:%S")

    condicao = st.selectbox("Condição Meteorológica", condicoes_options, index=condicoes_options.index("Bom") if "Bom" in condicoes_options else 0)


# -------------------------
# Previsão
# -------------------------
if st.button("Prever Acidentes", type="primary"):
    try:
        data_inversa = escolher_data_automatica(predictor)
        
        # Define um valor padrão para 'tipo_acidente'
        tipo_default = "DESCONHECIDO"
        if "tipo_acidente" in predictor.encoders:
            try:
                tipo_default = predictor.encoders["tipo_acidente"].classes_[0]
            except Exception:
                tipo_default = "DESCONHECIDO"

        # Define a hora média para o input (crucial para o modelo)
        hora_media_pred = horario_obj.hour 

        df_input = pd.DataFrame([{
            "data_inversa": data_inversa,
            "horario": horario_str,
            "uf": uf,
            "municipio": municipio,
            "tipo_acidente": tipo_default,
            "condicao_metereologica": condicao,
            "hora_media": hora_media_pred # Feature para o modelo
        }])
        
        # Chama a função de previsão do preditor_ofc
        resultado_array = predictor.prever(df_input) # É necessário implementar 'prever' no preditor_ofc.py
        
        previsao_val = resultado_array[-1]
        previsao_int = int(max(0,(previsao_val))) 
        
        # -------------------------
        # Mensagem principal (RESULTADO)
        # -------------------------
        st.markdown("---")
        st.subheader(f"Resultado da Previsão para {data_inversa} às {horario_str}:")
        
        if previsao_int == 0:
            st.success(f"✅ Previsão: **0 acidentes**.\n\nSem acidentes previstos para as condições informadas em **{municipio}**.")
        else:
            st.warning(f"🚨 Previsão: **{previsao_int} acidentes**.\n\nAcidentes previstos em **{municipio}** (UF: {uf}) nas condições de **{condicao}**.")

        # -------------------------
        # Contexto Histórico e Comparação Percentual
        # -------------------------
        with st.expander("📊 Análise e Contexto Histórico", expanded=True):
            contexto_msgs = []
            
            # --- Cálculo da Média e Porcentagem de Risco ---
            ref = None
            try:
                hist = predictor.historical_df
                if isinstance(hist, pd.DataFrame) and "acidentes" in hist.columns:
                    media_global = hist["acidentes"].mean()
                    
                    if "municipio" in hist.columns and municipio in hist["municipio"].unique():
                        media_mun = hist.loc[hist["municipio"] == municipio, "acidentes"].mean()
                        contexto_msgs.append(f"Média histórica de acidentes em **{municipio}**: **{media_mun:.2f}** acidentes/dia.")
                        ref = media_mun
                    else:
                        contexto_msgs.append(f"Média histórica geral (todos os municípios): **{media_global:.2f}** acidentes/dia.")
                        ref = media_global

                    # Comparação Percentual (o "Risco" em % da média)
                    if ref is not None and ref > 0:
                        diff = (previsao_int - ref) / ref * 100
                        sentido = "acima" if diff > 0 else "abaixo" if diff < 0 else "igual à"
                        
                        # EXIBIÇÃO DO % DE RISCO/DIFERENÇA
                        if abs(diff) > 0.1:
                            st.markdown(f"**Risco em Relação à Média Histórica de {municipio}**: A previsão é **{abs(diff):.1f}% {sentido}** da média de referência ({ref:.2f}).")
                        else:
                            st.markdown(f"**Risco em Relação à Média Histórica de {municipio}**: A previsão é **praticamente igual** à média de referência ({ref:.2f}).")
                    elif ref == 0:
                         st.info("A média histórica de acidentes é zero neste contexto, não sendo possível a comparação percentual de risco.")
                    
                    # --- Histórico Recente ---
                    if "data" in hist.columns:
                        ult7 = hist.sort_values("data", ascending=False).head(7)[["data", "acidentes"]]
                        ult7 = ult7.assign(data=lambda d: pd.to_datetime(d["data"], errors='coerce').dt.strftime("%d/%m/%Y"))
                        
                        st.markdown("---")
                        st.markdown("##### Histórico de Acidentes (Últimos 7 dias no *dataset*):")
                        
                        ult7_list = [f"- {row['data']}: **{int(row['acidentes'])}**" for _, row in ult7.iterrows() if pd.notna(row['data'])]
                        if ult7_list:
                             st.markdown("\n".join(ult7_list))

                # Exibir as mensagens de contexto (média local/global)
                if contexto_msgs:
                    st.markdown("---")
                    st.markdown("##### Médias de Referência:")
                    st.markdown("\n".join([f"* {msg}" for msg in contexto_msgs]))
                    
            except Exception as e:
                st.info(f"Dados históricos indisponíveis para comparação. {e}")

    except Exception as e:
        st.error(f"Erro durante a previsão: {e}")
        st.info("Verifique se o modelo foi treinado e se o arquivo 'modelo_acidentes.pkl' está acessível e se o método 'prever' existe em 'preditor_ofc.py'.")