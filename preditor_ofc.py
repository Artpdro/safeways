import pandas as pd
import numpy as np
import json
import pickle
import lightgbm as lgb
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import holidays
import warnings
import itertools

warnings.filterwarnings("ignore")


class AccidentPredictor:
    def __init__(self):
        self.modelo = lgb.LGBMRegressor(random_state=42)
        self.encoders = {}
        self.feature_names = []
        self.treinado = False
        self.best_params = {}
        self.r2_score_train = None
        self.rmse_score_train = None
        self.r2_score_test = None
        self.rmse_score_test = None
        self.holidays_br = holidays.Brazil()
        self.historical_df = None

    def _simplificar_clima(self, cond):
        if any(k in cond for k in ["Chuva", "Garoa"]):
            return "Chuva"
        if "Nublado" in cond:
            return "Nublado"
        if any(k in cond for k in ["Céu Claro", "Sol", "Bom"]):
            return "Bom"
        if "Vento" in cond:
            return "Vento"
        if any(k in cond for k in ["Nevoeiro", "Neblina"]):
            return "Nevoeiro/Neblina"
        return "Outro"

    def _processar_dados(self, df):
        df["data"] = pd.to_datetime(df["data_inversa"], format="%d/%m/%Y", errors="coerce")
        df = df[df["data"].dt.year >= 2019].dropna(subset=["data", "horario", "uf", "municipio", "tipo_acidente", "condicao_metereologica"])
        df["hora"] = pd.to_datetime(df["horario"], format="%H:%M:%S", errors="coerce").dt.hour
        df.dropna(subset=["hora"], inplace=True)
        df["condicao_metereologica"] = df["condicao_metereologica"].apply(self._simplificar_clima)

        agg = df.groupby("data").agg(
            acidentes=("data_inversa", "count"),
            uf=("uf", lambda x: x.mode()[0]),
            municipio=("municipio", lambda x: x.mode()[0]),
            tipo_acidente=("tipo_acidente", lambda x: x.mode()[0]),
            condicao_metereologica=("condicao_metereologica", lambda x: x.mode()[0]),
            hora_media=("hora", "mean")
        ).reset_index()

        agg["clima"] = agg["condicao_metereologica"].apply(self._simplificar_clima)
        agg = agg.drop(columns=["condicao_metereologica"])
        return agg.sort_values("data").reset_index(drop=True)

    def _criar_features(self, df):
        df["ano"] = df["data"].dt.year
        df["mes"] = df["data"].dt.month
        df["dia_semana"] = df["data"].dt.dayofweek
        df["dia_ano"] = df["data"].dt.dayofyear
        df["semana"] = df["data"].dt.isocalendar().week.astype(int)
        df["fim_semana"] = (df["dia_semana"] >= 5).astype(int)
        df["feriado"] = df["data"].apply(lambda x: int(x in self.holidays_br))
        df["feriado_fim_semana"] = df["feriado"] * df["fim_semana"]

        df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
        df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)
        df["dia_ano_sin"] = np.sin(2 * np.pi * df["dia_ano"] / 365.25)
        df["dia_ano_cos"] = np.cos(2 * np.pi * df["dia_ano"] / 365.25)

        if 'acidentes' in df.columns:
            for lag in [1, 2, 7, 14]:
                df[f"lag_{lag}"] = df["acidentes"].shift(lag)
            for w in [7, 14, 28]:
                df[f"media_{w}"] = df["acidentes"].shift(1).rolling(w, min_periods=1).mean()
                df[f"std_{w}"] = df["acidentes"].shift(1).rolling(w, min_periods=1).std()
        else:
            for lag in [1, 2, 7, 14]:
                df[f"lag_{lag}"] = 0
            for w in [7, 14, 28]:
                df[f"media_{w}"] = 0
                df[f"std_{w}"] = 0

        df.fillna(0, inplace=True)

        for col in ["uf", "municipio", "tipo_acidente", "clima"]:
            if col in df.columns:
                if col in self.encoders:
                    enc = self.encoders[col]
                    df.loc[:, f"{col}_enc"] = df[col].apply(lambda x: enc.transform([x])[0] if x in enc.classes_ else -1)
                else:
                    enc = LabelEncoder()
                    df.loc[:, f"{col}_enc"] = enc.fit_transform(df[col])
                    self.encoders[col] = enc
            else:
                df.loc[:, f"{col}_enc"] = 0

        features = [
            "ano", "mes", "dia_semana", "dia_ano", "semana", "fim_semana",
            "dia_semana_sin", "dia_semana_cos", "dia_ano_sin", "dia_ano_cos",
            "hora_media", "feriado", "feriado_fim_semana"
        ] + [f"lag_{i}" for i in [1, 2, 7, 14]] + \
            [f"media_{i}" for i in [7, 14, 28]] + \
            [f"std_{i}" for i in [7, 14, 28]] + \
            [f"{c}_enc" for c in ["uf", "municipio", "tipo_acidente", "clima"]]

        y = df["acidentes"] if "acidentes" in df.columns else None
        return df[features], y

    def treinar(self, arquivo_json):
        with open(arquivo_json, "r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))

        df = self._processar_dados(df)
        self.historical_df = df.copy() # Salva o df agregado para usar no contexto histórico
        X, y = self._criar_features(df)
        self.feature_names = X.columns.tolist()

        # Separação temporal 70% treino / 30% teste
        split_index = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        # Treinar modelo
        self.modelo.fit(X_train, y_train)

        # Avaliar desempenho
        y_pred_train = np.clip(np.round(self.modelo.predict(X_train)), 0, None)
        y_pred_test = np.clip(np.round(self.modelo.predict(X_test)), 0, None)

        self.r2_score_train = r2_score(y_train, y_pred_train)
        self.rmse_score_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        self.r2_score_test = r2_score(y_test, y_pred_test)
        self.rmse_score_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        self.treinado = True

        print(f"Treino (70%): R²={self.r2_score_train:.4f}, RMSE={self.rmse_score_train:.2f}")
        print(f"Teste  (30%): R²={self.r2_score_test:.4f}, RMSE={self.rmse_score_test:.2f}")

    def salvar_modelo(self, nome="modelo_acidentes.pkl"):
        if not self.treinado:
            raise RuntimeError("Treine o modelo antes de salvar.")

        with open(nome, "wb") as f:
            pickle.dump({
                "modelo": self.modelo,
                "encoders": self.encoders,
                "features": self.feature_names,
                "r2_train": self.r2_score_train,
                "rmse_train": self.rmse_score_train,
                "r2_test": self.r2_score_test,
                "rmse_test": self.rmse_score_test,
                "historical_df": self.historical_df
            }, f)
        print(f"Modelo salvo: {nome}")
        
    def prever(self, df_input):
        """
        Realiza a previsão em um novo DataFrame de input (não-agregado).
        Espera colunas: data_inversa, horario, uf, municipio, tipo_acidente, condicao_metereologica, hora_media.
        """
        if not self.treinado:
            raise RuntimeError("O modelo não foi treinado.")
        
        # 1. Copia o DF de entrada
        df = df_input.copy()
        
        # 2. Adiciona colunas para feature engineering
        df["data"] = pd.to_datetime(df["data_inversa"], format="%d/%m/%Y", errors="coerce")
        df["clima"] = df["condicao_metereologica"].apply(self._simplificar_clima)
        
        # 3. Cria as features com base na lógica de treinamento
        X_input, _ = self._criar_features(df)
        
        # 4. Filtra apenas as features que o modelo espera e prevê
        X_predict = X_input[self.feature_names]

        # 5. Previsão
        resultado = self.modelo.predict(X_predict)
        
        # Arredonda e garante que não é negativo
        resultado_clipado = np.clip(np.round(resultado), 0, None).astype(int)
        
        return resultado_clipado


if __name__ == "__main__":
    predictor = AccidentPredictor()
    predictor.treinar("datatran_consolidado.json")
    predictor.salvar_modelo()