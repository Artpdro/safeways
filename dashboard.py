# dashboard.py
import os
import json
import math
import hashlib
import pandas as pd
import folium
from folium.plugins import HeatMap

# --- Centróides aproximados das UFs (usados como fallback) ---
UF_CENTROIDS = {
    "AC": [-8.77, -70.55], "AL": [-9.62, -35.73], "AP": [1.41, -51.77], "AM": [-3.13, -60.02],
    "BA": [-12.96, -38.51], "CE": [-3.73, -38.54], "DF": [-15.79, -47.88], "ES": [-20.31, -40.34],
    "GO": [-16.64, -49.31], "MA": [-2.53, -44.34], "MT": [-12.64, -55.42], "MS": [-20.51, -54.54],
    "MG": [-19.92, -43.94], "PA": [-1.45, -48.50], "PB": [-7.12, -34.86], "PR": [-25.43, -49.27],
    "PE": [-8.05, -34.90], "PI": [-5.09, -42.80], "RJ": [-22.91, -43.17], "RN": [-5.79, -35.20],
    "RS": [-30.03, -51.23], "RO": [-10.83, -63.34], "RR": [2.82, -60.67], "SC": [-27.59, -48.55],
    "SP": [-23.55, -46.63], "SE": [-10.90, -37.07], "TO": [-10.25, -48.32]
}

def _deterministic_jitter(seed_text: str, lat_range=0.6, lon_range=0.9):
    """Gera deslocamento determinístico (reprodutível) a partir de um texto."""
    h = hashlib.md5(seed_text.encode('utf-8')).hexdigest()
    lat_frac = int(h[0:8], 16) / 0xFFFFFFFF
    lon_frac = int(h[8:16], 16) / 0xFFFFFFFF
    lat_off = (lat_frac - 0.5) * 2 * lat_range
    lon_off = (lon_frac - 0.5) * 2 * lon_range
    return lat_off, lon_off

def _try_parse_point_string(s):
    """Tenta extrair lat,lon de strings comuns (POINT(), 'lat,lon', 'lon lat')."""
    try:
        s2 = str(s).strip()
        if not s2:
            return None
        # POINT(lon lat) ou POINT (lon lat)
        if s2.lower().startswith("point"):
            inside = s2[s2.find("(")+1:s2.rfind(")")]
            parts = inside.replace(",", " ").split()
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                return lat, lon
        # formato "lat,lon" ou "lon,lat"
        if "," in s2:
            a, b = s2.split(",", 1)
            try:
                a_f = float(a.strip())
                b_f = float(b.strip())
                # heurística de ordem: lat geralmente entre -40..10, lon entre -80..-30
                if -40 <= a_f <= 10 and -80 <= b_f <= -30:
                    return a_f, b_f
                if -40 <= b_f <= 10 and -80 <= a_f <= -30:
                    return b_f, a_f
                # se não tiver certeza, só retorna (a,b)
                return a_f, b_f
            except Exception:
                pass
        # formato "lat lon" (espaço)
        parts = s2.split()
        if len(parts) >= 2:
            try:
                p1 = float(parts[0]); p2 = float(parts[1])
                if -40 <= p1 <= 10 and -80 <= p2 <= -30:
                    return p1, p2
                if -40 <= p2 <= 10 and -80 <= p1 <= -30:
                    return p2, p1
            except Exception:
                pass
    except Exception:
        pass
    return None

def create_heatmap():
    """
    Lê 'datatran_consolidado.json' no diretório atual e gera:
    - um HeatMap ponderado por número de registros por município (quando não há coordenadas por registro),
    - OU um HeatMap direto por pontos (se o JSON tiver latitude/longitude por registro).
    Retorna HTML (string) com o mapa e um painel lateral com quantitativos.
    """
    datapath = "datatran_consolidado.json"
    municipios_coords_file = "municipios_coords.json"  # opcional (se tiver coords reais por cidade)

    try:
        if not os.path.exists(datapath):
            return '<div style="padding:20px;">Arquivo <b>datatran_consolidado.json</b> não encontrado no diretório.</div>'

        # carregar JSON
        with open(datapath, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        df = pd.DataFrame(raw)
        if df.empty:
            return '<div style="padding:20px;">Nenhum registro encontrado em datatran_consolidado.json.</div>'

        # identificar colunas possíveis
        cols_lower = {c.lower(): c for c in df.columns}

        # tentar extrair lat/lon por registro
        lat_cols = [k for k in cols_lower if any(x in k for x in ['latitude','lat','y'])]
        lon_cols = [k for k in cols_lower if any(x in k for x in ['longitude','lon','lng','long','x'])]
        coords_points = []

        if lat_cols and lon_cols:
            latc = cols_lower[lat_cols[0]]
            lonc = cols_lower[lon_cols[0]]
            for _, row in df.iterrows():
                try:
                    lat = float(row.get(latc))
                    lon = float(row.get(lonc))
                    if math.isfinite(lat) and math.isfinite(lon):
                        coords_points.append([lat, lon, 1])
                except Exception:
                    pass

        # se não encontrou lat/lon explícitos, tentar colunas tipo geometry/point
        if not coords_points:
            geom_candidates = [k for k in cols_lower if any(x in k for x in ['geom','geometry','point','coord','coorden'])]
            for g in geom_candidates:
                for _, row in df.iterrows():
                    parsed = _try_parse_point_string(row.get(cols_lower[g], ''))
                    if parsed:
                        lat, lon = parsed
                        coords_points.append([lat, lon, 1])

        # Se encontramos pontos diretos, desenhar heatmap com eles e resumir total
        if coords_points:
            m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4, tiles='OpenStreetMap')
            HeatMap(coords_points, radius=6, blur=10, min_opacity=0.25).add_to(m)
            total = len(coords_points)
            summary_html = f"<div style='padding:12px;font-family:Segoe UI, sans-serif;'><h3>Total registros</h3><p style='font-size:18px'>{total}</p></div>"
            return f'<div style="display:flex;gap:12px;align-items:flex-start">' \
                   f'<div style="flex:1">{m._repr_html_()}</div>' \
                   f'<div style="width:320px">{summary_html}</div></div>'

        # --- Caso não haja coordenadas ponto-a-ponto: agregação por UF + Município ---
        # identificar colunas UF e Município
        uf_col = None
        mun_col = None
        for low, orig in cols_lower.items():
            if low == 'uf' or low.startswith('uf'):
                uf_col = orig
            if 'municipio' in low or 'municip' in low or 'cidade' in low:
                mun_col = orig

        # heurística adicional
        if uf_col is None:
            for low, orig in cols_lower.items():
                if len(low) <= 3 and low.isalpha():
                    uf_col = orig
                    break

        if mun_col is None:
            for low, orig in cols_lower.items():
                if 'nome' in low:
                    mun_col = orig
                    break

        if uf_col is None or mun_col is None:
            return '<div style="padding:20px;">Não foi possível identificar colunas de <b>UF</b> e <b>Município</b> no JSON.</div>'

        # normalizar strings e agregar
        df['uf_norm'] = df[uf_col].astype(str).str.strip().str.upper()
        df['mun_norm'] = df[mun_col].astype(str).str.strip()
        agg = df.groupby(['uf_norm', 'mun_norm']).size().reset_index(name='count')
        if agg.empty:
            return '<div style="padding:20px;">Nenhum registro válido após agregação por UF/Município.</div>'

        # tentar carregar coords reais por município se existir arquivo
        municipios_coords = {}
        if os.path.exists(municipios_coords_file):
            try:
                with open(municipios_coords_file, 'r', encoding='utf-8') as f:
                    municipios_coords = json.load(f)
            except Exception:
                municipios_coords = {}

        # preparar pontos ponderados por count
        heat_points = []
        m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4, tiles='OpenStreetMap')
        markers = folium.FeatureGroup(name="Cidades", show=False)

        for _, row in agg.iterrows():
            uf = row['uf_norm']
            mun = row['mun_norm']
            cnt = int(row['count'])
            lat = lon = None

            # procurar em municipios_coords: estrutura esperada { "SP": {"São Paulo":[lat,lon], ...}, ... }
            try:
                if municipios_coords:
                    if uf in municipios_coords and mun in municipios_coords[uf]:
                        lat, lon = municipios_coords[uf][mun]
                    elif f"{uf}|{mun}" in municipios_coords:
                        lat, lon = municipios_coords[f"{uf}|{mun}"]
                    elif mun in municipios_coords:
                        lat, lon = municipios_coords[mun]
            except Exception:
                lat = lon = None

            # fallback: centróide + jitter determinístico
            if lat is None or lon is None:
                centro = UF_CENTROIDS.get(uf, [-14.2350, -51.9253])
                lat_off, lon_off = _deterministic_jitter(f"{uf}|{mun}", lat_range=0.6, lon_range=0.9)
                lat = centro[0] + lat_off
                lon = centro[1] + lon_off

            heat_points.append([lat, lon, cnt])

            folium.CircleMarker(
                location=(lat, lon),
                radius=max(3, min(12, math.sqrt(cnt) * 1.5)),
                popup=f"{mun} ({uf}) — {cnt} registros",
                tooltip=f"{mun} — {cnt}",
                fill=True,
                fill_opacity=0.8,
                weight=0
            ).add_to(markers)

        if heat_points:
            HeatMap(heat_points, radius=10, blur=18, min_opacity=0.2, max_zoom=6).add_to(m)
            markers.add_to(m)
            folium.LayerControl(collapsed=True).add_to(m)

        # painel lateral com resumo: top 15 municípios e total por UF
        top_n = agg.sort_values('count', ascending=False).head(15)
        top_html = "<h3 style='margin-top:0;'>Top 15 Municípios</h3><ol style='padding-left:16px;'>"
        for _, r in top_n.iterrows():
            top_html += f"<li>{r['mun_norm']} ({r['uf_norm']}) — <b>{int(r['count'])}</b></li>"
        top_html += "</ol>"

        total_registros = int(agg['count'].sum())
        per_uf = agg.groupby('uf_norm')['count'].sum().reset_index().sort_values('count', ascending=False)
        uf_html = "<h4>Por UF</h4><ul style='padding-left:16px;'>"
        for _, r in per_uf.iterrows():
            uf_html += f"<li>{r['uf_norm']}: {int(r['count'])}</li>"
        uf_html += "</ul>"

        summary_html = f"""
            <div style="padding:12px;font-family:Segoe UI, sans-serif;">
                <h2 style="margin:0 0 8px 0;">Quantitativo</h2>
                <p style="margin:4px 0 12px 0;"><b>Total registros:</b> {total_registros}</p>
                {top_html}
                {uf_html}
            </div>
        """

        map_html = m._repr_html_()
        return f'<div style="display:flex;gap:12px;align-items:flex-start">' \
               f'<div style="flex:1">{map_html}</div>' \
               f'<div style="width:340px;max-height:720px;overflow:auto;background:#fff;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.08)">{summary_html}</div>' \
               f'</div>'

    except Exception as e:
        return f'<div style="padding:20px;color:#900;">Erro ao gerar heatmap: {str(e)}</div>'
