from fasthtml.common import *
import json
from datetime import datetime, date

# Carregar dados JSON       
def load_json_data():
    with open('uf_options.json', 'r', encoding='utf-8') as f:
        uf_options = json.load(f)
    
    with open('municipios_por_uf.json', 'r', encoding='utf-8') as f:
        municipios_por_uf = json.load(f)
    
    with open('condicoes_metereologicas_options.json', 'r', encoding='utf-8') as f:
        condicoes_options = json.load(f)
    
    return uf_options, municipios_por_uf, condicoes_options

uf_options, municipios_por_uf, condicoes_options = load_json_data()

def prediction_form():
    return Div(
        Div(
            H5("Formulário de Predição de Acidentes", cls="mb-0"),
            cls="card-header"
        ),
        Div(
            Form(
                Div(
                    Div(
                        Label("Data:", for_="data_input", cls="form-label"),
                        Input(
                            type="date",
                            id="data_input",
                            name="data",
                            cls="form-control",
                            value=date.today().isoformat(),
                            required=True
                        ),
                        cls="mb-3"
                    ),
                    Div(
                        Label("Horário:", for_="horario_input", cls="form-label"),
                        Input(
                            type="time",
                            id="horario_input",
                            name="horario",
                            cls="form-control",
                            value="12:00",
                            required=True
                        ),
                        cls="mb-3"
                    ),
                    cls="col-md-6"
                ),
                Div(
                    Div(
                        Label("UF (Estado):", for_="uf_select", cls="form-label"),
                        Select(
                            Option("Selecione um estado...", value="", selected=True),
                            *[Option(uf, value=uf) for uf in uf_options],
                            id="uf_select",
                            name="uf",
                            cls="form-select",
                            required=True,
                            onchange="updateMunicipios()"
                        ),
                        cls="mb-3"
                    ),
                    Div(
                        Label("Município:", for_="municipio_select", cls="form-label"),
                        Select(
                            Option("Primeiro selecione um estado", value="", selected=True),
                            id="municipio_select",
                            name="municipio",
                            cls="form-select",
                            required=True,
                            disabled=True
                        ),
                        cls="mb-3"
                    ),
                    cls="col-md-6"
                ),
                Div(
                    Label("Condição Meteorológica:", for_="condicao_select", cls="form-label"),
                    Select(
                        Option("Selecione uma condição...", value="", selected=True),
                        *[Option(condicao, value=condicao) for condicao in condicoes_options],
                        id="condicao_select",
                        name="condicao_meteorologica",
                        cls="form-select",
                        required=True
                    ),
                    cls="mb-4 col-12"
                ),
                Div(
                    Button(
                        "🔮 Fazer Predição",
                        type="submit",
                        cls="btn btn-primary btn-lg w-100"
                    ),
                    cls="col-12"
                ),
                cls="row",
                method="POST",
                action="/predict"
            ),
            cls="card-body"
        ),
        cls="card"
    )

def prediction_result(formatted_result=None):
    if formatted_result is not None:
        # Determinar classe CSS baseada no nível de risco
        result_class = "prediction-result"
        if formatted_result['level'] == 'success':
            result_class += " bg-success"
        elif formatted_result['level'] == 'warning':
            result_class += " bg-warning"
        elif formatted_result['level'] == 'danger':
            result_class += " bg-danger"
        
        return Div(
            H4("Resultado da Predição"),
            Div(formatted_result['percentage'], cls="result-number"),
            P(formatted_result['description']),
            Div(
                Strong(f"Nível de Risco: {formatted_result['confidence']}"),
                style="margin-top: 10px; font-size: 1.1rem;"
            ),
            cls=result_class
        )
    return ""

def prediction_page_content(formatted_result=None):
    return Div(
        # Formulário de predição
        Div(
            Div(
                prediction_form(),
                cls="col-md-8"
            ),
            Div(
                Div(
                    Div(
                        H5("Informações sobre a Predição", cls="mb-0"),
                        cls="card-header"
                    ),
                    Div(
                        H6("Como funciona?", cls="text-primary mb-3"),
                        P("Nosso modelo de machine learning analisa diversos fatores para prever a probabilidade de acidentes:"),
                        Ul(
                            Li("📅 Data e horário"),
                            Li("📍 Localização (UF e município)"),
                            Li("🌤️ Condições meteorológicas"),
                            Li("📊 Dados históricos de acidentes"),
                            Li("🎯 Padrões sazonais e temporais")
                        ),
                        Hr(),
                        H6("Precisão do Modelo", cls="text-success mb-3"),
                        P("Nosso modelo possui uma precisão de 89% baseada em dados históricos do DATATRAN."),
                        prediction_result(formatted_result),
                        cls="card-body"
                    ),
                    cls="card"
                ),
                cls="col-md-4"
            ),
            cls="row"
        ),
        
        # Script para atualizar municípios dinamicamente
        Script(f"""
            const municipiosPorUf = {json.dumps(municipios_por_uf)};
            
            function updateMunicipios() {{
                const ufSelect = document.getElementById('uf_select');
                const municipioSelect = document.getElementById('municipio_select');
                const selectedUf = ufSelect.value;
                
                // Limpar opções existentes
                municipioSelect.innerHTML = '';
                
                if (selectedUf && municipiosPorUf[selectedUf]) {{
                    // Habilitar select de município
                    municipioSelect.disabled = false;
                    
                    // Adicionar opção padrão
                    const defaultOption = document.createElement('option');
                    defaultOption.value = '';
                    defaultOption.textContent = 'Selecione um município...';
                    defaultOption.selected = true;
                    municipioSelect.appendChild(defaultOption);
                    
                    // Adicionar municípios do estado selecionado
                    municipiosPorUf[selectedUf].forEach(municipio => {{
                        const option = document.createElement('option');
                        option.value = municipio;
                        option.textContent = municipio;
                        municipioSelect.appendChild(option);
                    }});
                }} else {{
                    // Desabilitar select de município
                    municipioSelect.disabled = true;
                    const defaultOption = document.createElement('option');
                    defaultOption.value = '';
                    defaultOption.textContent = 'Primeiro selecione um estado';
                    defaultOption.selected = true;
                    municipioSelect.appendChild(defaultOption);
                }}
            }}
        """),
        cls="content-area"
    )
