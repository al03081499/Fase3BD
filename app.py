from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Rendimiento estudiantil", page_icon="📊", layout="wide")

RESULTADOS = Path(__file__).parent / "resultados"
AZUL, ROJO, GRIS = "#2F6DB5", "#C9483B", "#8A8F98"


@st.cache_data
def cargar(nombre):
    ruta = RESULTADOS / f"{nombre}.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


def falta(nombre):
    st.warning(f"No se encontró `resultados/{nombre}.csv`. Ejecuten el notebook del backend y copien la carpeta de resultados.")


def p_texto(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def tarjeta_decision(h):
    resumen = cargar("resumen_hipotesis")
    if resumen is None:
        return
    fila = resumen[resumen["hipotesis"] == h]
    if fila.empty:
        return
    fila = fila.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prueba", fila["prueba"])
    c2.metric("Estadístico", fila["estadistico"])
    c3.metric("Valor p", p_texto(fila["p_valor"]))
    c4.metric("Tamaño del efecto", fila["efecto"])
    if fila["decision"].startswith("Se acepta"):
        st.success(f"**Decisión: {fila['decision']}**")
    else:
        st.error(f"**Decisión: {fila['decision']}**")


def estilo(fig, alto=420):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="",
                      font=dict(size=13), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


st.sidebar.title("📊 Rendimiento estudiantil")
st.sidebar.caption("Proyecto integrador de tecnologías de Big Data\n\nPaloma Garza Muñoz · Emilio Duarte Morin")
pagina = st.sidebar.radio("Sección", [
    "Resumen",
    "H1 · Horas de estudio",
    "H2 · Asistencia",
    "H3 · Apoyo parental",
    "H4 · Trabajo de medio tiempo",
    "H5 · Internet y escuela",
    "Correlaciones",
    "Modelo de riesgo",
])
st.sidebar.divider()
st.sidebar.caption("Backend: PySpark · Frontend: Streamlit\n\nDatos: Student Performance Dataset (Kaggle)")


if pagina == "Resumen":
    st.title("Factores asociados al bajo rendimiento académico")
    st.caption("Un estudiante está **en riesgo** si su GPA está en el 25 % más bajo del conjunto.")
    kpis = cargar("kpis")
    if kpis is None:
        falta("kpis")
    else:
        k = kpis.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Estudiantes analizados", f"{int(k['registros_limpios']):,}")
        c2.metric("GPA medio", f"{k['gpa_medio']:.2f}")
        c3.metric("% en riesgo", f"{k['pct_riesgo']:.1f} %")
        c4.metric("Asistencia media", f"{k['asistencia_media']:.1f} %")
        c5.metric("Horas de estudio medias", f"{k['horas_estudio_media']:.1f}")
        st.caption(f"Umbral de riesgo: GPA ≤ {k['umbral_riesgo_gpa']:.2f} · Registros originales: {int(k['registros_originales']):,} · Duplicados: {int(k['duplicados']):,}")

    st.subheader("Resultado de las hipótesis")
    resumen = cargar("resumen_hipotesis")
    if resumen is None:
        falta("resumen_hipotesis")
    else:
        tabla = resumen.copy()
        tabla["p_valor"] = tabla["p_valor"].map(p_texto)
        tabla = tabla.rename(columns={"hipotesis": "Hipótesis", "descripcion": "Descripción", "prueba": "Prueba",
                                      "estadistico": "Estadístico", "p_valor": "Valor p", "efecto": "Efecto", "decision": "Decisión"})
        tabla = tabla[["Hipótesis", "Descripción", "Decisión", "Prueba", "Estadístico", "Valor p", "Efecto"]]
        st.dataframe(tabla.style.map(lambda v: f"color: {AZUL if str(v).startswith('Se acepta') else ROJO}; font-weight: 600",
                                     subset=["Decisión"]), hide_index=True, width="stretch")

    calidad = cargar("calidad_datos")
    if calidad is not None:
        with st.expander("Calidad de datos: valores nulos por variable"):
            st.dataframe(calidad, hide_index=True, width="stretch")


elif pagina == "H1 · Horas de estudio":
    st.title("H1 · A más horas de estudio, mayor GPA")
    tarjeta_decision("H1")
    tabla, muestra = cargar("h1_tabla"), cargar("h1_muestra")
    if tabla is None or muestra is None:
        falta("h1_tabla")
    else:
        c1, c2 = st.columns(2)
        fig = px.scatter(muestra, x="horas_estudio", y="gpa", opacity=0.3, color_discrete_sequence=[AZUL],
                         labels={"horas_estudio": "Horas de estudio", "gpa": "GPA"},
                         title="Horas de estudio vs GPA (muestra de 5,000)")
        m, b = np.polyfit(muestra["horas_estudio"], muestra["gpa"], 1)
        xs = np.array([muestra["horas_estudio"].min(), muestra["horas_estudio"].max()])
        fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", line=dict(color=ROJO, width=3), name="Tendencia"))
        c1.plotly_chart(estilo(fig), width="stretch")
        fig = go.Figure([
            go.Scatter(x=tabla["rango_horas"], y=tabla["gpa_medio"] + tabla["gpa_desv"], line=dict(width=0), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=tabla["rango_horas"], y=tabla["gpa_medio"] - tabla["gpa_desv"], line=dict(width=0), fill="tonexty",
                       fillcolor="rgba(47,109,181,0.15)", name="± 1 desviación", hoverinfo="skip"),
            go.Scatter(x=tabla["rango_horas"], y=tabla["gpa_medio"], mode="lines+markers", line=dict(color=AZUL, width=3), name="GPA medio"),
        ])
        fig.update_layout(title="GPA medio por rango de horas (todos los registros)", xaxis_title="Horas de estudio", yaxis_title="GPA medio")
        c2.plotly_chart(estilo(fig), width="stretch")
        st.dataframe(tabla.rename(columns={"rango_horas": "Rango de horas", "n": "Estudiantes", "gpa_medio": "GPA medio",
                                           "gpa_desv": "Desv. estándar", "pct_riesgo": "% en riesgo"}),
                     hide_index=True, width="stretch")


elif pagina == "H2 · Asistencia":
    st.title("H2 · Asistencia menor al 80 % duplica el riesgo")
    tarjeta_decision("H2")
    tabla = cargar("h2_tabla")
    if tabla is None:
        falta("h2_tabla")
    else:
        tabla["grupo"] = ["Asistencia < 80 %" if r in ("<70 %", "70–80 %") else "Asistencia ≥ 80 %" for r in tabla["rango_asistencia"]]
        fig = px.bar(tabla, x="rango_asistencia", y="pct_riesgo", color="grupo", text="pct_riesgo",
                     color_discrete_map={"Asistencia < 80 %": ROJO, "Asistencia ≥ 80 %": AZUL},
                     labels={"rango_asistencia": "Tasa de asistencia", "pct_riesgo": "% en riesgo"},
                     title="Porcentaje de estudiantes en riesgo por rango de asistencia")
        fig.update_traces(texttemplate="%{text:.1f} %", textposition="outside")
        st.plotly_chart(estilo(fig), width="stretch")
        st.dataframe(tabla.drop(columns="grupo").rename(columns={"rango_asistencia": "Asistencia", "n": "Estudiantes",
                     "en_riesgo": "En riesgo", "no_riesgo": "Sin riesgo", "gpa_medio": "GPA medio", "pct_riesgo": "% en riesgo"}),
                     hide_index=True, width="stretch")


elif pagina == "H3 · Apoyo parental":
    st.title("H3 · Mayor apoyo parental, mejores puntajes")
    tarjeta_decision("H3")
    tabla, anova = cargar("h3_tabla"), cargar("h3_anova")
    if tabla is None:
        falta("h3_tabla")
    else:
        materias = [m for m in ["matematicas", "lectura", "ciencias"] if m in tabla.columns]
        nombres = {"matematicas": "Matemáticas", "lectura": "Lectura", "ciencias": "Ciencias"}
        largo = tabla.melt(id_vars="apoyo_parental", value_vars=materias, var_name="Materia", value_name="Puntaje medio")
        largo["Materia"] = largo["Materia"].map(nombres)
        largo["apoyo_parental"] = largo["apoyo_parental"].astype(str)
        fig = px.bar(largo, x="Materia", y="Puntaje medio", color="apoyo_parental", barmode="group", text="Puntaje medio",
                     color_discrete_sequence=["#A9C6E8", "#5B93D1", "#1F4E8C"],
                     title="Puntaje medio por materia y nivel de apoyo parental")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_yaxes(range=[largo["Puntaje medio"].min() * 0.9, largo["Puntaje medio"].max() * 1.05])
        fig.update_layout(legend_title_text="Apoyo parental")
        st.plotly_chart(estilo(fig), width="stretch")
        c1, c2 = st.columns(2)
        c1.markdown("**Medias por nivel de apoyo**")
        c1.dataframe(tabla.rename(columns={"apoyo_parental": "Apoyo parental", "n": "Estudiantes", **nombres, "gpa_medio": "GPA medio"}),
                     hide_index=True, width="stretch")
        if anova is not None:
            a = anova.copy()
            a["materia"] = a["materia"].map(nombres)
            a["p_valor"] = a["p_valor"].map(p_texto)
            c2.markdown("**ANOVA por materia**")
            c2.dataframe(a.rename(columns={"materia": "Materia", "p_valor": "Valor p", "eta2": "η²", "medias_crecientes": "Medias crecientes"}),
                         hide_index=True, width="stretch")


elif pagina == "H4 · Trabajo de medio tiempo":
    st.title("H4 · Quienes trabajan tienen menor GPA y estudian menos")
    tarjeta_decision("H4")
    tabla, cuartiles = cargar("h4_tabla"), cargar("h4_cuartiles")
    if tabla is None or cuartiles is None:
        falta("h4_tabla")
    else:
        c1, c2 = st.columns(2)
        for col, variable, titulo in [(c1, "gpa", "GPA"), (c2, "horas_estudio", "Horas de estudio")]:
            q = cuartiles[cuartiles["variable"] == variable]
            fig = go.Figure()
            for _, f in q.iterrows():
                fig.add_trace(go.Box(name=f["grupo"], q1=[f["q1"]], median=[f["mediana"]], q3=[f["q3"]],
                                     lowerfence=[f["min"]], upperfence=[f["max"]],
                                     marker_color=ROJO if f["grupo"] == "Sí trabaja" else AZUL, showlegend=False))
            fig.update_layout(title=f"{titulo} según trabajo de medio tiempo", yaxis_title=titulo)
            col.plotly_chart(estilo(fig), width="stretch")
        t = tabla.copy()
        t["variable"] = t["variable"].map({"gpa": "GPA", "horas_estudio": "Horas de estudio"})
        t["p_valor"] = t["p_valor"].map(p_texto)
        st.dataframe(t.rename(columns={"variable": "Variable", "media_trabaja": "Media (trabaja)", "media_no_trabaja": "Media (no trabaja)",
                                       "diferencia": "Diferencia", "p_valor": "Valor p", "d_cohen": "d de Cohen",
                                       "n_trabaja": "n (trabaja)", "n_no_trabaja": "n (no trabaja)"}),
                     hide_index=True, width="stretch")


elif pagina == "H5 · Internet y escuela":
    st.title("H5 · Sin Internet aumenta el riesgo")
    tarjeta_decision("H5")
    tabla = cargar("h5_tabla")
    if tabla is None:
        falta("h5_tabla")
    else:
        tabla["tipo_escuela"] = tabla["tipo_escuela"].astype(str)
        fig = px.bar(tabla, x="tipo_escuela", y="pct_riesgo", color="internet", barmode="group", text="pct_riesgo",
                     color_discrete_map={"Sin Internet": ROJO, "Con Internet": AZUL},
                     labels={"tipo_escuela": "Tipo de escuela", "pct_riesgo": "% en riesgo"},
                     title="% de estudiantes en riesgo por acceso a Internet y tipo de escuela")
        fig.update_traces(texttemplate="%{text:.1f} %", textposition="outside")
        st.plotly_chart(estilo(fig), width="stretch")
        st.dataframe(tabla.rename(columns={"tipo_escuela": "Tipo de escuela", "internet": "Internet", "n": "Estudiantes",
                                           "en_riesgo": "En riesgo", "pct_riesgo": "% en riesgo"}),
                     hide_index=True, width="stretch")


elif pagina == "Correlaciones":
    st.title("Correlaciones entre variables")
    corr = cargar("correlaciones")
    if corr is None:
        falta("correlaciones")
    else:
        m = corr.set_index("variable")
        fig = px.imshow(m, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                        title="Matriz de correlación de Pearson (todos los registros)")
        st.plotly_chart(estilo(fig, 560), width="stretch")
        if "gpa" in m.columns:
            orden = m["gpa"].drop("gpa").sort_values()
            fig = px.bar(x=orden.values, y=orden.index, orientation="h", text=orden.values,
                         color=orden.values > 0, color_discrete_map={True: AZUL, False: ROJO},
                         labels={"x": "Correlación con GPA", "y": ""}, title="Correlación de cada variable con el GPA")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(estilo(fig, 380), width="stretch")


elif pagina == "Modelo de riesgo":
    st.title("Modelo de clasificación de riesgo (bosque aleatorio)")
    metricas, importancia, confusion = cargar("modelo_metricas"), cargar("modelo_importancia"), cargar("modelo_confusion")
    if metricas is None:
        falta("modelo_metricas")
    else:
        cols = st.columns(len(metricas))
        for col, (_, f) in zip(cols, metricas.iterrows()):
            col.metric(f["metrica"], f"{f['valor']:.3f}")
        c1, c2 = st.columns(2)
        if importancia is not None:
            imp = importancia.sort_values("importancia").copy()
            imp["variable"] = imp["variable"].replace({"horas_estudio": "Horas de estudio", "asistencia": "Asistencia",
                                                       "apoyo_parental": "Apoyo parental", "trabajo": "Trabajo de medio tiempo",
                                                       "internet": "Acceso a Internet", "tipo_escuela": "Tipo de escuela"})
            fig = px.bar(imp, x="importancia", y="variable", orientation="h", color_discrete_sequence=[AZUL],
                         labels={"importancia": "Importancia", "variable": ""}, title="Importancia de variables")
            c1.plotly_chart(estilo(fig), width="stretch")
        if confusion is not None:
            orden = ["En riesgo", "Sin riesgo"]
            m = confusion.pivot(index="real", columns="predicho", values="n").reindex(index=orden, columns=orden).fillna(0)
            fig = px.imshow(m, text_auto=",d", color_continuous_scale="Blues", aspect="auto",
                            labels={"x": "Predicho", "y": "Real", "color": "Estudiantes"}, title="Matriz de confusión")
            c2.plotly_chart(estilo(fig), width="stretch")
        st.caption("El modelo usa horas de estudio, asistencia, acceso a Internet, trabajo de medio tiempo, apoyo parental y tipo de escuela. "
                   "Se excluyen GPA y puntajes porque definen el riesgo.")
