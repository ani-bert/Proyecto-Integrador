import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="Sistema Predictivo de Ganancia",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# COLORES
# =====================================================

C1 = "#cf86b4"
C2 = "#99bdc6"
C3 = "#60ca9a"

# =====================================================
# CARGAR MODELO
# =====================================================

try:
    modelo = joblib.load("modelo_clasificacion_ganancia.pkl")
    preprocesador = joblib.load("preprocesador_clasificacion.pkl")
    # Regresión
    modelo_reg = joblib.load(
        "modelo_regresion_ganancia.pkl"
    )
    preprocesador_reg = joblib.load(
        "preprocesador_regresion.pkl"
    )

except Exception as e:
    st.error(f"Error cargando modelo: {e}")
    st.stop()

# =====================================================
# DATASET
# =====================================================

sheet_url = (
    "https://docs.google.com/spreadsheets/d/"
    "1r0w8oxZ8V2lx1o79jXFOxHwAjq_Sn-jXXj-d1nSLhCo/"
    "export?format=csv"
)

@st.cache_data
def cargar_datos():

    df = pd.read_csv(sheet_url)

    df.columns = df.columns.str.strip()

    return df

try:

    df = cargar_datos()

except Exception as e:

    st.error(f"Error cargando dataset: {e}")
    st.stop()

# =====================================================
# LIMPIEZA BÁSICA
# =====================================================

df["ganancia_total"] = pd.to_numeric(
    df["ganancia_total"],
    errors="coerce"
)

df = df.dropna(subset=["ganancia_total"])

# =====================================================
# CREAR NIVEL DE GANANCIA
# =====================================================

p25 = df["ganancia_total"].quantile(0.25)
p75 = df["ganancia_total"].quantile(0.75)

def clasificar_ganancia(g):

    if g < p25:
        return "Baja"

    elif g <= p75:
        return "Media"

    else:
        return "Alta"

df["nivel_ganancia"] = df["ganancia_total"].apply(
    clasificar_ganancia
)

# =====================================================
# PROMEDIOS REALES
# =====================================================

ganancias_reales = (
    df.groupby("nivel_ganancia")["ganancia_total"]
    .mean()
    .to_dict()
)

# =====================================================
# TÍTULO
# =====================================================

st.title("📊 Sistema Predictivo de Ganancia")

st.markdown(
    """
    Ingresar Datos
    """
)

# =====================================================
# FORMULARIO
# =====================================================

st.subheader("📝 Datos de la Venta")

col1, col2 = st.columns(2)

with col1:

    n_productos = st.number_input(
        "📦 Número de Productos",
        min_value=1,
        value=5
    )

    precio_promedio = st.number_input(
        "💰 Precio Promedio",
        min_value=1.0,
        value=100.0
    )

    desc_maximo = st.slider(
        "🏷️ Descuento Máximo",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01
    )

with col2:

    categoria_principal = st.selectbox(
        "🛍️ Categoría",
        sorted(
            df["categoria_principal"]
            .dropna()
            .unique()
        )
    )

    channel = st.selectbox(
        "📡 Canal",
        sorted(
            df["channel"]
            .dropna()
            .unique()
        )
    )

    country = st.selectbox(
        "🌎 País",
        sorted(
            df["country"]
            .dropna()
            .unique()
        )
    )

    marca_principal = st.selectbox(
        "🏷️ Marca",
        sorted(
            df["marca_principal"]
            .dropna()
            .unique()
        )
    )

# =====================================================
# BOTÓN DE PREDICCIÓN
# =====================================================

if st.button("🔮 Predecir Ganancia"):

    nuevo = pd.DataFrame([{
        "n_productos": n_productos,
        "precio_promedio": precio_promedio,
        "desc_maximo": desc_maximo,
        "categoria_principal": categoria_principal,
        "channel": channel,
        "country": country,
        "marca_principal": marca_principal
    }])

    nuevo_proc = preprocesador.transform(nuevo)

    pred = modelo.predict(nuevo_proc)[0]

    prob = modelo.predict_proba(nuevo_proc)[0]

    nuevo_proc_reg = preprocesador_reg.transform(
    nuevo
    )

    ganancia_numerica = modelo_reg.predict(
    nuevo_proc_reg
    )[0]

 
    clases = modelo.classes_
    

    # =====================================================
    # RESULTADO DE LA PREDICCIÓN
    # =====================================================

    st.markdown("## 🃏 Resultado de la Predicción")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if pred == "Alta":
            st.success("💰 Ganancia Estimada\n\nALTA")
        elif pred == "Media":
            st.warning("💰 Ganancia Estimada\n\nMEDIA")
        else:
            st.error("💰 Ganancia Estimada\n\nBAJA")

    with col2:
        if pred == "Alta":
            st.success("📈 Potencial de Rentabilidad\n\nALTO")
        elif pred == "Media":
            st.warning("📈 Potencial de Rentabilidad\n\nMEDIO")
        else:
            st.error("📈 Potencial de Rentabilidad\n\nBAJO")

    with col3:
        if pred == "Alta":
            st.success("🎯 Oportunidad Comercial\n\nFAVORABLE")
        elif pred == "Media":
            st.warning("🎯 Oportunidad Comercial\n\nMODERADA")
        else:
            st.error("🎯 Oportunidad Comercial\n\nLIMITADA")

    with col4:
        if pred == "Alta":
            st.success("💡 Promover la venta")
        elif pred == "Media":
            st.warning("💡 Mantener estrategia actual")
        else:
            st.error("💡 Revisar precios o costos")

    # =====================================================
    # GANANCIA APROXIMADA
    # =====================================================

    st.markdown("## 💰 Ganancia Aproximada")

    ganancia_numerica = modelo_reg.predict(
    nuevo_proc_reg
    )[0]

    ganancia_numerica = float(ganancia_numerica)

    st.metric(
        "Ganancia estimada",
        f"S/ {ganancia_numerica:,.2f}"
    )

        # =====================================================
    # GRÁFICOS
    # =====================================================

    st.markdown("## 📊 Análisis Visual")

    col_graf1, col_graf2 = st.columns(2)

    # -----------------------------------
    # Distribución de probabilidades
    # -----------------------------------

    with col_graf1:



        fig, ax = plt.subplots(figsize=(6, 4))

        barras = ax.bar(
            clases,
            prob,
            color=[C1, C2, C3]
        )

        ax.set_ylabel("Probabilidad")
        ax.set_xlabel("Nivel")
        ax.set_title("Nivel de Ganancia")

        for barra in barras:

            altura = barra.get_height()

            ax.text(
                barra.get_x() + barra.get_width() / 2,
                altura,
                f"{altura:.1%}",
                ha="center"
            )

        st.pyplot(fig)

    # -----------------------------------
    # Escenarios de ganancia
    # -----------------------------------

    with col_graf2:

        

        escenarios = [
            "Pesimista",
            "Esperado",
            "Optimista"
        ]

        valores = [
            ganancia_numerica * 0.80,
            ganancia_numerica,
            ganancia_numerica * 1.20
        ]

        fig2, ax2 = plt.subplots(figsize=(6, 4))

        ax2.plot(
            escenarios,
            valores,
            marker="o",
            linewidth=3,
            color=C3
        )

        ax2.set_title("Ganancia Estimada")
        ax2.set_ylabel("S/")

        for x, y in zip(escenarios, valores):

            ax2.text(
                x,
                y,
                f"S/ {y:,.0f}",
                ha="center"
            )

        st.pyplot(fig2)

    st.caption(
        "Las probabilidades muestran la confianza del modelo y los escenarios representan posibles variaciones de la ganancia estimada."
    )

    # =====================================================
    # RECOMENDACIONES AUTOMÁTICAS
    # =====================================================

    st.markdown("## 💡 Recomendaciones Comerciales")

    if pred == "Alta":

        st.success("✅ Venta recomendada")

        st.markdown("""
### Acciones sugeridas

• Proceder con la venta.

• Mantener la estrategia comercial actual.

• Considerar promociones complementarias.

• Aprovechar la oportunidad para incrementar el ticket promedio.

• Priorizar este tipo de operaciones en futuras campañas.
        """)

    elif pred == "Media":

        st.warning("⚠️ Venta con potencial moderado")

        st.markdown("""
### Acciones sugeridas

• Evaluar condiciones comerciales antes de cerrar la venta.

• Revisar posibles descuentos aplicados.

• Analizar productos complementarios para aumentar la ganancia.

• Comparar el desempeño con ventas similares.

• Monitorear el margen esperado.
        """)

    else:

        st.error("🚨 Venta de riesgo")

        st.markdown("""
### Acciones sugeridas

• Revisar precios y costos asociados.

• Analizar si el descuento es demasiado elevado.

• Evaluar alternativas de canal de venta.

• Considerar productos con mejor margen.

• Replantear la estrategia antes de concretar la operación.
        """)

# =====================================================
# EJECUTAR
# =====================================================

# streamlit run prueba.py