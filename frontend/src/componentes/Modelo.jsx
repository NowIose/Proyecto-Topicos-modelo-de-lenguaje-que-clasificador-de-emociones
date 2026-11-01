import { useEffect, useState } from "react";
import { api, porcentaje } from "../api.js";

const METODOS = { stemming: "Stemming (Snowball)", lematizacion: "Lematización (spaCy)", ninguno: "Sin reducción" };
const ORDEN_MODELOS = ["logistic_regression", "multinomial_nb", "baseline"];

// Barra horizontal simple: el ancho codifica el valor en [0, 1].
function Barra({ valor, destacada }) {
  return (
    <span className="barra" aria-hidden="true">
      <span className={`barra__relleno ${destacada ? "barra__relleno--destacada" : ""}`} style={{ width: `${valor * 100}%` }} />
    </span>
  );
}

function Comparacion({ m }) {
  return (
    <section className="bloque">
      <h2 className="seccion__titulo">Comparación de modelos en el conjunto de prueba</h2>
      <p className="bloque__ayuda">
        {m.dataset.test.toLocaleString("es")} reseñas que el modelo nunca vio al entrenar ni al
        ajustar hiperparámetros. La línea base siempre predice la clase más frecuente.
      </p>
      <div className="tabla-envoltura">
        <table className="tabla">
          <thead>
            <tr>
              <th scope="col">Modelo</th>
              <th scope="col">Hiperparámetro</th>
              <th scope="col" className="num">Accuracy</th>
              <th scope="col" className="num">Precisión</th>
              <th scope="col" className="num">Recall</th>
              <th scope="col">F1 macro</th>
            </tr>
          </thead>
          <tbody>
            {ORDEN_MODELOS.map((clave) => {
              const modelo = m.modelos[clave];
              const p = modelo.prueba;
              return (
                <tr key={clave} className={modelo.en_produccion ? "fila--destacada" : ""}>
                  <th scope="row">
                    {modelo.nombre}
                    {modelo.en_produccion && <span className="insignia">en producción</span>}
                  </th>
                  <td>
                    {Object.entries(modelo.hiperparametros).map(([k, v]) => `${k} = ${v}`).join(", ")}
                  </td>
                  <td className="num">{porcentaje(p.accuracy)}</td>
                  <td className="num">{porcentaje(p.precision_macro)}</td>
                  <td className="num">{porcentaje(p.recall_macro)}</td>
                  <td>
<div className="celda-barra">
                    <Barra valor={p.f1_macro} destacada={modelo.en_produccion} />
                    <span className="num">{p.f1_macro.toFixed(3)}</span>
                  </div>
</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function MatrizConfusion({ m }) {
  const [clave, setClave] = useState(m.modelo_en_produccion);
  const { etiquetas, valores } = m.modelos[clave].prueba.matriz_confusion;
  const total = valores.flat().reduce((a, b) => a + b, 0);
  const maximo = Math.max(...valores.flat());

  return (
    <section className="bloque">
      <h2 className="seccion__titulo">Matriz de confusión</h2>
      <div className="selector" role="group" aria-label="Modelo">
        {ORDEN_MODELOS.filter((k) => k !== "baseline").map((k) => (
          <button key={k} className="selector__opcion" aria-pressed={clave === k} onClick={() => setClave(k)}>
            {m.modelos[k].nombre}
          </button>
        ))}
      </div>
      <table className="matriz">
        <thead>
          <tr>
            <td />
            {etiquetas.map((e) => (
              <th key={e} scope="col">predicho {e}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {valores.map((fila, i) => (
            <tr key={etiquetas[i]}>
              <th scope="row">real {etiquetas[i]}</th>
              {fila.map((v, j) => (
                <td
                  key={j}
                  className={i === j ? "matriz__acierto" : "matriz__error"}
                  style={{ "--peso": v / maximo }}
                >
                  <strong>{v.toLocaleString("es")}</strong>
                  <span>{porcentaje(v / total)}</span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function Morfologia({ m }) {
  const mejor = Math.max(...m.comparacion_morfologica.map((f) => f.f1_macro));
  return (
    <section className="bloque">
      <h2 className="seccion__titulo">Stemming contra lematización</h2>
      <p className="bloque__ayuda">
        Regresión Logística, validación cruzada de {m.validacion.cv_folds} pliegues sobre{" "}
        {m.comparacion_morfologica[0].muestra.toLocaleString("es")} reseñas de entrenamiento. En
        producción: {METODOS[m.nlp.metodo_morfologico].toLowerCase()}.
      </p>
      <div className="tabla-envoltura">
        <table className="tabla">
          <thead>
            <tr>
              <th scope="col">Reducción</th>
              <th scope="col">F1 macro (CV)</th>
              <th scope="col" className="num">Vocabulario</th>
              <th scope="col" className="num">Tiempo</th>
            </tr>
          </thead>
          <tbody>
            {m.comparacion_morfologica.map((f) => (
              <tr key={f.metodo} className={f.metodo === m.nlp.metodo_morfologico ? "fila--destacada" : ""}>
                <th scope="row">{METODOS[f.metodo]}</th>
                <td>
<div className="celda-barra">
                  <Barra valor={f.f1_macro} destacada={f.f1_macro === mejor} />
                  <span className="num">{f.f1_macro.toFixed(4)}</span>
                </div>
</td>
                <td className="num">{f.vocabulario_distinto.toLocaleString("es")} palabras</td>
                <td className="num">{f.segundos_preprocesamiento} s</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Ngramas({ m }) {
  const modelos = ["logistic_regression", "multinomial_nb", "baseline"];
  const valor = (modelo, rep) => m.experimentos.find((e) => e.modelo === modelo && e.representacion === rep);
  return (
    <section className="bloque">
      <h2 className="seccion__titulo">Unigramas contra unigramas + bigramas</h2>
      <p className="bloque__ayuda">
        F1 macro de validación cruzada sobre las {m.dataset.train.toLocaleString("es")} reseñas de
        entrenamiento. Los bigramas capturan frases como «no funciona» o «muy bueno».
      </p>
      <div className="tabla-envoltura">
        <table className="tabla">
          <thead>
            <tr>
              <th scope="col">Modelo</th>
              <th scope="col">Unigramas</th>
              <th scope="col">Unigramas + bigramas</th>
            </tr>
          </thead>
          <tbody>
            {modelos.map((k) => {
              const uni = valor(k, "unigramas");
              const bi = valor(k, "unigramas + bigramas");
              return (
                <tr key={k}>
                  <th scope="row">{m.modelos[k].nombre}</th>
                  <td>
<div className="celda-barra">
                    <Barra valor={uni.f1_macro} />
                    <span className="num">{uni.f1_macro.toFixed(4)}</span>
                  </div>
</td>
                  <td>
<div className="celda-barra">
                    <Barra valor={bi.f1_macro} destacada={bi.f1_macro > uni.f1_macro} />
                    <span className="num">{bi.f1_macro.toFixed(4)}</span>
                  </div>
</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Ajuste({ m }) {
  return (
    <section className="bloque">
      <h2 className="seccion__titulo">Ajuste de hiperparámetros (GridSearchCV)</h2>
      <div className="dos-columnas">
        {Object.entries(m.ajuste_hiperparametros).map(([clave, a]) => (
          <div key={clave}>
            <h3 className="subtitulo">{m.modelos[clave].nombre}</h3>
            <ul className="grilla">
              {a.grilla.map((g) => (
                <li key={g.valor} className={g.valor === a.mejor_valor ? "grilla__mejor" : ""}>
                  <span>
                    {a.parametro} = {g.valor}
                  </span>
                  <span className="num">{g.f1_macro.toFixed(4)}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}

function Terminos({ m }) {
  return (
    <section className="bloque">
      <h2 className="seccion__titulo">Términos que más pesan en el modelo</h2>
      <div className="dos-columnas">
        {[
          ["positivo", "Hacia positiva"],
          ["negativo", "Hacia negativa"],
        ].map(([clase, titulo]) => (
          <div key={clase}>
            <h3 className="subtitulo">{titulo}</h3>
            <ul className="fichas">
              {m.terminos_influyentes[clase].map((t) => (
                <li key={t.termino} className={`ficha ficha--${clase === "positivo" ? "pos" : "neg"}`}>
                  {t.termino}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}

function Errores({ m }) {
  const e = m.analisis_errores;
  const Lista = ({ items }) => (
    <ul className="errores">
      {items.map((x, i) => (
        <li key={i}>
          <p className="errores__texto">{x.texto}</p>
          <p className="errores__meta">
            Era {x.real}, el modelo dijo {x.predicho} con {porcentaje(x.confianza)}
          </p>
        </li>
      ))}
    </ul>
  );
  return (
    <section className="bloque">
      <h2 className="seccion__titulo">Dónde se equivoca</h2>
      <p className="bloque__ayuda">
        {e.total_errores.toLocaleString("es")} errores en prueba: {e.falsos_positivos.toLocaleString("es")} falsos
        positivos y {e.falsos_negativos.toLocaleString("es")} falsos negativos. Abajo, los errores en los que el
        modelo estaba más seguro.
      </p>
      <div className="dos-columnas">
        <div>
          <h3 className="subtitulo">Negativas que tomó por positivas</h3>
          <Lista items={e.ejemplos_falsos_positivos} />
        </div>
        <div>
          <h3 className="subtitulo">Positivas que tomó por negativas</h3>
          <Lista items={e.ejemplos_falsos_negativos} />
        </div>
      </div>
    </section>
  );
}

export default function Modelo() {
  const [m, setM] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.metricas().then(setM).catch((e) => setError(e.message));
  }, []);

  if (error) return <p className="aviso aviso--error">{error}</p>;
  if (!m) return <p className="vacio">Cargando métricas…</p>;

  const prod = m.modelos[m.modelo_en_produccion];
  return (
    <div className="modelo">
      <section className="resumen-modelo">
        <p className="resumen-modelo__cifra">{porcentaje(prod.prueba.f1_macro)}</p>
        <p className="resumen-modelo__texto">
          de F1 macro en prueba con <strong>{prod.nombre}</strong> sobre TF-IDF de{" "}
          {m.nlp.ngram_range[1] > 1 ? "unigramas y bigramas" : "unigramas"} ({m.nlp.vocabulario.toLocaleString("es")}{" "}
          términos, min_df = {m.nlp.min_df}, max_df = {m.nlp.max_df}). Entrenado con{" "}
          {m.dataset.train.toLocaleString("es")} reseñas balanceadas el{" "}
          {new Date(m.generado).toLocaleString("es")}.
        </p>
      </section>
      <Comparacion m={m} />
      <MatrizConfusion m={m} />
      <Morfologia m={m} />
      <Ngramas m={m} />
      <Ajuste m={m} />
      <Terminos m={m} />
      <Errores m={m} />
    </div>
  );
}
