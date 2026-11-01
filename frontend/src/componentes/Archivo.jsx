import { useState } from "react";
import { api, porcentaje } from "../api.js";

function descargarCsv(predicciones) {
  const escapar = (v) => `"${String(v).replaceAll('"', '""')}"`;
  const filas = [
    ["texto", "sentimiento", "confianza"],
    ...predicciones.map((p) => [p.texto, p.sentimiento, p.confianza]),
  ];
  const blob = new Blob(["﻿" + filas.map((f) => f.map(escapar).join(",")).join("\n")], {
    type: "text/csv;charset=utf-8",
  });
  const enlace = document.createElement("a");
  enlace.href = URL.createObjectURL(blob);
  enlace.download = "resenas_clasificadas.csv";
  enlace.click();
  URL.revokeObjectURL(enlace.href);
}

export default function Archivo() {
  const [archivo, setArchivo] = useState(null);
  const [resultado, setResultado] = useState(null);
  const [error, setError] = useState("");
  const [cargando, setCargando] = useState(false);

  async function analizar(e) {
    e.preventDefault();
    if (!archivo) return;
    setCargando(true);
    setError("");
    try {
      setResultado(await api.predecirArchivo(archivo));
    } catch (err) {
      setError(err.message);
      setResultado(null);
    } finally {
      setCargando(false);
    }
  }

  const positivas = resultado?.resumen.positivo ?? 0;
  const negativas = resultado?.resumen.negativo ?? 0;

  return (
    <section className="archivo">
      <form className="entrada" onSubmit={analizar}>
        <label htmlFor="archivo" className="entrada__titulo">
          Sube varias reseñas a la vez
        </label>
        <p className="paso__ayuda">
          Un .txt con una reseña por línea, o un .csv con una columna llamada «texto». Hasta 500
          reseñas y 1 MB.
        </p>
        <div className="entrada__pie">
          <input
            id="archivo"
            type="file"
            accept=".txt,.csv"
            onChange={(e) => {
              setArchivo(e.target.files[0] ?? null);
              setResultado(null);
            }}
          />
          <button className="boton" type="submit" disabled={!archivo || cargando}>
            {cargando ? "Analizando…" : "Analizar archivo"}
          </button>
        </div>
        {error && (
          <p className="aviso aviso--error" role="alert">
            {error}
          </p>
        )}
      </form>

      {resultado && (
        <div className="lote">
          <div className="lote__resumen">
            <p>
              <strong>{resultado.total}</strong> reseñas: {positivas} positivas y {negativas} negativas.
            </p>
            <div className="proporcion" aria-hidden="true">
              <span className="proporcion__neg" style={{ flexGrow: negativas }} />
              <span className="proporcion__pos" style={{ flexGrow: positivas }} />
            </div>
            <button className="boton boton--secundario" onClick={() => descargarCsv(resultado.predicciones)}>
              Descargar resultados (.csv)
            </button>
          </div>

          <div className="tabla-envoltura">
            <table className="tabla">
              <thead>
                <tr>
                  <th scope="col">Reseña</th>
                  <th scope="col">Sentimiento</th>
                  <th scope="col" className="num">Confianza</th>
                </tr>
              </thead>
              <tbody>
                {resultado.predicciones.map((p, i) => (
                  <tr key={i}>
                    <td className="tabla__texto">{p.texto}</td>
                    <td>
                      <span className={`etiqueta etiqueta--${p.sentimiento === "positivo" ? "pos" : "neg"}`}>
                        {p.sentimiento}
                      </span>
                    </td>
                    <td className="num">{porcentaje(p.confianza)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}
