import { useEffect, useState } from "react";
import { api, porcentaje } from "../api.js";

function formatearFecha(iso) {
  if (!iso) return "-";
  try {
    const f = new Date(iso);
    return f.toLocaleString("es-ES", {
      dateStyle: "short",
      timeStyle: "short",
    });
  } catch {
    return iso;
  }
}

export default function Historial() {
  const [items, setItems] = useState([]);
  const [cargando, setCargando] = useState(true);
  const [error, setError] = useState("");

  async function cargarHistorial() {
    setCargando(true);
    setError("");
    try {
      const res = await api.historial(100);
      setItems(Array.isArray(res) ? res : (res.historial ?? []));
    } catch (err) {
      setError(err.message);
    } finally {
      setCargando(false);
    }
  }

  async function vaciar() {
    if (!window.confirm("¿Estás seguro de que deseas vaciar todo el historial de predicciones?")) {
      return;
    }
    setCargando(true);
    try {
      await api.limpiarHistorial();
      setItems([]);
    } catch (err) {
      setError(err.message);
    } finally {
      setCargando(false);
    }
  }

  useEffect(() => {
    cargarHistorial();
  }, []);

  return (
    <section className="historial">
      <div className="historial__cabecera">
        <div>
          <h2 className="historial__titulo">Historial de Predicciones</h2>
          <p className="bloque__ayuda">
            Registro persistente almacenado localmente en SQLite (<code>data/historial.db</code>).
            Incluye las calificaciones de acierto dadas por los usuarios.
          </p>
        </div>
        <div className="historial__acciones">
          <button
            type="button"
            className="boton boton--secundario"
            onClick={cargarHistorial}
            disabled={cargando}
          >
            {cargando ? "Cargando…" : "🔄 Actualizar"}
          </button>
          {items.length > 0 && (
            <button
              type="button"
              className="boton boton--secundario boton--peligro"
              onClick={vaciar}
              disabled={cargando}
            >
              🗑️ Vaciar
            </button>
          )}
        </div>
      </div>

      {error && (
        <p className="aviso aviso--error" role="alert">
          {error}
        </p>
      )}

      {items.length === 0 && !cargando && (
        <div className="historial__vacio">
          <p>No hay predicciones registradas aún.</p>
          <p className="paso__ayuda">
            Analiza una reseña individual o carga un archivo para ver los registros aquí.
          </p>
        </div>
      )}

      {items.length > 0 && (
        <div className="tabla-envoltura">
          <table className="tabla">
            <thead>
              <tr>
                <th scope="col" style={{ width: "120px" }}>Fecha</th>
                <th scope="col">Reseña</th>
                <th scope="col">Sentimiento</th>
                <th scope="col" className="num">Confianza</th>
                <th scope="col" style={{ textAlign: "center", width: "120px" }}>Feedback</th>
              </tr>
            </thead>
            <tbody>
              {items.map((it) => (
                <tr key={it.id}>
                  <td className="tabla__fecha">{formatearFecha(it.fecha)}</td>
                  <td className="tabla__texto">{it.texto}</td>
                  <td>
                    <span className={`insignia insignia--${it.sentimiento === "positivo" ? "pos" : "neg"}`}>
                      {it.sentimiento}
                    </span>
                  </td>
                  <td className="num">{porcentaje(it.confianza)}</td>
                  <td style={{ textAlign: "center" }}>
                    {it.feedback === 1 && (
                      <span className="badge-feedback badge-feedback--ok" title="El usuario confirmó que fue acertado">
                        👍 Acertó
                      </span>
                    )}
                    {it.feedback === 0 && (
                      <span className="badge-feedback badge-feedback--err" title="El usuario indicó que falló">
                        👎 Falló
                      </span>
                    )}
                    {it.feedback === null && (
                      <span className="badge-feedback badge-feedback--none" title="Sin calificar">
                        —
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
