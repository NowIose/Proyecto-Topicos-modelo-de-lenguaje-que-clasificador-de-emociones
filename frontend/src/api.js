// Cliente de la API. En desarrollo Vite reenvía /api a FastAPI (vite.config.js);
// en producción la misma FastAPI sirve este frontend, así que la ruta es relativa.

async function pedir(ruta, opciones = {}) {
  let respuesta;
  try {
    respuesta = await fetch(ruta, opciones);
  } catch {
    throw new Error("No hay conexión con la API. Levanta el backend con: uvicorn api.main:app");
  }
  const cuerpo = await respuesta.json().catch(() => ({}));
  if (!respuesta.ok) {
    const detalle = Array.isArray(cuerpo.detalle) ? cuerpo.detalle.join(" ") : "";
    throw new Error(detalle || cuerpo.error || `Error ${respuesta.status}`);
  }
  return cuerpo;
}

const json = (datos) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(datos),
});

export const api = {
  estado: () => pedir("/api/health"),
  predecir: (texto) => pedir("/api/predict", json({ texto })),
  preprocesar: (texto) => pedir("/api/preprocess", json({ texto })),
  metricas: () => pedir("/api/metrics"),
  predecirArchivo: (archivo) => {
    const datos = new FormData();
    datos.append("archivo", archivo);
    return pedir("/api/predict/archivo", { method: "POST", body: datos });
  },
  historial: (limite = 50) => pedir(`/api/history?limite=${limite}`),
  feedback: (id, acertado) => pedir("/api/feedback", json({ id, acertado })),
  limpiarHistorial: () => pedir("/api/history", { method: "DELETE" }),
};

export const porcentaje = (valor, decimales = 1) => `${(valor * 100).toFixed(decimales)} %`;
