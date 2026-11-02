import { useEffect, useState } from "react";
import { api } from "./api.js";
import Analizar from "./componentes/Analizar.jsx";
import Archivo from "./componentes/Archivo.jsx";
import Modelo from "./componentes/Modelo.jsx";

const VISTAS = [
  { id: "analizar", nombre: "Analizar una reseña" },
  { id: "archivo", nombre: "Analizar un archivo" },
  { id: "modelo", nombre: "Cómo se entrenó" },
];

function Estado({ estado }) {
  if (!estado) return <span className="estado estado--cargando">Conectando con la API…</span>;
  if (estado.error) return <span className="estado estado--error">API sin conexión</span>;
  if (!estado.modelo_cargado) return <span className="estado estado--error">API sin modelo entrenado</span>;
  return <span className="estado estado--ok">{estado.modelo} en línea</span>;
}

export default function App() {
  const [vista, setVista] = useState(
    () => new URLSearchParams(window.location.search).get("vista") ?? "analizar"
  );
  const [estado, setEstado] = useState(null);

  useEffect(() => {
    api.estado().then(setEstado).catch((e) => setEstado({ error: e.message }));
  }, []);

  return (
    <div className="app">
      <header className="cabecera">
        <div className="marca">
          <span className="marca__nombre">CritiqAI</span>
          <span className="marca__lema">¿La reseña es positiva o negativa, y por qué?</span>
        </div>
        <Estado estado={estado} />
      </header>

      <nav className="pestanas" aria-label="Secciones">
        {VISTAS.map((v) => (
          <button
            key={v.id}
            className="pestana"
            aria-current={vista === v.id ? "page" : undefined}
            onClick={() => setVista(v.id)}
          >
            {v.nombre}
          </button>
        ))}
      </nav>

      <main>
        {vista === "analizar" && <Analizar />}
        {vista === "archivo" && <Archivo />}
        {vista === "modelo" && <Modelo />}
      </main>

      <footer className="pie">
        Reseñas de Amazon en español (amazon_reviews_multi). Proyecto integrador de Tópicos y
        Programación Avanzada.
      </footer>
    </div>
  );
}
